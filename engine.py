import enum
import gc
import re
import queue
import threading
import time
from llama_cpp import Llama
import os

physical_cores = os.cpu_count() // 2
llm_threads = max(2, physical_cores // 2)

_DISCALAIMER_PATTERNS = [
    re.compile(r"\n*\(?note[:\s].+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\(?disclaimer[:\s].+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\(?out of character[:\s].+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\(?OOC[:\s].+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\[note[:\s].+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\[disclaimer[:\s].+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\(?this is a fictional.+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\(?this conversation is fictional.+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\(?i'?m an? (AI|artificial|language model|computer|program|chatbot).+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\(?as an? (AI|artificial|language model|computer|program|chatbot).+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*\(?please remember.+fictional.+", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n*---+\s*.+", re.DOTALL),  # --- separator followed by meta-text
]

# Instruction/template artifacts that models leak into output
_ARTIFACT_RE = re.compile(
    r'\[/?(?:INST|CHAR|Response|Your response|Character Context|Current Message)\]'
    r'|<\|/?(?:assistant|user|system)\|>',
    re.IGNORECASE,
)

# Strip a leading "Name:" prefix (any capitalized name) at the very start of the response.
_LEADING_NAME_RE = re.compile(r'^[A-Z][a-z]+\s*[:,\-]\s*', re.MULTILINE)

# Detect when the model generates another speaker's turn mid-response.
# Matches "Name:" at start of a line OR inline after punctuation/quotes (e.g. '..."  Bob: "').
_MULTI_TURN_RE = re.compile(r'\n\s*[A-Z][a-z]+\s*:\s|[.!?"\'"]\s{2,}[A-Z][a-z]+\s*:\s')


def _sanitize_response(text):
    # Strip instruction artifacts
    text = _ARTIFACT_RE.sub('', text)
    # Strip leading name prefix (e.g. "Alice: " at the start)
    text = text.strip()
    text = _LEADING_NAME_RE.sub('', text, count=1)
    # Truncate at the first sign of another speaker's turn
    m = _MULTI_TURN_RE.search(text)
    if m:
        text = text[:m.start()]
    for pat in _DISCALAIMER_PATTERNS:
        text = pat.sub("", text)
    # Clean up leftover whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


_RULE_LINE_RE = re.compile(
    r'(?:^|\n)\s*[-•]\s*(?:do not|respond with only|never |stay in character|use no emojis)',
    re.IGNORECASE,
)
_SYSTEM_PROMPT_START_RE = re.compile(
    r'^you are [A-Z][a-z]',
    re.IGNORECASE,
)


def _is_valid_response(text: str) -> bool:
    """Return True if text is plausibly natural English.

    Checks (all must pass):
    1. Non-ASCII ratio <= 20%
    2. No single non-space char dominates > 35% of non-space chars
       (only applied when non-space char count > 20, case-insensitive)
    3. At least 3 whitespace-separated words
    4. Does not look like an echoed system prompt / rule list
    5. No verbatim sentence repeated within the response
    """
    if not text or len(text.split()) < 3:
        return False

    # Check 1: non-ASCII ratio
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.20:
        return False

    # Check 2: single-character dominance (case-insensitive, spaces excluded)
    non_space = [c.lower() for c in text if c != ' ']
    if len(non_space) > 20:
        from collections import Counter
        counts = Counter(non_space)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(non_space) > 0.35:
            return False

    # Check 4: reject echoed system prompts / rule lists
    if _RULE_LINE_RE.search(text):
        return False
    if _SYSTEM_PROMPT_START_RE.match(text.strip()):
        return False

    # Check 5: verbatim sentence repetition (within-response loop)
    sentences = [s.strip().lower() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 8]
    if len(sentences) != len(set(sentences)):
        return False

    return True


class ModelSlot(enum.Enum):
    NONE = "none"
    MAIN = "main"
    SMALL = "small"


# Minimum seconds between swaps away from MAIN to prevent thrashing
_MAIN_COOLDOWN_SECS = 5.0


class LLMEngine:

    @staticmethod
    def _build_messages(system_prompt, messages):
        """Inject system prompt into first user message for Mistral compatibility.

        Mistral 7B Instruct v0.2 does not support role: "system" — it silently
        drops it.  Instead we prepend the system prompt to the first user message
        so the model actually sees the character context.

        Consecutive messages with the same role are merged (newline-joined) to
        satisfy models that require strictly alternating user/assistant turns.
        This happens naturally when multiple other bots speak back-to-back —
        each maps to role "user" in the listener's history.
        """
        # Strip leading assistant entries — they indicate a reconnect where the
        # bot was the opener and prior context is missing. Leaving them first
        # produces [assistant, user, ...] which fails Mistral chat templates.
        while messages and messages[0]["role"] == "assistant":
            messages = messages[1:]
        merged = []
        system_injected = False
        for msg in messages:
            if not system_injected and msg["role"] == "user":
                merged.append({
                    "role": "user",
                    "content": (
                        f"[Character Context]\n{system_prompt}\n\n"
                        f"[Current Message]\n{msg['content']}"
                    ),
                })
                system_injected = True
            elif merged and merged[-1]["role"] == msg["role"]:
                # Merge consecutive same-role messages to keep alternating turns
                merged[-1]["content"] += "\n" + msg["content"]
            else:
                merged.append(msg)
        # If no user message existed yet, create one with a kickoff so the model
        # doesn't echo the context back verbatim.
        if not system_injected:
            merged.insert(0, {
                "role": "user",
                "content": f"[Character Context]\n{system_prompt}\n\n[Current Message]\nBegin speaking.",
            })
        return merged

    def __init__(self, model_path, n_ctx=4096, n_threads=llm_threads, small_model_path=None, n_batch=512):
        self.model_path = model_path
        self.small_model_path = small_model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch

        # Hot-swap state: only one model loaded at a time
        self._active_model = None
        self._active_slot = ModelSlot.NONE
        self._active_path = None        # path of currently loaded model (for path-based swap)
        self._last_main_use = 0.0  # timestamp of last MAIN model use

        self._thread = None
        self._token_queue = queue.Queue()
        self._generating = False
        self._full_response = ""
        self._error = None
        self._model_lock = threading.Lock()  # Prevents concurrent model access
        self._preload_thread = None
        self._image_worker = None  # SD worker reference for GPU coordination

    @staticmethod
    def _detect_gpu_layers():
        """Detect GPU availability and return n_gpu_layers setting."""
        try:
            import torch
            if torch.cuda.is_available():
                print("GPU detected: CUDA — offloading all layers to GPU")
                return -1
            if torch.backends.mps.is_available():
                print("GPU detected: Metal (Apple Silicon) — offloading all layers to GPU")
                return -1
        except ImportError:
            pass

        # Check for Metal without torch (llama-cpp has built-in Metal support)
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("GPU detected: Apple Silicon — offloading all layers to Metal")
            return -1

        print("No GPU detected — running LLM on CPU")
        return 0

    def _load_llm(self, model_path, n_ctx):
        """Load a GGUF model with optimal settings."""
        n_gpu_layers = self._detect_gpu_layers()
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=self.n_batch,
            use_mmap=True,
            use_mlock=False,
            flash_attn=(n_gpu_layers != 0),  # GPU only — not supported on all CPU builds
            verbose=False,
        )

    def _unload_active(self):
        """Unload the currently active model and free memory."""
        if self._active_model is not None:
            del self._active_model
            self._active_model = None
            self._active_slot = ModelSlot.NONE
            self._active_path = None
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, AttributeError):
                pass

    def set_image_worker(self, worker):
        """Store reference to SD image worker for GPU coordination."""
        self._image_worker = worker

    def _load_slot(self, slot):
        """Unload current model and load the requested slot."""
        if slot == self._active_slot:
            return
        was_main = self._active_slot == ModelSlot.MAIN
        loading_main = slot == ModelSlot.MAIN
        # Only pause SD worker when loading the large MAIN model
        if loading_main and self._image_worker is not None:
            self._image_worker.pause()
        self._unload_active()
        if slot == ModelSlot.MAIN:
            print(f"Hot-swap: loading MAIN model ({os.path.basename(self.model_path)})")
            self._active_model = self._load_llm(self.model_path, self.n_ctx)
            self._active_slot = ModelSlot.MAIN
        elif slot == ModelSlot.SMALL:
            path = self.small_model_path
            if path and os.path.exists(path):
                print(f"Hot-swap: loading SMALL model ({os.path.basename(path)})")
                self._active_model = self._load_llm(path, 2048)
                self._active_slot = ModelSlot.SMALL
            else:
                # No small model available — load main as fallback
                print("Hot-swap: no small model, falling back to MAIN")
                self._active_model = self._load_llm(self.model_path, self.n_ctx)
                self._active_slot = ModelSlot.MAIN
        # Resume SD worker when switching away from MAIN to SMALL
        if was_main and not loading_main and self._image_worker is not None:
            self._image_worker.resume()

    def ensure_model(self, slot):
        """Ensure the requested model slot is loaded. Must be called under _model_lock."""
        if slot == self._active_slot:
            return
        self._load_slot(slot)

    def request_model(self, slot):
        """Non-blocking background pre-load. Acquires lock in a background thread."""
        if slot == self._active_slot:
            return
        # Don't pile up preload threads
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return

        def _preload():
            with self._model_lock:
                self._load_slot(slot)

        self._preload_thread = threading.Thread(target=_preload, daemon=True)
        self._preload_thread.start()

    @property
    def last_main_use(self):
        """Timestamp of last MAIN model usage."""
        return self._last_main_use

    @property
    def _main_model(self):
        """Ensure MAIN is loaded and return it."""
        self.ensure_model(ModelSlot.MAIN)
        self._last_main_use = time.time()
        return self._active_model

    @property
    def _bg_model(self):
        """Ensure SMALL is loaded and return it."""
        self.ensure_model(ModelSlot.SMALL)
        return self._active_model

    def load_model(self):
        """Startup: only load the SMALL model. MAIN loads on-demand."""
        if self.small_model_path and os.path.exists(self.small_model_path):
            print(f"Loading small model at startup: {os.path.basename(self.small_model_path)}")
            self._active_model = self._load_llm(self.small_model_path, 2048)
            self._active_slot = ModelSlot.SMALL
        else:
            print("No small model — loading main model at startup")
            self._active_model = self._load_llm(self.model_path, self.n_ctx)
            self._active_slot = ModelSlot.MAIN

    def is_loaded(self) -> bool:
        return self._active_model is not None

    def is_generating(self) -> bool:
        return self._generating

    def start_generation(self, system_prompt, messages):
        if self._generating: # Already generating, don't start another
            return
        self._generating = True
        self._full_response = ""
        self._error = None
        # Drain any leftover tokens from previous generation
        while not self._token_queue.empty():
            try:
                self._token_queue.get_nowait()
            except queue.Empty:
                break
        # Spawn background thread
        self._thread = threading.Thread(
            target=self._generate_worker,
            args=(system_prompt, messages),
            daemon=True,
        )
        self._thread.start()

    def _generate_worker(self, system_prompt, messages): # Runs on background thread
        try:
            with self._model_lock:
                model = self._main_model  # ensures MAIN is loaded
                full_messages = self._build_messages(system_prompt, messages)
                stream = model.create_chat_completion(
                    messages=full_messages,
                    max_tokens=128,   # 1-3 sentences only need ~60-80 tokens
                    temperature=0.8,
                    top_p=0.95,
                    stream=True,
                )
                for chunk in stream:
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        self._full_response += token
                        self._token_queue.put(token)
            self._token_queue.put(None) # Sentinel: generation done
        except Exception as e:
            self._error = e
            self._token_queue.put(None)
        finally:
            self._generating = False
            # Resume SD worker so it can generate between conversation turns
            if self._image_worker is not None:
                self._image_worker.resume()

    def poll_tokens(self) -> list: # Non-blocking drain, called every frame
        tokens = []
        while not self._token_queue.empty():
            try:
                tokens.append(self._token_queue.get_nowait())
            except queue.Empty:
                break
        return tokens

    def get_full_response(self) -> str:
        return _sanitize_response(self._full_response)

    def generate_for_path(self, model_path, system_prompt, messages, n_ctx=None, max_tokens=128, temperature=0.8, seed=None):
        """Blocking generation, hot-swapping to model_path if not already loaded."""
        with self._model_lock:
            if self._active_path != model_path:
                if self._image_worker is not None:
                    self._image_worker.pause()
                self._unload_active()
                print(f"Hot-swap: loading {os.path.basename(model_path)}")
                self._active_model = self._load_llm(model_path, n_ctx or self.n_ctx)
                self._active_path = model_path
                self._active_slot = ModelSlot.NONE
            msgs = self._build_messages(system_prompt, messages)
            kwargs = dict(messages=msgs, max_tokens=max_tokens, temperature=temperature, top_p=0.95)
            if seed is not None:
                kwargs["seed"] = seed
            result = self._active_model.create_chat_completion(**kwargs)
        return result["choices"][0]["message"]["content"].strip()

    def generate_reply(self, slot, system_prompt, messages, max_tokens=128, temperature=0.8, seed=None):
        """Blocking single-turn generation using the given ModelSlot."""
        with self._model_lock:
            self.ensure_model(slot)
            if slot == ModelSlot.MAIN:
                self._last_main_use = time.time()
            model = self._active_model
            msgs = self._build_messages(system_prompt, messages)
            kwargs = dict(
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            )
            if seed is not None:
                kwargs["seed"] = seed
            result = model.create_chat_completion(**kwargs)
        return result["choices"][0]["message"]["content"].strip()

    def evaluate_interaction(self, system_prompt, user_message):
        """Blocking call to rate an interaction. Returns int in [-3, +3]. Uses small model."""
        try:
            with self._model_lock:
                model = self._bg_model
                result = model.create_chat_completion(
                    messages=self._build_messages(system_prompt, [
                        {"role": "user", "content": user_message},
                    ]),
                    max_tokens=10,
                    temperature=0.3,
                    )
            text = result["choices"][0]["message"]["content"].strip()
            match = re.search(r"-?\d+", text)
            if match:
                return max(-3, min(3, int(match.group())))
        except Exception:
            pass
        return 0

    def summarize_conversation(self, system_prompt, user_message):
        """Blocking call to summarize conversation exchanges. Uses small model."""
        try:
            with self._model_lock:
                model = self._bg_model
                result = model.create_chat_completion(
                    messages=self._build_messages(system_prompt, [
                        {"role": "user", "content": user_message},
                    ]),
                    max_tokens=150,
                    temperature=0.3,
                )
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    def identify_key_moments(self, system_prompt, user_message):
        """Blocking call to identify key moments. Returns raw text. Uses small model."""
        try:
            with self._model_lock:
                model = self._bg_model
                result = model.create_chat_completion(
                    messages=self._build_messages(system_prompt, [
                        {"role": "user", "content": user_message},
                    ]),
                    max_tokens=100,
                    temperature=0.3,
                )
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            return "NONE"

    def generate_npc_profile(self, system_prompt, user_prompt):
        """Blocking LLM call to generate an NPC profile as JSON. Returns raw text. Uses small model."""
        try:
            with self._model_lock:
                model = self._bg_model
                result = model.create_chat_completion(
                    messages=self._build_messages(system_prompt, [
                        {"role": "user", "content": user_prompt},
                    ]),
                    max_tokens=400,
                    temperature=0.9,
                )
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    def try_acquire_lock(self):
        """Non-blocking lock attempt. Returns True if lock acquired."""
        return self._model_lock.acquire(blocking=False)

    def release_lock(self):
        """Release the model lock."""
        self._model_lock.release()

    def get_error(self):
        return self._error
