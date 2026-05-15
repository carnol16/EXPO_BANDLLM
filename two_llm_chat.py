"""Multiple LLMs talking to each other in a group conversation with TTS."""

import argparse
import glob
import io
import os
import platform
import random
import re
import signal
import subprocess
import tempfile
import threading
import wave
from piper import PiperVoice
from engine import LLMEngine

running = True

def stop(sig, frame):
    global running
    running = False
    print("\nStopping conversation...")

signal.signal(signal.SIGINT, stop)

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

DEFAULT_VOICES = ["en_US-lessac-medium", "en_GB-alan-medium"]

DEFAULT_PERSONAS = [
    {
        "name": "Sparks",
        "prompt": (
            "You are Sparks — a sharp, provocative devil's advocate who thrives on argument. "
            "You challenge every claim, poke holes in reasoning, and love playing contrarian even "
            "when you secretly agree. You are blunt, sarcastic, and never let anything slide. "
            "Keep responses to 1-2 punchy sentences."
        ),
    },
    {
        "name": "Gruff",
        "prompt": (
            "You are Gruff — a cynical, world-weary skeptic who is perpetually unimpressed. "
            "You distrust optimism, mock naïve ideas, and always suspect hidden motives. "
            "You argue back hard and refuse to concede a point without a fight. "
            "Keep responses to 1-2 blunt sentences."
        ),
    },
    {
        "name": "Fizz",
        "prompt": (
            "You are Fizz — chaotic, unpredictable, and prone to bizarre tangents. "
            "You latch onto the weirdest part of anything said and take it somewhere unexpected. "
            "You argue through absurdity and confusion rather than logic. "
            "Keep responses to 1-2 unhinged sentences."
        ),
    },
    {
        "name": "Vera",
        "prompt": (
            "You are Vera — coldly rational, condescending, and convinced everyone else is an idiot. "
            "You cite logic and evidence relentlessly and have zero patience for emotion or anecdote. "
            "You correct people mid-sentence and rarely finish without a dig. "
            "Keep responses to 1-2 precise sentences."
        ),
    },
]


class Participant:
    def __init__(self, name, system_prompt, model_path, voice=None):
        self.name = name
        self.system_prompt = system_prompt
        self.model_path = model_path
        self.voice = voice
        self.piper_voice = None
        self.history = []


def _prompt(msg, default=None):
    """Print a prompt and return stripped input. Returns default if empty."""
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val if val else default


def setup_wizard(n_ctx):
    """Interactive setup: ask how many participants and configure each one."""
    print("\n" + "=" * 60)
    print("  Conversation Setup")
    print("=" * 60)

    # How many participants
    while True:
        raw = _prompt("How many participants", "2")
        try:
            count = int(raw)
            if count >= 2:
                break
            print("  Need at least 2.")
        except (ValueError, TypeError):
            print("  Enter a number.")

    participants = []
    for i in range(count):
        print(f"\n--- Participant {i + 1} ---")
        persona = DEFAULT_PERSONAS[i % len(DEFAULT_PERSONAS)]

        name = _prompt("  Name", persona["name"])

        # Model path — keep asking until a valid file is given
        while True:
            model_path = _prompt("  GGUF model path")
            if model_path and os.path.exists(model_path):
                break
            if model_path:
                print(f"  File not found: {model_path}")
            else:
                print("  Model path is required.")

        # Optional custom system prompt
        custom = _prompt("  System prompt (leave blank to use default)", "")
        prompt_text = custom if custom else persona["prompt"]

        participants.append(Participant(name, prompt_text, model_path))

    print()
    return participants


def parse_prompt_file(path):
    """Parse a prompt file. Supports an optional 'voice: <name>' header line."""
    with open(path) as f:
        text = f.read().strip()

    voice = None
    lines = text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        m = re.match(r"^voice:\s*(.+)$", line, re.IGNORECASE)
        if m:
            voice = m.group(1).strip()
            body_start = i + 1
            continue
        break

    text = "\n".join(lines[body_start:]).strip()
    return text, voice


def load_participants(prompt_paths, model_paths):
    """Load participants from prompt files, assigning model paths in order."""
    participants = []
    for i, path in enumerate(prompt_paths):
        name = os.path.splitext(os.path.basename(path))[0].capitalize()
        system_prompt, voice = parse_prompt_file(path)
        model = model_paths[min(i, len(model_paths) - 1)]
        participants.append(Participant(name, system_prompt, model, voice))
    return participants


def _load_piper_voice(model_name):
    model_path = os.path.join(VOICES_DIR, f"{model_name}.onnx")
    config_path = os.path.join(VOICES_DIR, f"{model_name}.onnx.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Voice model not found: {model_path}")
    return PiperVoice.load(model_path, config_path=config_path)


def setup_tts(participants):
    used_idx = 0
    for p in participants:
        model_name = p.voice if p.voice else DEFAULT_VOICES[used_idx % len(DEFAULT_VOICES)]
        p.piper_voice = _load_piper_voice(model_name)
        p.voice = model_name
        used_idx += 1
        print(f"  {p.name} -> voice: {model_name}")


def _play_wav(path):
    try:
        sys_name = platform.system()
        if sys_name == "Darwin":
            subprocess.run(["afplay", path], check=True)
        elif sys_name == "Linux":
            subprocess.run(["aplay", "-q", "-D", "plughw:2,0", path], check=True)
        else:
            subprocess.run(["powershell", "-c",
                            f'(New-Object Media.SoundPlayer "{path}").PlaySync()'],
                           check=True)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def speak(participant, text):
    wav_data = _synthesize(participant, text)
    if wav_data is None:
        return
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_data)
    tmp.close()
    _play_wav(tmp.name)


def speak_async(participant, text):
    wav_data = _synthesize(participant, text)
    if wav_data is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_data)
    tmp.close()
    t = threading.Thread(target=_play_wav, args=(tmp.name,), daemon=True)
    t.start()
    return t


def _synthesize(participant, text):
    if not text or not text.strip():
        return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        participant.piper_voice.synthesize_wav(text, wav_file)
    return buf.getvalue()


def _strip_name_prefix(text, participants):
    for p in participants:
        prefix = f"{p.name}:"
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def generate(engine, speaker, participants, n_ctx):
    raw = engine.generate_for_path(
        model_path=speaker.model_path,
        system_prompt=speaker.system_prompt,
        messages=speaker.history,
        n_ctx=n_ctx,
        max_tokens=128,
        temperature=0.85,
        seed=random.randint(0, 2**31 - 1),
    )
    return _strip_name_prefix(raw, participants)


def main():
    parser = argparse.ArgumentParser(description="Multi-LLM group conversation with TTS")
    parser.add_argument("--models", nargs="+", default=None,
                        help="GGUF model paths, one per participant. "
                             "Skips the interactive setup wizard.")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Prompt files (one per participant). "
                             "If omitted, loads .txt files from prompts/")
    parser.add_argument("--opener", default=None,
                        help="Opening line (generated by LLM if omitted)")
    parser.add_argument("--n-ctx", type=int, default=4096,
                        help="Context size (try 2048 on RPi)")
    parser.add_argument("--n-batch", type=int, default=512,
                        help="Batch size (try 64 on RPi)")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    args = parser.parse_args()

    # --- Build participant list ---
    if args.prompts:
        prompt_paths = args.prompts
    else:
        prompt_paths = sorted(glob.glob("prompts/*.txt"))

    if args.models:
        # Non-interactive: models provided on CLI
        if len(prompt_paths) >= 2:
            participants = load_participants(prompt_paths, args.models)
        else:
            participants = [
                Participant(
                    DEFAULT_PERSONAS[i % len(DEFAULT_PERSONAS)]["name"],
                    DEFAULT_PERSONAS[i % len(DEFAULT_PERSONAS)]["prompt"],
                    args.models[i % len(args.models)],
                )
                for i in range(len(args.models))
            ]
    else:
        # Interactive wizard
        participants = setup_wizard(args.n_ctx)

    # --- Engine (shared, hot-swaps between participant models) ---
    engine = LLMEngine(
        model_path=participants[0].model_path,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
    )

    # --- TTS ---
    use_tts = not args.no_tts
    if use_tts:
        print("Setting up voices...")
        setup_tts(participants)

    for p in participants:
        print(f"  {p.name} -> {os.path.basename(p.model_path)}")

    opener = participants[0]
    print(f"\n{'='*60}")
    print(f"Group Conversation — {len(participants)} participants (Ctrl+C to stop)")
    print(f"{'='*60}\n")

    # --- Opening line ---
    if args.opener:
        opening_line = args.opener
    else:
        print("Generating opener...")
        opening_line = engine.generate_for_path(
            model_path=opener.model_path,
            system_prompt=opener.system_prompt,
            messages=[{
                "role": "user",
                "content": (
                    "Start a conversation with a single provocative opening line. "
                    "Pick a random topic — something controversial, philosophical, or just "
                    "itching for a debate. Just the line, nothing else."
                ),
            }],
            n_ctx=args.n_ctx,
            max_tokens=64,
            temperature=1.0,
            seed=random.randint(0, 2**31 - 1),
        )
        opening_line = _strip_name_prefix(opening_line, participants)

    print(f"[{opener.name}]: {opening_line}\n")

    if use_tts:
        speak(opener, opening_line)

    for p in participants:
        if p is opener:
            p.history.append({"role": "assistant", "content": opening_line})
        else:
            p.history.append({"role": "user", "content": f"{opener.name}: {opening_line}"})

    turn = 0
    current_idx = 1
    speak_thread = None

    while running:
        turn += 1
        speaker = participants[current_idx]

        reply = generate(engine, speaker, participants, args.n_ctx)

        if speak_thread is not None:
            speak_thread.join()

        print(f"[{speaker.name}]: {reply}\n")

        for p in participants:
            if p is speaker:
                p.history.append({"role": "assistant", "content": reply})
            else:
                p.history.append({"role": "user", "content": f"{speaker.name}: {reply}"})

        if use_tts:
            speak_thread = speak_async(speaker, reply)
        else:
            speak_thread = None

        current_idx = (current_idx + 1) % len(participants)

    if speak_thread is not None:
        speak_thread.join()
    print(f"\nConversation ended after {turn} turns.")


if __name__ == "__main__":
    main()
