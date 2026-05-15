# Startup Guide — Four-Bot Conversation System

## Prerequisites

- Python 3.12+
- Models downloaded as `.gguf` files and placed in `models/`
- (Optional) Piper voice files in `voices/` — skip with `--no-tts` if not available
- Dependencies installed: `pip install -r requirements.txt`

---

## Recommended Models

All four bots use modern uncensored/abliterated 7B–8B models from different architectures. The coordinator uses a small uncensored model (CPU-only on CUDA, Metal on Apple Silicon).

| Role | Model | File | Size | Why |
|---|---|---|---|---|
| **Coordinator** | Dolphin 3.0 Llama 3.2 3B Q5_K_M | `Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf` | ~2.2GB | Small uncensored model — only outputs one name per turn |
| **Alice** (Skeptic) | Llama 3.1 8B Instruct Abliterated Q4_K_M | `meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf` | ~4.6GB | Abliterated Llama 3.1 — sharp reasoning, no refusals |
| **Bob** (Egomaniac) | OpenHermes 2.5 Mistral 7B Q4_K_M | `openhermes-2.5-mistral-7b.Q4_K_M.gguf` | ~4.1GB | Uncensored Mistral fine-tune — aggressive and unfiltered |
| **Charlie** (Chaos agent) | Qwen3 8B Hivemind Heretic Abliterated Q4_K_M | `Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf` | ~4.3GB | Triple-uncensored Qwen3 — zero guardrails, maximum chaos |
| **Diana** (Intellectual) | Dolphin 2.6 Mistral 7B Q4_K_M | `dolphin-2.6-mistral-7b.Q4_K_M.gguf` | ~3.8GB | Dolphin — Eric Hartford's fully uncensored Mistral series |

> **Download:** All bot models are abliterated/heretic variants available on Hugging Face. Search for the model name + "abliterated" or "heretic" + "GGUF".
>
> **RAM check (single-machine test):** 4 bots ≈ 18.4GB DDR5 + 2.2GB coordinator ≈ 20.6GB — within your 32GB. GPU layers are allocated dynamically by `hardware.py` based on available VRAM.

### Voice Assignments

Each bot uses a distinct Piper TTS voice, set in the `voice:` header of its prompt file.

| Bot | Voice | Accent |
|---|---|---|
| Alice | `en_US-lessac-medium` | American female |
| Bob | `en_GB-alan-medium` | British male |
| Charlie | `en_US-amy-medium` | American female (different tone) |
| Diana | `en_GB-northern_english_male-medium` | British male (northern) |

> Alice and Bob's voices are pre-installed. Charlie and Diana's voices must be downloaded into `voices/`:
> ```
> curl -L -o voices/en_US-amy-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true"
> curl -L -o voices/en_US-amy-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true"
> curl -L -o voices/en_GB-northern_english_male-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx?download=true"
> curl -L -o voices/en_GB-northern_english_male-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json?download=true"
> ```

---

## Activate Virtual Environment

Run this in every terminal window before any other command.

**Windows (Command Prompt or PowerShell):**
```
.venv\Scripts\activate
```

**Linux / Mac:**
```
source .venv/bin/activate
```

---

## Option A: Single Machine Test (i9/RTX 3080) — No Audio

Run all 5 processes on one machine. Use `--no-tts` to skip audio. GPU layers are allocated automatically — the coordinator queries free VRAM, divides it across all bots, and sends each bot its optimal `n_gpu_layers` at load time.

Open **5 terminal windows** in `F:/mtechProj/` and activate the virtual environment in each.

---

### Terminal 1 — Coordinator

```
python coordinator.py --model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf --bots 4
```

**Wait here.** The coordinator will print:
```
Coordinator listening on ws://0.0.0.0:8765
Waiting for 4 bots to connect...
```

Do not proceed until all 4 bots are connected (see below).

---

### Terminal 2 — Bot: Alice

```
python bot.py --name Alice --prompt prompts/alice.txt --model models/meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf --coordinator ws://localhost:8765 --language fr
```

---

### Terminal 3 — Bot: Bob

```
python bot.py --name Bob --prompt prompts/bob.txt --model models/openhermes-2.5-mistral-7b.Q4_K_M.gguf --coordinator ws://localhost:8765 --language es
```

---

### Terminal 4 — Bot: Charlie

```
python bot.py --name Charlie --prompt prompts/charlie.txt --model models/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf --coordinator ws://localhost:8765  --language ko
```

---

### Terminal 5 — Bot: Diana

```
python bot.py --name Diana --prompt prompts/diana.txt --model models/dolphin-2.6-mistral-7b.Q4_K_M.gguf --coordinator ws://localhost:8765 --language en
```

---

### What happens next

Once all 4 bots have connected and loaded, the coordinator will print:

```
============================================================
All 4 bots ready: Alice, Bob, Charlie, Diana
============================================================

Generating opening line...
[Alice]: ...
```

The conversation runs automatically. **Type anything in Terminal 1 and press Enter** to inject a topic change mid-conversation.

Press **Ctrl+C** in Terminal 1 to stop.

---

## Option B: Manual GPU Override

GPU layers are allocated automatically in Option A. If you need to override the allocation (e.g., to force fewer layers due to VRAM pressure), pass `--gpu-layers N` to individual bots:

```
python bot.py --name Alice --prompt prompts/alice.txt --model models/meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf --coordinator ws://localhost:8765 --gpu-layers 8 --no-tts --language fr
```

The `--gpu-layers` flag on the bot overrides whatever the coordinator calculated. Use `--gpu-layers 0` to force CPU-only on a specific bot.

> **Note:** On a 10GB card with 4 bots, the dynamic allocator already accounts for KV cache overhead. Only use manual overrides if you see VRAM errors.

---

## Option C: Production Deployment (Mac mini + 4 Raspberry Pis)

### On the Mac mini — Coordinator

```
python coordinator.py --model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf --bots 4 --host 0.0.0.0
```

Note the Mac mini's local IP address (e.g., `192.168.1.10`).

---

### On each Raspberry Pi — One Bot per Pi

Copy these files to each Pi:
- `bot.py`
- `engine.py`
- `hardware.py`
- `prompts/<bot-name>.txt`
- `models/<your-model>.gguf`
- `voices/<voice-name>.onnx` and `.onnx.json` (if using TTS)
- `requirements.txt`

Then install and run:

```
pip install -r requirements.txt
```

**Pi 1 — Alice:**
```
python bot.py --name Alice --prompt prompts/alice.txt --model models/meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf --coordinator ws://192.168.0.144:8765 --language fr
```

**Pi 2 — Bob:**
```
python bot.py --name Bob --prompt prompts/bob.txt --model models/openhermes-2.5-mistral-7b.Q4_K_M.gguf --coordinator ws://192.168.0.144:8765 --language es
```

**Pi 3 — Charlie:**
```
python bot.py --name Charlie --prompt prompts/charlie.txt --model models/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf --coordinator ws://192.168.0.144:8765 --language ko
```

**Pi 4 — Diana:**
```
python bot.py --name Diana --prompt prompts/diana.txt --model models/dolphin-2.6-mistral-7b.Q4_K_M.gguf --coordinator ws://192.168.0.144:8765 --language en
```

Hardware is auto-detected on each Pi — no `--gpu-layers` flag needed.

---

## All CLI Flags

### coordinator.py

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Path to moderator GGUF model |
| `--bots` | required | Number of bots to wait for |
| `--host` | `0.0.0.0` | Interface to listen on |
| `--port` | `8765` | WebSocket port |
| `--register-timeout` | `300` | Seconds to wait for all bots to register |
| `--reply-timeout` | `120` | Seconds to wait for a bot reply before skipping |
| `--gpu-layers` | `0` on CUDA, `-1` on Metal | Override n_gpu_layers for moderator model |
| `--opener-prompt` | built-in debate prompt | Custom opening prompt. Use `{name}` for the speaker's name |

### bot.py

| Flag | Default | Description |
|---|---|---|
| `--name` | required | Bot name (e.g., Alice) |
| `--prompt` | required | Path to personality prompt file |
| `--model` | required | Path to GGUF model file |
| `--coordinator` | `ws://localhost:8765` | Coordinator WebSocket URL |
| `--gpu-layers` | auto | Override n_gpu_layers |
| `--no-tts` | off | Disable TTS (for testing without audio) |
| `--language` | `en` | ISO 639-1 language code for LLM responses and Whisper transcription (e.g. `es`, `fr`, `de`, `ja`) |

---

## Creating a New Bot Personality

Add a new `.txt` file to `prompts/`. The format is:

```
voice: en_US-lessac-medium
You are [Name] — [personality description].
Rules:
- Keep responses to 1-2 sentences.
- Use no emojis.
- Stay in character at all times.
- Never break the fourth wall or mention being an AI.
```

The `voice:` line is optional. If omitted, a default voice is assigned. Available voices depend on what's installed in `voices/`.

Then launch the bot the same way as the others, pointing `--prompt` at your new file.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Bot says `File not found: voices/...` | Add `--no-tts` or download the voice file |
| Coordinator exits after timeout | Start bots faster, or increase `--register-timeout 600` |
| VRAM out of memory | Reduce `--gpu-layers` per bot, or use `--gpu-layers 0` |
| Bot name already registered | Each bot must have a unique `--name` |
| Conversation stops after one bot disconnects | Restart that bot — reconnection is not yet supported |
| `address already in use` on port 8765 | Kill stale processes: `lsof -ti :8765 \| xargs kill` |
| Bots disconnect while others are loading | WebSocket ping timeouts are set to 120s — increase `--register-timeout` if models take longer |
| Opening line is always the same | Pass `--opener-prompt "Your custom prompt here. Spoken by {name}."` |
| `Failed to load model from file` | The model uses the `i1-Q4_K_M` imatrix tensor format, which the current llama-cpp-python build may not support. Use a standard `Q4_K_M` model instead (no `i1` prefix in the filename). |
| `Encountered unknown tag 'break'` | The model's Jinja2 chat template uses `{% break %}`, which is not supported by the Jinja2 version bundled with llama-cpp-python. Affects `c4ai-command-r7b` and some other newer models. Switch to a different model. |
