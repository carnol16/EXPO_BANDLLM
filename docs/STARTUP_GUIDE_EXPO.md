# The Band — Startup Guide

## Hardware

| Machine | Role |
|---|---|
| i9 / RTX 3080 (10GB VRAM, 32GB DDR5) | Single-machine — all components in one terminal |
| Mac mini M2 Pro + 4× Raspberry Pi 5 | Multi-machine — coordinator on Mac mini, one bot per Pi |

`band_coordinator.py` supports both modes. Use `--remote-bots` to run coordinator + DayArc only and let bots connect from other machines.

---

## Quick Launch

If the environment is already set up, this is the full gallery launch in one line:

```powershell
python band_coordinator.py --moderator-model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf --singer-model models/meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf --guitarist-model models/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf --bassist-model models/c4ai-command-r7b-12-2024-abliterated-Q4_K_M.gguf --drummer-model models/mistralai_Ministral-3-8B-Instruct-2512-abliterated.i1-Q4_K_M.gguf --small-model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf
```

Add `--fast` for a test run (~2 hours for the full week). Add `--no-tts` to skip audio entirely.

---

## Step 1 — Activate the virtual environment

**Windows (PowerShell):**
```powershell
cd f:\EXPO_BANDLLM
.venv\Scripts\activate
```

**Mac:**
```bash
cd /Volumes/ColtonWork/EXPO_BANDLLM
source .venv/bin/activate
```

---

## Step 2 — Install Python dependencies

Install all dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

**Critical — llama-cpp-python must be built with CUDA support.** The line in `requirements.txt` installs a CPU-only build by default. Override it:

**Windows (PowerShell) — single line, no backslash:**
```powershell
pip install llama-cpp-python --force-reinstall --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**Mac/Linux (bash):**
```bash
pip install llama-cpp-python --force-reinstall \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

> **Windows note:** The `\` line continuation is bash-only and will fail in PowerShell. Always run the CUDA install as a single line in PowerShell.

> **CUDA version:** The `cu124` wheel targets CUDA 12.4 but works fine on CUDA 13.x — the driver is backward-compatible.

Verify GPU support loaded correctly — you should see `CUDA detected` at startup, not the CPU fallback warning.

> **Mac (Apple Silicon):** Skip the CUDA override. The standard `pip install -r requirements.txt` is sufficient — Metal is detected automatically.

---

## Step 3 — Install ACE-Step V1.5

ACE-Step generates the music tracks each night. The system detects whichever installation method you have.

> **Important:** ACE-Step 1.5 uses `uv` as its package manager — `pip install -e .` does **not** work. The `nano-vllm` dependency is a local path only resolvable by uv, and pip will error trying to find it on PyPI.

**One-time setup — install uv (if not already installed):**
```powershell
cd f:\EXPO_BANDLLM\ACE-Step-1.5
.\install_uv.bat
```
Restart your terminal after installation so `uv` is on the PATH.

**Option A — REST API (recommended):**

Run in a separate terminal and leave it running before launching the band:
```powershell
cd f:\EXPO_BANDLLM\ACE-Step-1.5
.\start_api_server.bat
```
On first run this downloads ~10GB of model weights. The API will be at `http://127.0.0.1:8001`.

**Option B — Python import (no separate server):**
```powershell
cd f:\EXPO_BANDLLM\ACE-Step-1.5
uv sync
```
The bridge will import ACE-Step directly.

> **Mac:** Use `./install_uv.sh` and `./start_api_server_macos.sh` instead of the `.bat` files.

---

## Step 4 — Assign voices

Run the interactive voice setup. This plays a sample for each voice so you can audition them, then writes `voice_config.json`.

```bash
python band_setup.py --assign-voices
```

You will be asked to pick a Kokoro voice ID and a vocal style description for each of the four bots (singer, guitarist, bassist, drummer). The vocal style description is passed to ACE-Step for the music generation (e.g. `"intense male vocals, raw, urgent"`).

To see all available Kokoro voices without committing:
```bash
python -c "import kokoro; print('\n'.join(kokoro.list_voices()))"
```

---

## Step 5 — Preflight check

```bash
python band_setup.py --check
```

This verifies: prompt files, `voice_config.json`, ACE-Step, Kokoro, and the `output/` directory. Fix anything marked `[FAIL]` before continuing.

> **Note:** The model check will show `[FAIL] model_config.json — file not found`. This is expected — models are now passed as CLI args, not a config file. Ignore that specific failure; everything else counts.

---

## Step 6 — Launch the installation

### Full installation (real timing)

Sessions run 8am–midnight each day, bots sleep 2:30–3:30am, Thursday listening party at 7pm. Use this for the actual gallery run.

```bash
python band_coordinator.py \
  --moderator-model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf \
  --singer-model    models/meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf \
  --guitarist-model models/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf \
  --bassist-model   models/c4ai-command-r7b-12-2024-abliterated-Q4_K_M.gguf \
  --drummer-model   models/mistralai_Ministral-3-8B-Instruct-2512-abliterated.i1-Q4_K_M.gguf \
  --small-model     models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf
```

### Fast mode (test — ~20 minutes per day, full week in ~2 hours)

```bash
python band_coordinator.py \
  --moderator-model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf \
  --singer-model    models/meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf \
  --guitarist-model models/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf \
  --bassist-model   models/c4ai-command-r7b-12-2024-abliterated-Q4_K_M.gguf \
  --drummer-model   models/mistralai_Ministral-3-8B-Instruct-2512-abliterated.i1-Q4_K_M.gguf \
  --small-model     models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf \
  --fast
```

### Silent test (no audio — fastest way to verify the conversation loop)

```bash
python band_coordinator.py \
  --moderator-model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf \
  --singer-model    models/meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf \
  --guitarist-model models/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf \
  --bassist-model   models/c4ai-command-r7b-12-2024-abliterated-Q4_K_M.gguf \
  --drummer-model   models/mistralai_Ministral-3-8B-Instruct-2512-abliterated.i1-Q4_K_M.gguf \
  --small-model     models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf \
  --fast --no-tts
```

### Multi-machine (coordinator + DayArc on one machine, bots on separate machines)

**On the coordinator machine** — no bot model args needed:

```bash
python band_coordinator.py \
  --moderator-model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf \
  --remote-bots
```

**On each bot machine** — connect to the coordinator's IP:

```bash
# Singer
python bot.py --name singer \
  --prompt prompts/singer.txt \
  --model models/meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf \
  --coordinator ws://<coordinator-ip>:8765

# Guitarist
python bot.py --name guitarist \
  --prompt prompts/guitarist.txt \
  --model models/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf \
  --coordinator ws://<coordinator-ip>:8765

# Bassist
python bot.py --name bassist \
  --prompt prompts/bassist.txt \
  --model models/c4ai-command-r7b-12-2024-abliterated-Q4_K_M.gguf \
  --coordinator ws://<coordinator-ip>:8765

# Drummer
python bot.py --name drummer \
  --prompt prompts/drummer.txt \
  --model models/mistralai_Ministral-3-8B-Instruct-2512-abliterated.i1-Q4_K_M.gguf \
  --coordinator ws://<coordinator-ip>:8765
```

Add `--small-model <path>` to each bot for the overnight model swap. The coordinator waits for all 4 to connect before starting.

---

## Optional flags

| Flag | Description |
|---|---|
| `--remote-bots` | Skip launching local subprocesses — wait for bots to connect remotely |
| `--fast` | Compress each day to ~20 real minutes for testing |
| `--no-tts` | Disable audio on all bots (local mode only) |
| `--osc-ip <ip>` | Send OSC messages to this IP (for Max/MSP, TouchDesigner, etc.) |
| `--day sunday\|monday\|...` | Force a specific day instead of auto-advancing |
| `--language <code>` | Response language, e.g. `es`, `fr` (default: `en`) |
| `--conv-log <path>` | Conversation log file (default: `conv_log.jsonl`) |

---

## What happens during a run

**Daytime (8am–midnight / first 12 min in fast mode)**
- Bots argue about the song — title, theme, lyrics, structure, instruments
- All conversation is logged to `conv_log.jsonl`
- On Sunday: band name discussion for the first 30 min, then locked in

**Midnight / session end**
- Injection fires: `"Session's over. Keep talking."`
- Moderator LLM reads the last 200 lines of conversation and extracts song fields into `song_state.json`
- Bots swap to the small model and go to sleep one by one (2:30–3:30am / ~2 min in fast mode)
- ACE-Step generates the track in the background → saved to `output/track_<day>_<timestamp>.wav`
- EP state updated in `ep_state.json`

**Thursday evening (7pm / 18 min in fast mode)**
- Listening party: each track plays, bots react in performance mode
- Final injection: `"That's the set. We made something."`

---

## Output files

| File | Contents |
|---|---|
| `conv_log.jsonl` | Full conversation log (one JSON object per line) |
| `song_state.json` | Current day's song fields (title, lyrics, tempo, key, mood, etc.) |
| `ep_state.json` | EP record — band name + all completed tracks |
| `output/track_<day>_<timestamp>.wav` | Generated music files |
| `voice_config.json` | Voice assignments and vocal style descriptions |

---

## Injecting into the conversation

Type directly into the terminal where `band_coordinator.py` is running and press Enter. The text is sent to all bots as a human prompt, visible to everyone in the conversation.

---

## Stopping and restarting

`Ctrl+C` terminates the coordinator and all bot subprocesses cleanly.

The installation auto-advances days by reading `ep_state.json` — whichever days have completed tracks are skipped. To restart a specific day without losing other progress, delete the relevant track entry from `ep_state.json` and use `--day <name>`.

To start completely fresh:

**Windows (PowerShell):**
```powershell
Remove-Item -ErrorAction SilentlyContinue conv_log.jsonl, song_state.json, ep_state.json, output\*.wav
```

**Mac/Linux:**
```bash
rm -f conv_log.jsonl song_state.json ep_state.json output/*.wav
```
