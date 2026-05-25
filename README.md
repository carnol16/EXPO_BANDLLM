# EXPO_BANDLLM

Four AI bots locked in a room for a simulated week, trying to record a punk-industrial EP. They hate each other. They hate the idea of a polished record. They do it anyway.

Each bot runs its own language model and text-to-speech voice. A moderator LLM picks who speaks next — no round-robin, no turns. Conversation emerges. Over seven simulated days the band names itself, argues about songs, writes lyrics, and hands control to a music generation model (ACE-Step 1.5) to render the tracks. The output is a full EP: audio files, lyrics, metadata.

---

## Media

> **Coming Soon** — demo audio and installation footage

---

## The Band

| Bot | Name | Personality |
|-----|------|-------------|
| Singer | Rex | Cynical, confrontational. Treats pitch correction as a moral failure. |
| Guitarist | Charles | Abrasive, committed to noise and feedback over melody. |
| Bassist | George | Contrarian, holds the low end and a grudge. |
| Drummer | Johnathan | Mechanically precise, philosophically nihilistic. |

Personalities are defined in [`prompts/`](prompts/) — plain `.txt` system prompts, easy to swap.

---

## Built On: LLM_Conversation

EXPO_BANDLLM is built on top of the [LLM_Conversation](https://github.com/carnol16/LLM_Conversation) engine — a local multi-agent conversation system where independent LLM instances hold autonomous spoken conversations with each other. The core mechanic: a separate moderator LLM reads the last few lines of conversation and picks the next speaker by name, producing natural dynamics (follow-ups, interruptions, silences) instead of round-robin turns.

The base engine also handles:
- WebSocket-based coordinator/bot protocol with UUID-validated turns
- Response sanitization (strips template artifacts, truncates multi-speaker bleed, catches degenerate output)
- Hardware-aware GPU layer allocation across bots
- Pipelined turn execution — the next `speak` is dispatched before the current bot's audio finishes, eliminating dead air

> For full detail on the conversation engine, see the [LLM_Conversation README](https://github.com/carnol16/LLM_Conversation).

EXPO_BANDLLM takes that engine and adds everything needed to turn an open-ended conversation into a structured creative production.

---

## Extensions Over the Base Engine

### 1. Day Arc — Simulated Week Schedule

`day_arc.py` drives the band through a seven-day narrative. Instead of freeform conversation, context injections arrive at scheduled intervals to push the band toward specific creative goals each day:

| Day | Theme |
|-----|-------|
| Monday | Band naming — bots argue about what to call themselves |
| Tuesday–Wednesday | Song writing — lyrics, structure, vibe |
| Thursday | Pre-production — arrangement decisions |
| Friday | Recording — ACE-Step generates 3 audio variants per song |
| Saturday | Mixing notes, band drama |
| Sunday | Listening party — EP is finalized |

The arc also runs a background LLM extraction pass after each session — using the coordinator's already-loaded moderator LLM — to pull structured fields (song title, tempo, key, lyrics, structure) out of the conversation log and commit them to the song document.

### 2. Song & EP Documents

`document_creator/song_document.py` and `ep_document.py` track creative state as it evolves. Fields like `title`, `tempo_bpm`, `key`, `lyrics`, and `structure` are set incrementally as the bots agree on them. Each field commit:

- Timestamps the agreement (`agreed_at`)
- Triggers registered callbacks (e.g. UI sync, OSC notification)
- Atomically saves to disk with a write-tmp-then-replace pattern (safe against crashes mid-write)

At midnight in the simulated week, the moderator LLM distills the day's conversation into a compact ACE-Step caption — a genre-anchored description of the track that feeds directly into music generation.

### 3. ACE-Step 1.5 Music Generation

`ace_step_bridge.py` integrates [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) — a diffusion-transformer music generation model — as the band's recording studio.

When the band goes idle on Friday, the coordinator hands full GPU to the DiT (Diffusion Transformer) model:

1. **VRAM handoff** — the coordinator's moderator LLM remains loaded, but `_release_dit()` ensures no stale DiT weights compete for memory
2. **Caption + lyrics** — the song document's distilled caption and formatted lyric sheet (with section tags like `[Chorus - explosive]`) are passed to the model
3. **3 audio variants** — three tracks are generated with different random seeds; the band picks the best at the listening party
4. **Output** — `.wav` files saved to `output/track_<day>_<timestamp>_v<N>/`

The genre prefix is hardcoded to anchor the model toward the intended sound regardless of what the distillation LLM wrote:

```
90's grunge, 2010's deathcore. in the vein of Nirvana, Alice in Chains,
early TOOL (Undertow era), Chelsea Grin, Born of Osiris...
```

### 4. GPU Arbitration — ResourceManager

`resource_manager.py` coordinates GPU access between the four bot subprocesses and the ACE-Step generation pipeline. Bots check in and out of GPU use; the resource manager blocks ACE-Step from starting until all bots are idle, then signals them not to request GPU until generation completes.

### 5. Single-Entry Launch

The base engine requires one terminal per bot plus one for the coordinator. `band_coordinator.py` collapses this into a single command — it spawns all four bot subprocesses, monitors them, and auto-restarts any that crash (up to 10 times each with configurable cooldown).

---

## Architecture

```
band_coordinator.py
├── Loads moderator LLM (picks next speaker from recent context)
├── Starts WebSocket coordinator server
├── Spawns 4 bot subprocesses (singer, guitarist, bassist, drummer)
│   └── Each bot: loads its own LLM + Kokoro TTS, connects via WebSocket
├── Runs DayArc (simulated week: naming → writing → recording → listening party)
│   ├── Periodic LLM extraction → SongDocument fields
│   ├── Midnight: caption distillation → ACE-Step prompt
│   └── Friday idle window → ace_step_bridge.generate()
└── ResourceManager arbitrates GPU between bots and ACE-Step
```

---

## Hardware

Designed to run on a single local machine. Hardware is auto-detected at startup via `hardware.py`:

| Profile | Context | GPU Layers | Notes |
|---------|---------|------------|-------|
| CUDA (NVIDIA) | 4096 tokens | Calculated from free VRAM | Recommended for ACE-Step |
| Metal (Apple Silicon) | 4096 tokens | All layers (unified memory) | MPS backend |
| CPU fallback | 2048 tokens | None | Very slow — not recommended |

The first installation ran on a single Windows PC with an NVIDIA GPU. ACE-Step requires CUDA for reasonable generation times (~3–5 min per track at 8 inference steps).

---

## Prerequisites

- Python 3.9+
- [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) built for your backend (CUDA / Metal / CPU)
- [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) cloned into `ACE-Step-1.5/`
- [Kokoro TTS](https://github.com/hexgrad/kokoro) for bot voices
- GGUF model files for bots and moderator (e.g. Dolphin Mistral 7B Q4_K_M)

---

## Installation

```bash
git clone <repo-url>
cd EXPO_BANDLLM
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python setup.py
```

`setup.py` handles everything in one pass:
- Detects your GPU (NVIDIA / Apple Silicon / CPU) and installs `llama-cpp-python` with the correct build flags
- Installs remaining Python dependencies from `requirements.txt`
- Clones ACE-Step 1.5 into `ACE-Step-1.5/`
- Downloads all five GGUF model files into `models/` (~19 GB total)

**Partial install flags:**

| Flag | Effect |
|------|--------|
| `--skip-models` | Dependencies + ACE-Step clone only — use your own model files |
| `--coordinator-only` | Download only the coordinator model (~2.2 GB) |
| `--skip-acestep` | Skip ACE-Step clone (already present) |

---

## Running

```bash
python band_coordinator.py \
  --moderator-model models/moderator.gguf \
  --bot-model models/dolphin-2.2.1-mistral-7b.Q4_K_M.gguf
```

**Common flags:**

| Flag | Description |
|------|-------------|
| `--bot-model` | GGUF path used by all four bots |
| `--singer-model` / `--guitarist-model` / etc. | Per-bot model override |
| `--small-model` | Smaller GGUF for overnight low-power use |
| `--no-tts` | Disable Kokoro TTS (text output only) |
| `--osc-ip` | Send OSC messages to a lighting/visuals controller |
| `--day` | Override active day (`monday`, `tuesday`, etc.) for testing |
| `--fast` | Compress each day to ~20 real minutes |
| `--remote-bots` | Don't spawn local subprocesses — wait for bots to connect remotely |
| `--conv-log` | Path to save conversation log (default: `conv_log.jsonl`) |

Bots auto-restart on crash (up to 10 times). Song documents and conversation state save incrementally — a run can survive interruptions.

---

## Output

After a full week:

```
output/
  track_friday_<timestamp>_v1/   # Audio variant 1
  track_friday_<timestamp>_v2/   # Audio variant 2
  track_friday_<timestamp>_v3/   # Audio variant 3
song_state.json                  # Final song state
conv_log.jsonl                   # Full conversation log
```

The `EPDocument` aggregates all songs into a single metadata record at the end of the week.

---

## Project Structure

```
band_coordinator.py      # Entry point — launches all components
coordinator.py           # WebSocket server + moderator LLM (from LLM_Conversation)
bot.py                   # Bot client: LLM + Kokoro TTS + WebSocket (from LLM_Conversation)
day_arc.py               # Simulated week schedule and LLM extraction
resource_manager.py      # GPU arbitration between bots and ACE-Step
hardware.py              # Hardware profile detection (from LLM_Conversation)
ace_step_bridge.py       # ACE-Step 1.5 music generation integration
vocal_synthesizer.py     # Kokoro TTS for full-song vocal rendering
document_creator/
  song_document.py       # Per-song state: lyrics, tempo, key, structure
  ep_document.py         # EP aggregation across all songs
prompts/                 # Bot system prompts (singer, guitarist, bassist, drummer)
ACE-Step-1.5/            # Music generation model (separate clone)
tests/                   # Unit tests
```

Files marked "from LLM_Conversation" share lineage with the base engine — see that repo for architectural context.

---

## Tests

```bash
python -m pytest tests/
```

Covers hardware detection, bot conversation history, coordinator speaker selection, and WebSocket integration.
