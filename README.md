# EXPO_BANDLLM

Four AI bots locked in a room for a simulated week, trying to record a punk-industrial EP. They hate each other. They hate the idea of a polished record. They do it anyway.

Each bot runs its own language model and text-to-speech voice. A moderator LLM picks who speaks next — no round-robin, no turns. Conversation emerges. Over seven simulated days the band names itself, argues about songs, writes lyrics, and hands control to a music generation model (ACE-Step 1.5) to render the tracks. The output is a full EP: audio files, lyrics, metadata.

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

## How It Works

```
band_coordinator.py
├── Loads moderator LLM (picks next speaker from recent context)
├── Starts WebSocket coordinator server
├── Spawns 4 bot subprocesses (singer, guitarist, bassist, drummer)
│   └── Each bot: loads its own LLM + Kokoro TTS, connects via WebSocket
├── Runs DayArc (simulated week: naming day → writing days → listening party)
└── Triggers ACE-Step 1.5 for music generation when bots are idle
```

**Moderator LLM** — A small GGUF model on the coordinator reads the last N messages and picks the next speaker by name. This prevents round-robin monotony and produces natural-feeling conversation with follow-ups, interruptions, and silences.

**Turn ID validation** — Every conversation turn gets a UUID. Stale replies from slow models are discarded without poisoning the conversation log.

**GPU handoff** — The `ResourceManager` tracks which processes hold GPU. When all bots go idle between recording sessions, full GPU is handed to ACE-Step for audio generation.

**Song documents** — Agreement on song details (title, key, tempo, lyrics, structure) is tracked in a `SongDocument`. Fields trigger callbacks when finalized, enabling real-time sync. All songs roll up into an `EPDocument` at the end of the week.

---

## Day Arc

| Day | Theme |
|-----|-------|
| Monday | Band naming — bots argue about what to call themselves |
| Tuesday–Wednesday | Song writing — lyrics, structure, vibe |
| Thursday | Pre-production — arrangement decisions |
| Friday | Recording — ACE-Step generates 3 audio variants per song |
| Saturday | Mixing notes, band drama |
| Sunday | Listening party — EP is finalized |

The arc is driven by `day_arc.py`, which injects context messages into the conversation at scheduled intervals.

---

## Hardware

This project runs on a single local machine. Hardware is auto-detected at startup via `hardware.py`:

| Profile | Context | GPU Layers |
|---------|---------|------------|
| CUDA (NVIDIA) | 4096 tokens | 10 layers offloaded |
| Metal (Apple Silicon) | 4096 tokens | All layers (unified memory) |
| CPU fallback | 2048 tokens | None |

A CUDA GPU or Apple Silicon Mac is strongly recommended — running four LLMs plus a moderator on CPU is very slow.

---

## Prerequisites

- Python 3.9+
- [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) built for your backend (CUDA / Metal / CPU)
- [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) installed in `ACE-Step-1.5/`
- [Kokoro TTS](https://github.com/hexgrad/kokoro) for bot voices
- GGUF model files for the bots and moderator (e.g. Dolphin Mistral 7B)

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

pip install -r requirements.txt
```

Place your GGUF model files in a `models/` directory (excluded from git):

```
models/
  dolphin-2.2.1-mistral-7b.Q4_K_M.gguf   # or any compatible GGUF
  moderator.gguf
```

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
| `--bot-model` | GGUF path used by all four bots (default) |
| `--singer-model` / `--guitarist-model` / etc. | Override model per bot |
| `--no-tts` | Disable Kokoro TTS (text output only) |
| `--osc-ip` | Send OSC messages to a lighting/visuals controller |
| `--conv-log` | Path to save conversation log |
| `--language` | TTS language code (default: `en`) |

Bots auto-restart on crash up to 10 times. Conversation state and song documents are saved incrementally so a run can survive interruptions.

---

## Output

After a full run:

```
output/
  ep_document.json       # Full EP metadata
  song_state.json        # Final song states
  *.wav / *.mp3          # Generated audio (3 variants per song)
runs/
  Run<N>/
    conversation_log.txt
    song_states/
```

---

## Project Structure

```
band_coordinator.py      # Entry point
coordinator.py           # WebSocket server + moderator logic
bot.py                   # Bot client (LLM + TTS)
day_arc.py               # Simulated week schedule
resource_manager.py      # GPU arbitration
hardware.py              # Hardware profile detection
ace_step_bridge.py       # ACE-Step music generation integration
vocal_synthesizer.py     # Kokoro TTS for full-song vocal rendering
document_creator/
  song_document.py       # Per-song state tracking
  ep_document.py         # EP aggregation
prompts/                 # Bot system prompts (singer, guitarist, bassist, drummer)
ACE-Step-1.5/            # Music generation model (submodule / separate install)
tests/                   # Unit tests
```

---

## Tests

```bash
python -m pytest tests/
```

Covers hardware detection, bot conversation history, coordinator speaker selection, and WebSocket integration.

---

## License

MIT
