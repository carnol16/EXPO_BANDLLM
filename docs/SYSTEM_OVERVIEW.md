# EXPO_BANDLLM — System Overview

## Concept

A generative art installation. Four LLM-powered bots roleplay as members of a punk-industrial band called **Sonic Anarchy** and spend one simulated week arguing, writing, and recording an EP. Each night, the conversation is harvested for lyrics and song parameters, and ACE-Step generates a real music track. By Thursday evening, five tracks exist and the bots hold a listening party.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      band_coordinator.py                         │
│  Entry point — loads moderator LLM, spawns coordinator +        │
│  DayArc, launches 4 bot subprocesses                            │
└────────┬──────────────────────────┬─────────────────────────────┘
         │                          │
┌────────▼────────┐       ┌─────────▼──────────┐
│  coordinator.py  │       │     day_arc.py      │
│  WebSocket server│       │  Day/session clock  │
│  Moderator LLM   │       │  Midnight injection │
│  Turn sequencing │       │  ACE-Step trigger   │
│  conv_log.jsonl  │       │  Song extraction    │
└────────┬─────────┘       └─────────────────────┘
         │ WebSocket (port 8765)
   ┌─────┼──────────────────────────┐
   │     │                          │
┌──▼──┐ ┌▼─────┐ ┌──────┐ ┌───────┐
│ Rex │ │ Volt │ │Gloom │ │ Crash │
│Singer│ │Gtrist│ │Bassist│ │Drummer│
│ LLM │ │ LLM  │ │ LLM  │ │  LLM  │
│ TTS │ │ TTS  │ │ TTS  │ │  TTS  │
└─────┘ └──────┘ └──────┘ └───────┘
         (bot.py × 4)

                        ┌─────────────────┐
                        │ ace_step_bridge  │
                        │  ACE-Step 1.5   │
                        │  DiT + LLM CoT  │
                        │  WAV output     │
                        └─────────────────┘
```

---

## Components

### `band_coordinator.py` — Entry Point

Single command launches the whole system. Responsibilities:

- Loads the moderator GGUF model (CPU-only on CUDA, Metal-accelerated on Apple Silicon)
- Creates `SongDocument` and `EPDocument` state objects
- Instantiates `Coordinator`, `DayArc`, and `ResourceManager`
- Spawns the 4 bot subprocesses via `subprocess.Popen` (or skips them in `--remote-bots` mode)
- Runs a monitor coroutine that logs if a bot process exits unexpectedly
- All four bots: singer, guitarist, bassist, drummer — each gets its own model path, prompt file, and optional small-model for overnight swaps

---

### `coordinator.py` — WebSocket Server + Moderator

The conversation engine. Runs as a WebSocket server on port 8765.

**Startup handshake:**
1. Bots connect and send `register` with their name and model file size
2. Coordinator calculates GPU layer allocation across all models from available VRAM
3. Sends `load` with `n_gpu_layers` to each bot
4. Bots load their models and send `ready`
5. Once all ready, coordinator fires an opening injection and starts the conversation loop

**Per-turn loop:**
1. Moderator LLM reads the last 5 lines of conversation history, outputs a single name
2. `pick_next_speaker()` validates the name — falls back to random if garbled or same as last speaker
3. Sends `speak` to the chosen bot
4. Bot replies with `reply` (text + turn_id) — stale turn_ids (from timeouts) are discarded
5. Text is broadcast to all bots as `broadcast`
6. Bot plays TTS audio and sends `tts_done`
7. Next bot's `speak` is **pre-dispatched** while current audio plays — eliminates dead air between speakers

**Human injection:** typing in the coordinator terminal sends an `inject` message to all bots mid-conversation.

---

### `bot.py` — Bot Client (× 4)

One subprocess per band member. Each maintains its own conversation history and LLM context.

**Key behaviors:**
- Loads a GGUF model via `llama-cpp-python` with GPU layers allocated by the coordinator
- On `speak`: generates a reply via `LLMEngine`, validates it, sanitizes it, synthesizes TTS
- On `broadcast`: appends the speaker's line to its history (role = `user` for others, `assistant` for itself)
- On `inject`: prepends `[From the crowd]: ` and appends to history
- **History window:** last 10 messages — older context is dropped to keep inference fast
- **Overnight model swap:** swaps to the `--small-model` at sleep time (2:30–3:30am / ~1 min fast mode), swaps back at wake
- **TTS:** Kokoro 82M neural TTS, voice assigned per-bot from `voice_config.json`

**Sampling parameters:** `temperature=0.75`, `repeat_penalty=1.15`, `top_k=40`, `top_p=0.95`

---

### `engine.py` — LLM Message Building + Quality Guard

Shared by all bots. Two main jobs:

**Response sanitization** (`_sanitize_response`):
- Strips instruction template artifacts: `[INST]`, `[CHAR]`, `<|assistant|>`, etc.
- Removes leading name echo (`"Rex: ..."` → `"..."`)
- Truncates at the first sign of another character's turn (`\nName:`)
- Strips AI disclaimers, OOC notes, `---` separators

**Validation + retry** (`_is_valid_response`):
- Fewer than 3 words → rejected
- Non-ASCII ratio > 20% → rejected
- Single character dominates > 35% of non-space content → rejected

If a reply fails validation, bot retries once with hardened params (`temperature=0.5`, `repeat_penalty=1.3`). If that fails too, the turn is skipped entirely. Invalid broadcasts are never appended to history — prevents garbage from poisoning future context.

---

### `day_arc.py` — Day/Session Lifecycle

Runs concurrently with the conversation. Manages the simulated week.

**Session structure (real-time / fast-mode):**

| Phase | Real time | Fast mode |
|---|---|---|
| Daytime session | 8am–midnight (~16 hrs) | ~15 min |
| Band name window (Sunday only) | First 30 min | ~5 min |
| Midnight injection + extraction | ~30 min | ~15 min |
| Bot sleep (one by one) | 2:30–3:30am | ~1 min each |
| Thursday listening party | 7pm–midnight | ~35 min |

**What fires at midnight:**
1. Injection: `"Session's over. Keep talking."`
2. Moderator LLM reads last 200 lines of `conv_log.jsonl`, extracts song fields into `song_state.json`: title, lyrics, tempo, key, mood, instruments, vocal style
3. Bots are sent to sleep one by one (model swap to small model)
4. `ResourceManager` detects all bots sleeping → hands full GPU to ACE-Step
5. ACE-Step generates the track → saved to `output/track_<day>_<timestamp>.wav`
6. EP state updated in `ep_state.json`

---

### `ace_step_bridge.py` — Music Generation

Wraps ACE-Step 1.5 for track generation.

**Pipeline:**
- **DiT handler** (`AceStepHandler`) — diffusion-based audio generation, SFT config for best prompt adherence
- **LLM handler** (`LLMHandler`) — Chain-of-Thought reasoning about the prompt before generation; offloaded to CPU after CoT so DiT has full VRAM
- Minimum track duration: **3.5 minutes** (`_MIN_DURATION = 210`)
- Output: WAV written to `output/`

**Supports two integration modes:**
- REST API (separate process, `start_api_server.bat`) — recommended
- Direct Python import (`uv sync`) — in-process, no server needed

---

### `vocal_synthesizer.py` — Overnight Vocal Track

Separate from bot TTS. Synthesizes lyric lines for the ACE-Step vocal reference track. Uses Kokoro 82M at 24kHz mono, one pipeline per bot voice. Has a 70% chance of adding a backing vocal layer per line. Retries each line up to 3 times on synthesis failure.

---

### `resource_manager.py` — GPU Handoff

Tracks which bots are sleeping. When all 4 bots have sent `sleeping` state, fires the `on_full_handoff` callback — which signals `day_arc.py` to boost ACE-Step's OS process priority (`ABOVE_NORMAL_PRIORITY_CLASS` on Windows, `nice(-5)` on Linux) so it can fully use the GPU uncontested.

---

### `hardware.py` — Platform Detection

Auto-detects runtime environment:

| Platform | Profile | GPU Layers | n_ctx |
|---|---|---|---|
| CUDA (RTX 3080) | `CUDA` | calculated from free VRAM | 4096 |
| Apple Metal (M2 Pro) | `METAL` | `-1` (all layers) | 4096 |
| Raspberry Pi 5 | `RPI` | `0` (CPU only) | 2048 |
| CPU fallback | `CPU` | `0` | 2048 |

GPU layer calculation:
```
gpu_layers = floor((free_vram - kv_reserve) × 32 / total_model_size_mb)
```

Also patches `jinja2` at startup for `{% break %}` support in model chat templates (required by c4ai-command-r).

---

## The Band

| Bot | Role | Model | Character |
|---|---|---|---|
| Rex | Singer | `meta-llama-3.1-8b-instruct-abliterated.Q4_K_M` | Confrontational frontman, lyrics-first, anti-commercial |
| Volt | Guitarist | `Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat` | Gear-obsessed, fake vintage equipment, disrupts consensus |
| Gloom | Bassist | `c4ai-command-r7b-12-2024-abliterated-Q4_K_M` | Low-end pessimist, structural thinker, dark lyrical lens |
| Crash | Drummer | `mistralai_Ministral-3-8B-Instruct-2512-abliterated.i1-Q4_K_M` | Rhythm anchor, texture/dynamics obsessive, physically expressive |
| Moderator | Coordinator | `Dolphin3.0-Llama3.2-3B-Q5_K_M` | CPU-only (CUDA) — picks speakers, extracts song state |

**Prompt rules enforced per bot:**
- **Anti-Sellout Rule** — novelty-seeking is character identity; repeating an idea you already said is breaking character
- **Glitch & Destroy Rule** — specific fake gear inventories, glitch techniques, unconventional sound sources
- **Contribution Mandate** — every turn must add a role-specific thing (lyric fragment, chord, rhythmic pattern, structural change)

---

## WebSocket Message Protocol

| Message | Direction | Key Fields | Purpose |
|---|---|---|---|
| `register` | bot → coord | `name`, `model_size_mb` | Bot announces itself on connect |
| `load` | coord → bot | `n_gpu_layers` | Tells bot how much GPU to use |
| `ready` | bot → coord | `name` | Bot finished loading, ready to speak |
| `load_failed` | bot → coord | `name`, `error` | Bot failed to load |
| `speak` | coord → bot | `turn_id` | Bot's turn to generate |
| `reply` | bot → coord | `name`, `turn_id`, `text` | Bot's generated response |
| `broadcast` | coord → all | `speaker`, `text` | Everyone hears what was said |
| `play_tts` | coord → bot | `turn_id` | Signal to play audio now |
| `tts_done` | bot → coord | `name`, `turn_id` | Bot finished playing |
| `inject` | coord → all | `text` | Human or DayArc injects into conversation |
| `sleeping` | bot → coord | `name` | Bot swapped to small model, going idle |
| `error` | coord → bot | `text` | Error notification |

`turn_id` is on every speak/reply pair — stale replies from timed-out bots are discarded rather than poisoning the conversation.

---

## State Files

| File | Contents |
|---|---|
| `conv_log.jsonl` | Full conversation log — one JSON object per line |
| `song_state.json` | Current day's song: title, lyrics, tempo, key, mood, instruments, vocal style |
| `ep_state.json` | EP record: band name + all completed tracks with file paths |
| `voice_config.json` | Voice ID and vocal style description per bot (written by `band_setup.py`) |
| `output/track_<day>_<timestamp>.wav` | Generated music files |

The installation auto-resumes: on startup it reads `ep_state.json` and skips days that already have a completed track. To restart a specific day, delete its entry from `ep_state.json` and use `--day <name>`.

---

## Deployment Modes

**Single machine (i9/RTX 3080):** all 4 bots run as subprocesses alongside the coordinator. GPU is time-shared; bots run sequentially (one speaks at a time, so VRAM contention is minimal).

**Multi-machine (Mac mini + 4× Raspberry Pi 5):** coordinator runs on the Mac mini with `--remote-bots`; each Pi runs one `bot.py` instance. Bots connect to the coordinator's IP. Each Pi has its own speaker — audio is physically distributed across the room.
