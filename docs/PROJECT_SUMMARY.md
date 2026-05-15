# Four-Bot Multi-LLM Conversation System — Project Summary

## Starting Point

The project began with a working two-bot conversation script (`two_llm_chat.py`) that ran two LLMs in a strict round-robin — Alice speaks, then Bob, then Alice, forever. Everything ran on one machine, one model hot-swapped at a time.

---

## What We Designed

A full brainstorming and design session produced a 4-bot system with the following key decisions:

### Architecture — Hub and Spoke

- One **coordinator** machine (Mac mini M2 Pro in production, i9/RTX 3080 for testing) acts as the WebSocket server and conversation brain
- Four **bot machines** (Raspberry Pi 5s, 16GB each) each run one bot — their own LLM loaded permanently, their own TTS voice, their own speaker

### Natural Conversation (not round-robin)

- A small moderator LLM on the coordinator reads the last few messages and picks who should speak next by name
- It avoids picking the same bot twice in a row and falls back to random if its output is garbled

### Hardware Auto-Detection

Rather than hardcoding GPU settings, the system detects what hardware it's on and configures itself:

| Platform | Profile | n_gpu_layers | Notes |
|---|---|---|---|
| CUDA (i9/RTX 3080) | SPLIT | 10 | 10 layers to VRAM, rest to DDR5 |
| Apple Metal (Mac mini) | METAL | -1 | All layers to unified memory |
| Raspberry Pi | RPI | 0 | CPU only, reduced context (2048) |
| CPU fallback | CPU | 0 | Pure CPU |

Override anytime with `--gpu-layers N`.

### Per-Bot Personality Files

Each bot gets its own `.txt` file in `prompts/` with an optional `voice:` header and a system prompt defining their character:

| Bot | Personality |
|---|---|
| Alice | Skeptic — challenges every claim |
| Bob | Egomaniac — self-absorbed, occasionally rude |
| Charlie | Toxic optimist — cheerful, passive-aggressively undermining |
| Diana | Pretentious intellectual — corrects everyone, offended by simplicity |

### WebSocket Protocol

Clean message types with a two-phase startup:

```
Bot connects → sends "register" → coordinator sends "load" with GPU layers → bot loads LLM+TTS → sends "ready"
Coordinator waits for all "ready" signals → generates opening line → starts conversation

Per turn:
  Coordinator → bot:   {"type": "speak", "turn_id": N}
  Bot → coordinator:   {"type": "reply", "name": "Alice", "turn_id": N, "text": "..."}
  Coordinator → all:   {"type": "broadcast", "speaker": "Alice", "text": "..."}

Human interrupt:
  Coordinator → all:   {"type": "inject", "text": "What about creative jobs?"}
```

A `turn_id` on every speak/reply pair means stale replies (from timeouts) get discarded rather than poisoning the conversation.

---

## What Was Built

### `hardware.py`

Auto-detects the runtime environment and returns the right `n_gpu_layers` and `n_ctx` for llama-cpp. Both `bot.py` and `coordinator.py` call this at startup.

### `bot.py`

The client that runs on each Raspberry Pi (or terminal window for testing):

- Connects to coordinator, registers its name, loads LLM + Piper TTS
- Handles `speak` messages: generates a reply (non-blocking via `asyncio.to_thread`), plays audio on the local speaker in a background thread, sends reply back
- Maintains its own conversation history with correct role labeling (its own words = `assistant`, everyone else's = `user`)
- Handles `broadcast` (someone else spoke — update history) and `inject` (human changed topic)
- `--no-tts` flag skips audio for testing without speakers or voice files

### `coordinator.py`

The server that runs on the Mac mini / test machine:

- Waits for all bots to register and signal ready
- Generates an opening line using the moderator LLM (customizable via `--opener-prompt`)
- Moderator LLM runs CPU-only on CUDA (preserves all VRAM for bots), but uses Metal acceleration on Apple Silicon (unified memory, no VRAM to protect)
- Runs the moderation loop: call moderator → pick speaker → send `speak` → wait for `reply` (with timeout) → validate turn_id → broadcast to all bots → repeat
- Human input thread lets you type mid-conversation to change the subject
- Rejects duplicate bot names and late-joining connections

### `prompts/charlie.txt` and `prompts/diana.txt`

Two new personalities added to complement Alice and Bob.

### Test Suite

19 unit tests covering:

- Hardware profile detection (mocked for all 4 platforms)
- Bot history management logic (role assignment, inject prefix, accumulation)
- Coordinator moderator parsing (name matching, punctuation stripping, fallback, turn_id validation)
- Integration test skeleton (full WebSocket flow with mock LLM)

---

### Connection Stability

WebSocket connections use extended ping/pong timeouts (120s interval, 120s timeout) on both the coordinator and bots. This prevents early-connecting bots from being dropped while later bots are still loading their models — particularly important on slower hardware like Raspberry Pis.

---

## What Was Not Changed

`engine.py` and `two_llm_chat.py` were never touched. The original single-machine round-robin conversation still works exactly as before.

---

## Production Deployment (Target)

```
Mac mini M2 Pro (16GB)          Raspberry Pi 5 x4 (16GB each)
┌─────────────────────┐         ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│   coordinator.py    │◄───────►│ bot  │ │ bot  │ │ bot  │ │ bot  │
│   moderator LLM     │WebSocket│Alice │ │ Bob  │ │Charlie│ │Diana │
│   conversation log  │         │ LLM  │ │ LLM  │ │ LLM  │ │ LLM  │
│   human input       │         │ TTS  │ │ TTS  │ │ TTS  │ │ TTS  │
└─────────────────────┘         │spkr  │ │spkr  │ │spkr  │ │spkr  │
                                └──────┘ └──────┘ └──────┘ └──────┘
```

## Spec and Plan Documents

- Spec: `docs/superpowers/specs/2026-03-21-four-bot-chat-design.md`
- Plan: `docs/superpowers/plans/2026-03-21-four-bot-chat.md`
