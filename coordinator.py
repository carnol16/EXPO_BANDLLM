"""Coordinator — WebSocket server, moderator LLM, conversation orchestration."""

import argparse
import asyncio
import datetime
import json
import os
import random
import threading

import websockets

from hardware import detect_hardware_profile, register_cuda_dll_dirs, patch_jinja_loopcontrols, calculate_gpu_layers, plan_gpu_allocation

register_cuda_dll_dirs()      # Must run before importing llama_cpp (Python 3.8+ Windows DLL search)
patch_jinja_loopcontrols()    # Enable {% break %} in model chat templates (e.g. c4ai-command-r)
from llama_cpp import Llama
from engine import _sanitize_response

MODERATOR_N_CTX = 2048
MODERATOR_MAX_TOKENS = 4
MODERATOR_TEMPERATURE = 0.75


def parse_moderator_output(raw: str, registered: list) -> str | None:
    cleaned = raw.strip().strip(".,!?\"'").strip()
    for name in registered:
        if name.lower() == cleaned.lower():
            return name
    return None


def pick_next_speaker(raw: str, registered: list, last_speaker: str | None) -> str:
    name = parse_moderator_output(raw, registered)
    if name is None or name == last_speaker:
        candidates = [n for n in registered if n != last_speaker]
        if not candidates:
            candidates = registered
        chosen = random.choice(candidates)
        if name is None:
            print(f"  [Coordinator] Moderator gave no valid name ('{raw.strip()}') — random: {chosen}")
        else:
            print(f"  [Coordinator] Avoiding repeat speaker — picking: {chosen}")
        return chosen
    return name


def validate_tts_done(tts_msg: dict, expected_turn: int, speaker_name: str) -> str:
    """Validate a tts_done message. Returns 'ok', 'wrong_type', 'stale', or 'invalid'."""
    if not isinstance(tts_msg, dict):
        return "invalid"
    if tts_msg.get("type") != "tts_done":
        return "wrong_type"
    if tts_msg.get("turn_id") != expected_turn:
        return "stale"
    return "ok"


class Coordinator:
    DEFAULT_OPENER = (
        "The band is about to start their recording session and need to begin discussing the parameters of the song." \
        "The details needed for the song include bpm, lyrics, structure, title, key, and mood." \
        "Pick one of the needed details to kick off the coversation "
        "This will be spoken by {name}."
    )

    def __init__(self, moderator_llm, expected_bots, reply_timeout, register_timeout,
                 opener_prompt=None, osc_sender=None, conv_log_path="conv_log.jsonl"):
        self.llm = moderator_llm
        self.expected_bots = expected_bots
        self.reply_timeout = reply_timeout
        self.register_timeout = register_timeout
        self.opener_prompt = opener_prompt or self.DEFAULT_OPENER
        self.osc_sender = osc_sender
        self.conv_log_path = conv_log_path

        self.connections = {}      # name -> websocket
        self.pending_bots = {}     # name -> {ws, model_size_mb} (pre-load phase)
        self.ready_bots = set()    # names of bots that completed loading
        self.ready_event = asyncio.Event()
        self.conv_log = []
        self.turn_id = 0
        self.last_speaker = None
        self.timed_out = set()     # bots that timed out — skip until they respond
        self.sleeping_bots = set() # bots put to sleep — skipped by moderator
        self._load_lock = None     # initialized in run() (needs event loop)
        self._all_registered = None  # asyncio.Event — initialized in run()
        self.pause_gate = None     # asyncio.Event — initialized in run()
        self._allocation_started = False  # guard against double-trigger
        self.loop = None
        self._bot_queues: dict[str, asyncio.Queue] = {}
        self._turn_count: int = 0
        self._session_summary: str = ""
        self._tasks: set = set()
        self._llm_lock: asyncio.Lock = asyncio.Lock()
        self._bot_gpu_layers: dict[str, int] = {}  # cached per-bot GPU allocation for rejoin

    def _log(self, speaker, text):
        self.conv_log.append({"role": "user", "content": f"{speaker}: {text}"})
        try:
            with open(self.conv_log_path, "a") as f:
                f.write(json.dumps({
                    "speaker": speaker,
                    "text": text,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }) + "\n")
        except Exception as e:
            print(f"[Coordinator] WARNING: conv_log write failed: {e}")

    def _log_window(self):
        return self.conv_log[-3:]

    def _moderator_prompt(self):
        names = ", ".join(self.connections.keys())
        return (
            f"You are a conversation moderator. Based on the conversation so far, "
            f"pick who should speak next. Reply with exactly one name from: {names}. "
            f"No explanation — just the name."
        )

    def _call_moderator(self) -> str:
        names = list(self.connections.keys())
        names_str = ", ".join(names)
        # Build messages directly — do NOT use _build_messages which adds
        # [Character Context] tags that confuse the model into generating dialogue.
        log_text = "\n".join(
            entry["content"] for entry in self._log_window()
        )
        msgs = [{"role": "user", "content": (
            f"Conversation so far:\n{log_text}\n\n"
            f"Pick who should speak next from: {names_str}\n"
            f"Reply with ONLY the name, nothing else."
        )}]
        result = self.llm.create_chat_completion(
            messages=msgs,
            max_tokens=MODERATOR_MAX_TOKENS,
            temperature=MODERATOR_TEMPERATURE,
        )
        return result["choices"][0]["message"]["content"].strip()

    async def _broadcast(self, speaker, text):
        """Send broadcast to ALL connected bots including the speaker."""
        try:
            if self.osc_sender is not None:
                self.osc_sender.send_message("/band/speaker", speaker)
        except Exception as e:
            print(f"[Coordinator] WARNING: OSC send failed: {e}")
        payload = json.dumps({"type": "broadcast", "speaker": speaker, "text": text})
        for name, ws in list(self.connections.items()):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                print(f"  [Coordinator] WARNING: lost connection to {name}")

    async def _plan_and_load_all(self):
        """Plan GPU allocation and load all bots sequentially."""
        names = list(self.pending_bots.keys())
        model_sizes = [self.pending_bots[n]["model_size_mb"] for n in names]
        allocations = plan_gpu_allocation(model_sizes)

        print(f"\n  [Coordinator] Planning GPU allocation for {len(names)} bots...")
        for name, n_gpu_layers in zip(names, allocations):
            print(f"    {name}: {n_gpu_layers} GPU layers")

        for name, n_gpu_layers in zip(names, allocations):
            ws = self.pending_bots[name]["ws"]
            async with self._load_lock:
                try:
                    await ws.send(json.dumps({"type": "load", "n_gpu_layers": n_gpu_layers}))
                    print(f"  [Coordinator] Sent load signal to {name} (n_gpu_layers={n_gpu_layers})")

                    raw = await asyncio.wait_for(ws.recv(), timeout=self.register_timeout)
                    msg = json.loads(raw)
                    if msg.get("type") == "ready":
                        self.connections[name] = ws
                        self.ready_bots.add(name)
                        self._bot_queues[name] = asyncio.Queue()
                        self._bot_gpu_layers[name] = n_gpu_layers
                        asyncio.create_task(self._pump_bot(name, ws))
                        print(f"  [Coordinator] Ready: {name} ({len(self.ready_bots)}/{len(names)})")
                    elif msg.get("type") == "load_failed":
                        error = msg.get("error", "unknown")
                        print(f"  [Coordinator] {name} failed to load: {error}")
                        self.expected_bots -= 1
                    else:
                        print(f"  [Coordinator] WARNING: expected ready from {name}, got {msg.get('type')}")
                except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
                    print(f"  [Coordinator] {name} lost during loading: {e}")
                    self.expected_bots -= 1

        if self.ready_bots:
            self.ready_event.set()
        else:
            print("  [Coordinator] ERROR: No bots loaded successfully")

    async def _pump_bot(self, name, ws):
        """Single recv loop per bot WebSocket — routes raw messages into the bot's queue.

        Prevents ConcurrencyError: only this coroutine calls ws.recv() after load
        completes. Both moderation_loop and broadcast_model_swap read from the queue.
        """
        q = self._bot_queues[name]
        try:
            async for raw in ws:
                await q.put(raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            await q.put(None)  # sentinel — unblocks moderation_loop immediately on disconnect
            # Eagerly clear stale entries so the bot can rejoin without being blocked by
            # handle_registration's wait_closed() still running on the old websocket.
            if self.connections.get(name) is ws:
                self.connections.pop(name, None)
                self.ready_bots.discard(name)
                print(f"  [Coordinator] {name} disconnected")
            if self._bot_queues.get(name) is q:
                self._bot_queues.pop(name, None)

    async def handle_registration(self, websocket):
        registered_name = None
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.register_timeout)
            msg = json.loads(raw)

            if msg.get("type") != "register":
                await websocket.send(json.dumps({"type": "error", "text": "Expected register first"}))
                return

            name = msg.get("name", "").strip()

            if self.ready_event.is_set():
                if name in self._bot_gpu_layers and name not in self.connections:
                    # Rejoin: bot crashed and restarted — restore it with its cached GPU allocation
                    n_gpu_layers = self._bot_gpu_layers[name]
                    print(f"  [Coordinator] {name} rejoining (n_gpu_layers={n_gpu_layers})")
                    try:
                        await websocket.send(json.dumps({"type": "load", "n_gpu_layers": n_gpu_layers}))
                        raw_ready = await asyncio.wait_for(
                            websocket.recv(), timeout=self.register_timeout
                        )
                        join_msg = json.loads(raw_ready)
                        if join_msg.get("type") == "ready":
                            self.connections[name] = websocket
                            self.ready_bots.add(name)
                            self.timed_out.discard(name)
                            self.sleeping_bots.discard(name)  # always wake on rejoin
                            self._bot_queues[name] = asyncio.Queue()
                            asyncio.create_task(self._pump_bot(name, websocket))
                            registered_name = name
                            print(f"  [Coordinator] {name} rejoin complete")
                            if self._session_summary:
                                try:
                                    await websocket.send(json.dumps({
                                        "type": "summary", "text": self._session_summary
                                    }))
                                except Exception:
                                    pass
                        else:
                            print(f"  [Coordinator] {name} rejoin: expected ready, got {join_msg.get('type')}")
                            return
                    except (asyncio.TimeoutError, json.JSONDecodeError, websockets.ConnectionClosed) as e:
                        print(f"  [Coordinator] {name} rejoin failed: {e}")
                        return
                else:
                    await websocket.send(json.dumps({"type": "error", "text": "Conversation in progress"}))
                    return
            else:
                if name in self.pending_bots or name in self.connections:
                    await websocket.send(json.dumps({"type": "error", "text": f"Name already registered: {name}"}))
                    return

                model_size_mb = msg.get("model_size_mb", 0)
                self.pending_bots[name] = {"ws": websocket, "model_size_mb": model_size_mb}
                registered_name = name
                print(f"  [Coordinator] Registered: {name} ({len(self.pending_bots)}/{self.expected_bots})"
                      f" — model: {model_size_mb}MB")

                # Check if all bots have registered
                if len(self.pending_bots) >= self.expected_bots and not self._allocation_started:
                    self._allocation_started = True
                    self._all_registered.set()
                    await self._plan_and_load_all()
                else:
                    # Wait for all bots to register
                    await self._all_registered.wait()

            # Hold connection open — moderation_loop owns ws.recv() from here
            await websocket.wait_closed()

        except asyncio.TimeoutError:
            print(f"  [Coordinator] Timeout during registration")
        except websockets.ConnectionClosed:
            pass
        except json.JSONDecodeError:
            print(f"  [Coordinator] Invalid JSON during registration")
        finally:
            if registered_name:
                self.pending_bots.pop(registered_name, None)
                # Only clean up if this websocket is still the active one —
                # a rejoin may have already replaced it via _pump_bot cleanup.
                if self.connections.get(registered_name) is websocket:
                    del self.connections[registered_name]
                    self.ready_bots.discard(registered_name)
                    self._bot_queues.pop(registered_name, None)
                    print(f"  [Coordinator] {registered_name} disconnected (handler exit)")

    async def _generate_session_summary(self):
        recent = self.conv_log[-10:]
        recent_text = "\n".join(entry["content"] for entry in recent)
        prompt = (
            f"Previous session summary:\n{self._session_summary}\n\n"
            f"Recent conversation (last 10 turns):\n{recent_text}\n\n"
            "Update the summary. Be cumulative — include everything from the previous "
            "summary plus what's new. Capture:\n"
            "- All agreed song details (title, key, tempo, mood, structure, instruments)\n"
            "- Every lyric line proposed and by whom\n"
            "- Key arguments and what was resolved vs. still open\n"
            "- Any strong character moments worth remembering\n"
            "Keep it under 300 words."
        )
        try:
            async with self._llm_lock:
                result = await asyncio.to_thread(
                    self.llm.create_chat_completion,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.5,
                )
            self._session_summary = result["choices"][0]["message"]["content"].strip()
            payload = json.dumps({"type": "summary", "text": self._session_summary})
            for name, ws in list(self.connections.items()):
                try:
                    await ws.send(payload)
                except websockets.ConnectionClosed:
                    print(f"[Coordinator] WARNING: {name} disconnected during summary broadcast")
            print(f"[Coordinator] Session summary updated ({len(self._session_summary)} chars)")
        except Exception as e:
            print(f"[Coordinator] WARNING: summary generation failed: {e}")

    async def _broadcast_song_state(self):
        try:
            with open("song_state.json") as f:
                state = json.load(f)
        except FileNotFoundError:
            print("[Coordinator] song_state.json not found — skipping state broadcast")
            return
        except json.JSONDecodeError:
            print("[Coordinator] WARNING: song_state.json is malformed — skipping state broadcast")
            return

        lines = ["=== WHAT THE BAND HAS AGREED SO FAR ==="]
        if state.get("title"):
            lines.append(f"Title: {state['title']}")
        parts = []
        if state.get("key"):
            parts.append(f"Key: {state['key']}")
        if state.get("tempo_bpm"):
            parts.append(f"Tempo: {state['tempo_bpm']} BPM")
        if state.get("mood"):
            parts.append(f"Mood: {state['mood']}")
        if parts:
            lines.append("  |  ".join(parts))
        if state.get("structure"):
            lines.append(f"Structure: {' → '.join(state['structure'])}")
        if state.get("instruments"):
            lines.append(f"Instruments: {', '.join(state['instruments'])}")
        if state.get("lyrics"):
            lines.append("Lyrics:")
            for lyric in state["lyrics"]:
                text = lyric.get("text", "") if isinstance(lyric, dict) else str(lyric)
                author = lyric.get("author", "Unknown") if isinstance(lyric, dict) else ""
                lines.append(f'  - "{text}" ({author})')
        lines.append("========================================")

        block = "\n".join(lines)
        payload = json.dumps({"type": "state_update", "text": block})
        for name, ws in list(self.connections.items()):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                print(f"[Coordinator] WARNING: {name} disconnected during state broadcast")
        print(f"[Coordinator] Song state broadcast sent")

    async def moderation_loop(self):
        registered = list(self.connections.keys())
        print(f"\n{'='*60}")
        print(f"All {self.expected_bots} bots ready: {', '.join(registered)}")
        print(f"{'='*60}\n")

        # Generate and broadcast opening line
        opener_name = random.choice(registered)
        print("Generating opening line...")
        try:
            opener_raw = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=[{"role": "user", "content": self.opener_prompt.format(name=opener_name)}],
                max_tokens=64,
                temperature=1.2,
                top_p=0.9,
                seed=random.randint(0, 2**31 - 1),
            )
            opening = _sanitize_response(opener_raw["choices"][0]["message"]["content"].strip())
        except Exception as e:
            print(f"[Coordinator] ERROR generating opener: {e}")
            opening = "Let's talk about something interesting."

        print(f"[{opener_name}]: {opening}\n")
        self._log(opener_name, opening)
        await self._broadcast(opener_name, opening)
        self.last_speaker = opener_name

        pending_speak = None  # (speaker_name, ws, turn_id) if speak already sent

        while True:
            await self.pause_gate.wait()
            registered = list(self.connections.keys())
            if not registered:
                print("[Coordinator] No bots connected — waiting for reconnect...")
                await asyncio.sleep(5)
                continue

            if pending_speak is not None:
                # Next bot's speak was already sent during pipeline
                speaker_name, ws, current_turn = pending_speak
                pending_speak = None
                if speaker_name not in self.connections:
                    continue
            else:
                # First turn or pipeline failed — pick speaker normally
                try:
                    async with self._llm_lock:
                        moderator_raw = await asyncio.to_thread(self._call_moderator)
                except Exception as e:
                    print(f"[Coordinator] ERROR calling moderator: {e}")
                    moderator_raw = ""

                available = [n for n in registered if n not in self.timed_out and n not in self.sleeping_bots]
                if not available:
                    if all(n in self.sleeping_bots for n in registered):
                        print("[Coordinator] All active bots are sleeping — waiting")
                        await asyncio.sleep(5)
                        continue
                    print("[Coordinator] All bots previously timed out — resetting")
                    self.timed_out.clear()
                    available = [n for n in registered if n not in self.sleeping_bots]

                speaker_name = pick_next_speaker(moderator_raw, available, self.last_speaker)
                ws = self.connections.get(speaker_name)
                if ws is None:
                    print(f"[Coordinator] {speaker_name} not connected, skipping")
                    continue

                self.turn_id += 1
                current_turn = self.turn_id

                try:
                    await ws.send(json.dumps({"type": "speak", "turn_id": current_turn}))
                except websockets.ConnectionClosed:
                    print(f"[Coordinator] Lost connection to {speaker_name}")
                    self.connections.pop(speaker_name, None)
                    continue

            # Await reply — drain stale messages
            reply_msg = None
            try:
                deadline = asyncio.get_event_loop().time() + self.reply_timeout
                _reply_q = self._bot_queues.get(speaker_name)
                if _reply_q is None:
                    self.connections.pop(speaker_name, None)
                    continue
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    raw_reply = await asyncio.wait_for(_reply_q.get(), timeout=remaining)
                    if raw_reply is None:
                        print(f"[Coordinator] {speaker_name} disconnected — skipping turn")
                        self.connections.pop(speaker_name, None)
                        self._bot_queues.pop(speaker_name, None)
                        break
                    try:
                        msg = json.loads(raw_reply)
                    except json.JSONDecodeError:
                        print(f"[Coordinator] WARNING: invalid JSON from {speaker_name}, skipping")
                        continue
                    if msg.get("turn_id") != current_turn:
                        print(f"  [Coordinator] Draining stale message from {speaker_name} "
                              f"(turn {msg.get('turn_id')} != {current_turn})")
                        continue
                    if msg.get("type") == "reply":
                        reply_msg = msg
                        break
                    continue
            except asyncio.TimeoutError:
                print(f"[Coordinator] WARNING: {speaker_name} timed out on turn {current_turn}, skipping")
                self.timed_out.add(speaker_name)
                continue
            except websockets.ConnectionClosed:
                print(f"[Coordinator] {speaker_name} disconnected during reply")
                self.connections.pop(speaker_name, None)
                continue

            if reply_msg is None:
                continue

            text = reply_msg.get("text", "").strip()
            if not text:
                continue

            self._log(speaker_name, text)
            print(f"[{speaker_name}]: {text}\n")
            await self._broadcast(speaker_name, text)
            self.last_speaker = speaker_name
            self.timed_out.discard(speaker_name)
            self._turn_count += 1
            if self._turn_count % 10 == 0:
                for coro in (self._generate_session_summary(), self._broadcast_song_state()):
                    t = asyncio.create_task(coro)
                    self._tasks.add(t)
                    t.add_done_callback(self._tasks.discard)

            # Pipeline: pick next speaker and send speak BEFORE waiting for TTS
            # Moderator is CPU-only (~50-100ms), so this is fast
            try:
                async with self._llm_lock:
                    next_moderator_raw = await asyncio.to_thread(self._call_moderator)
            except Exception as e:
                print(f"[Coordinator] ERROR calling moderator: {e}")
                next_moderator_raw = ""

            registered = list(self.connections.keys())
            available = [n for n in registered if n not in self.timed_out and n not in self.sleeping_bots]
            if not available:
                self.timed_out.clear()
                available = [n for n in registered if n not in self.sleeping_bots]

            if available:
                next_speaker = pick_next_speaker(next_moderator_raw, available, speaker_name)
                next_ws = self.connections.get(next_speaker)
                if next_ws is not None:
                    self.turn_id += 1
                    next_turn = self.turn_id
                    try:
                        await next_ws.send(json.dumps({"type": "speak", "turn_id": next_turn}))
                        pending_speak = (next_speaker, next_ws, next_turn)
                        print(f"  [Coordinator] Pre-dispatched speak to {next_speaker} (generating during TTS)")
                    except websockets.ConnectionClosed:
                        print(f"[Coordinator] Lost connection to {next_speaker}")
                        self.connections.pop(next_speaker, None)

            # Tell current speaker to play TTS now (next bot generates concurrently)
            try:
                await ws.send(json.dumps({"type": "play_tts", "turn_id": current_turn}))
            except websockets.ConnectionClosed:
                self.connections.pop(speaker_name, None)

            # Wait for tts_done — current bot's TTS plays while next bot generates
            try:
                tts_deadline = asyncio.get_event_loop().time() + self.reply_timeout
                _tts_q = self._bot_queues.get(speaker_name)
                while True:
                    if _tts_q is None:
                        break
                    remaining = tts_deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        print(f"  [Coordinator] WARNING: tts_done timeout from {speaker_name}, proceeding")
                        break
                    raw_tts = await asyncio.wait_for(_tts_q.get(), timeout=remaining)
                    if raw_tts is None:
                        print(f"  [Coordinator] {speaker_name} disconnected waiting for tts_done")
                        self.connections.pop(speaker_name, None)
                        break
                    try:
                        tts_msg = json.loads(raw_tts)
                    except json.JSONDecodeError:
                        print(f"  [Coordinator] WARNING: invalid JSON tts_done from {speaker_name}")
                        continue
                    if tts_msg.get("turn_id") != current_turn:
                        print(f"  [Coordinator] Draining stale message from {speaker_name} "
                              f"(turn {tts_msg.get('turn_id')} != {current_turn})")
                        continue
                    if tts_msg.get("type") == "tts_done":
                        break
                    print(f"  [Coordinator] WARNING: expected tts_done from {speaker_name}, "
                          f"got {tts_msg.get('type')}")
                    break
            except asyncio.TimeoutError:
                print(f"  [Coordinator] WARNING: tts_done timeout from {speaker_name}, proceeding")
            except websockets.ConnectionClosed:
                print(f"  [Coordinator] {speaker_name} disconnected waiting for tts_done")
                self.connections.pop(speaker_name, None)

            # If pending speaker disconnected during TTS, clear it
            if pending_speak and pending_speak[0] not in self.connections:
                pending_speak = None

    async def broadcast_model_swap(self, slot):
        for name in list(self.connections.keys()):
            ws = self.connections.get(name)
            if ws is None:
                continue
            try:
                _swap_q = self._bot_queues.get(name)
                if _swap_q is None:
                    continue
                # Drain stale messages (e.g. sleep closing-line replies) before
                # sending swap_model so we don't consume them as swap_done.
                drained = 0
                while not _swap_q.empty():
                    try:
                        stale = _swap_q.get_nowait()
                        if stale is None:
                            await _swap_q.put(None)  # put sentinel back — bot disconnected
                            break
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    print(f"[Coordinator] Drained {drained} stale message(s) from {name} before swap→{slot}")
                await ws.send(json.dumps({"type": "swap_model", "slot": slot}))
                try:
                    raw = await asyncio.wait_for(_swap_q.get(), timeout=600)
                    if raw is None:
                        print(f"[Coordinator] {name} disconnected during swap")
                    else:
                        msg = json.loads(raw)
                        if msg.get("type") == "swap_done":
                            print(f"[Coordinator] {name} swap complete")
                        else:
                            print(f"[Coordinator] WARNING: unexpected response from {name}: {msg.get('type')}")
                except asyncio.TimeoutError:
                    print(f"[Coordinator] WARNING: {name} swap_done timeout")
            except websockets.ConnectionClosed:
                print(f"[Coordinator] WARNING: {name} disconnected during swap")
            await asyncio.sleep(2)

    async def set_bot_sleep(self, name):
        ws = self.connections.get(name)
        if ws is None:
            print(f"[Coordinator] WARNING: {name} not connected, cannot send sleep")
            return
        try:
            await ws.send(json.dumps({"type": "sleep"}))
            self.sleeping_bots.add(name)
            print(f"[Coordinator] Sent sleep to {name}")
        except websockets.ConnectionClosed:
            print(f"[Coordinator] WARNING: {name} disconnected when sending sleep")

    async def set_bot_wake(self, name):
        self.sleeping_bots.discard(name)  # always clear — bot may reconnect after this fires
        ws = self.connections.get(name)
        if ws is None:
            print(f"[Coordinator] WARNING: {name} not connected, marking as awake for when it reconnects")
            return
        try:
            await ws.send(json.dumps({"type": "wake"}))
            print(f"[Coordinator] Sent wake to {name}")
        except websockets.ConnectionClosed:
            print(f"[Coordinator] WARNING: {name} disconnected when sending wake")

    async def set_mode(self, mode):
        payload = json.dumps({"type": "mode", "mode": mode})
        for name, ws in list(self.connections.items()):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                print(f"[Coordinator] WARNING: {name} disconnected during set_mode")
        print(f"[Coordinator] Mode set: {mode}")

    def _human_input_worker(self):
        while True:
            try:
                line = input()
                if line.strip():
                    self.loop.call_soon_threadsafe(
                        lambda l=line.strip(): asyncio.ensure_future(self._inject(l))
                    )
            except EOFError:
                break

    async def _inject(self, text):
        self._log("[Human]", text)
        payload = json.dumps({"type": "inject", "text": text})
        for ws in list(self.connections.values()):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                pass
        print(f"  [Human injected]: {text}")

    async def run(self, host, port):
        self.loop = asyncio.get_running_loop()
        self._load_lock = asyncio.Lock()
        self._all_registered = asyncio.Event()
        self.pause_gate = asyncio.Event()
        self.pause_gate.set()  # set = proceed; clear = pause

        input_thread = threading.Thread(target=self._human_input_worker, daemon=True)
        input_thread.start()

        print(f"Coordinator listening on ws://{host}:{port}")
        print(f"Waiting for {self.expected_bots} bots to connect...")

        async with websockets.serve(
            self.handle_registration, host, port,
            ping_interval=120,
            ping_timeout=120,
        ):
            try:
                all_ready_timeout = self.register_timeout * max(self.expected_bots, 1)
                await asyncio.wait_for(self.ready_event.wait(), timeout=all_ready_timeout)
            except asyncio.TimeoutError:
                print(f"ERROR: Only {len(self.connections)}/{self.expected_bots} bots registered within timeout.")
                return
            await self.moderation_loop()


def main():
    parser = argparse.ArgumentParser(description="Coordinator for four-bot conversation system")
    parser.add_argument("--model", required=True, help="Path to moderator GGUF model")
    parser.add_argument("--bots", type=int, required=True, help="Number of bots to wait for")
    parser.add_argument("--register-timeout", type=int, default=300)
    parser.add_argument("--reply-timeout", type=int, default=120)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-layers", type=int, default=None)
    parser.add_argument("--opener-prompt", type=str, default=None,
                        help="Custom opening prompt. Use {name} for the speaker's name.")
    parser.add_argument("--osc-ip", type=str, default=None,
                        help="IP address for OSC output (ports 9020/9030/9040)")
    parser.add_argument("--conv-log", type=str, default="conv_log.jsonl",
                        help="Path for conversation log file")
    args = parser.parse_args()

    profile = detect_hardware_profile()
    # On CUDA: moderator runs CPU-only so all VRAM goes to bots.
    # On Metal: unified memory, no VRAM to protect — use GPU acceleration.
    if args.gpu_layers is not None:
        n_gpu_layers = args.gpu_layers
    elif profile.name == "METAL":
        n_gpu_layers = -1
    else:
        n_gpu_layers = 0
    print(f"Coordinator hardware: {profile.name} (n_gpu_layers={n_gpu_layers}, n_ctx={MODERATOR_N_CTX})")

    print(f"Loading moderator model: {os.path.basename(args.model)}")
    llm = Llama(
        model_path=args.model,
        n_ctx=MODERATOR_N_CTX,
        n_gpu_layers=n_gpu_layers,
        n_batch=256,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )

    osc_sender = None
    if args.osc_ip is not None:
        from osc_send import OSC_Sender
        osc_sender = OSC_Sender(args.osc_ip)

    coordinator = Coordinator(
        moderator_llm=llm,
        expected_bots=args.bots,
        reply_timeout=args.reply_timeout,
        register_timeout=args.register_timeout,
        opener_prompt=args.opener_prompt,
        osc_sender=osc_sender,
        conv_log_path=args.conv_log,
    )
    asyncio.run(coordinator.run(args.host, args.port))


if __name__ == "__main__":
    main()
