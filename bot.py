"""Bot client — connects to coordinator, runs local LLM and TTS."""
import argparse
import asyncio
import gc
import io
import os
import re
import wave
import platform
import json
import warnings

import websockets

from hardware import detect_hardware_profile, detect_whisper_device, register_cuda_dll_dirs, patch_jinja_loopcontrols
from osc_send import OSC_Sender

register_cuda_dll_dirs()      # Must run before importing llama_cpp (Python 3.8+ Windows DLL search)
patch_jinja_loopcontrols()    # Enable {% break %} in model chat templates (e.g. c4ai-command-r)
from llama_cpp import Llama
from engine import LLMEngine, _sanitize_response, _is_valid_response
from faster_whisper import WhisperModel




def parse_prompt_file(path):
    """Parse personality prompt file. Returns (system_prompt, None)."""
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    return text, None


def synthesize_wav(kokoro_pipeline, text, voice_id=None):
    """Synthesize text to WAV using Kokoro or Parler. Returns (wav_data, duration_seconds)."""
    if not text or not text.strip():
        return b"", 0.0
    import numpy as np
    audio_chunks = []
    call_kwargs = {"speed": 1.0}
    if voice_id is not None:
        call_kwargs["voice"] = voice_id
    for _, _, audio in kokoro_pipeline(text, **call_kwargs):
        audio_chunks.append(audio)
    if not audio_chunks:
        return b"", 0.0
    samples = np.concatenate(audio_chunks)
    sample_rate = getattr(kokoro_pipeline, "sample_rate", 24000)
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    wav_data = buf.getvalue()
    duration = len(pcm) / 2 / sample_rate
    return wav_data, duration


def play_wav(wav_data):
    """Play WAV bytes on the local speaker (blocking)."""
    import tempfile
    import subprocess
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_data)
    tmp.close()
    try:
        sys_name = platform.system()
        if sys_name == "Linux":
            subprocess.run(["aplay", "-q", "-D", "plughw:2,0", tmp.name], check=True)
        else:
            subprocess.run(
                ["powershell", "-c", f'(New-Object Media.SoundPlayer "{tmp.name}").PlaySync()'],
                check=True,
            )
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def synthesize_and_play(piper_voice, text):
    """Synthesize text to WAV and play on local speaker (blocking)."""
    wav_data, _ = synthesize_wav(piper_voice, text)
    if wav_data:
        play_wav(wav_data)


def build_history_entry(speaker, text, self_name):
    if speaker == self_name:
        return {"role": "assistant", "content": text}
    return {"role": "user", "content": f"{speaker}: {text}"}


def strip_name_prefix(text, name):
    """Remove bot's own name prefix from LLM response.

    LLMs see 'Bob: text' in history and mimic the pattern. Strip it.
    """
    text = re.sub(
        r'^' + re.escape(name) + r'\s*[:,\-]\s*',
        '', text, count=1, flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r'^' + re.escape(name) + r'\s*\n',
        '', text, count=1, flags=re.IGNORECASE,
    ).strip()
    return text


async def run_bot(args, kokoro_pipeline, system_prompt, voice_id=None):
    osc_sender = OSC_Sender(args.osc_ip) if args.osc_ip else None

    local_history = []
    speaking = False
    sleeping = False
    current_slot = "main"
    small_model_path = args.small_model or args.model
    performance_context = ""
    session_summary = ""
    song_state_block = ""
    llm = None
    consecutive_fails = 0
    model_reloading = False

    try:
        async with websockets.connect(
            args.coordinator,
            ping_interval=None,
        ) as ws:
            model_size_mb = int(os.path.getsize(args.model) / (1024 * 1024))
            await ws.send(json.dumps({
                "type": "register",
                "name": args.name,
                "model_size_mb": model_size_mb,
            }))
            print(f"[{args.name}] Registered with coordinator (model: {model_size_mb}MB)")
            if osc_sender:
                osc_sender.send_message(f"/{args.name}/talking", "silent")
            # Wait for load signal from coordinator
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("type") != "load":
                print(f"[{args.name}] Expected load signal, got: {msg.get('type')}")
                return

            # Use coordinator-planned GPU layers (CLI override takes precedence)
            profile = detect_hardware_profile()
            if args.gpu_layers is not None:
                n_gpu_layers = args.gpu_layers
            else:
                n_gpu_layers = msg.get("n_gpu_layers", 0)
            n_ctx = profile.n_ctx

            print(f"[{args.name}] Loading LLM: {os.path.basename(args.model)} "
                  f"(n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx})")

            whisper_device, whisper_compute = detect_whisper_device()
            # Whisper is only used for word-timing during TTS playback; skip it with --no-tts
            need_whisper = kokoro_pipeline is not None
            try:
                if need_whisper:
                    llm_result, whisper = await asyncio.gather(
                        asyncio.to_thread(
                            Llama,
                            model_path=args.model,
                            n_ctx=n_ctx,
                            n_gpu_layers=n_gpu_layers,
                            n_batch=512,
                            use_mmap=True,
                            use_mlock=False,
                            verbose=False,
                        ),
                        asyncio.to_thread(WhisperModel, "tiny", device=whisper_device, compute_type=whisper_compute),
                    )
                    llm = llm_result
                    print(f"[{args.name}] Whisper loaded ({whisper_device})")
                else:
                    llm = await asyncio.to_thread(
                        Llama,
                        model_path=args.model,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        n_batch=512,
                        use_mmap=True,
                        use_mlock=False,
                        verbose=False,
                    )
                    whisper = None
            except Exception as e:
                print(f"[{args.name}] Failed to load model: {e}")
                await ws.send(json.dumps({
                    "type": "load_failed",
                    "name": args.name,
                    "error": str(e),
                }))
                return

            await ws.send(json.dumps({"type": "ready", "name": args.name}))
            print(f"[{args.name}] Ready")

            async def _reload_model():
                nonlocal llm, consecutive_fails, model_reloading
                reload_path = args.model if current_slot == "main" else small_model_path
                reload_layers = n_gpu_layers if current_slot == "main" else 0
                print(f"[{args.name}] Reloading model after {consecutive_fails} consecutive failures...")
                old = llm
                llm = None
                del old
                gc.collect()
                try:
                    new_llm = await asyncio.to_thread(
                        Llama,
                        model_path=reload_path,
                        n_ctx=n_ctx,
                        n_gpu_layers=reload_layers,
                        n_batch=512,
                        use_mmap=True,
                        use_mlock=False,
                        verbose=False,
                    )
                    llm = new_llm
                    consecutive_fails = 0
                    print(f"[{args.name}] Model reload complete")
                except Exception as e:
                    print(f"[{args.name}] Model reload failed: {e}")
                finally:
                    model_reloading = False

            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    print(f"[{args.name}] WARNING: received invalid JSON, skipping")
                    continue
                msg_type = msg.get("type")

                if msg_type == "speak":
                    if sleeping:
                        print(f"[{args.name}] Sleeping — ignoring speak signal")
                        continue
                    if speaking:
                        print(f"[{args.name}] WARNING: received speak while already speaking, ignoring")
                        continue
                    if model_reloading or llm is None:
                        print(f"[{args.name}] Model reloading — skipping speak signal")
                        continue
                    speaking = True
                    turn_id = msg.get("turn_id")

                    try:
                        extra = ""
                        if song_state_block:
                            extra += f"\n\n{song_state_block}"
                        if session_summary:
                            extra += f"\n\n=== SESSION SO FAR ===\n{session_summary}\n====================="
                        msgs = LLMEngine._build_messages(
                            system_prompt + performance_context + extra,
                            local_history[-10:],
                        )
                        result = await asyncio.to_thread(
                            llm.create_chat_completion,
                            messages=msgs,
                            max_tokens=220,
                            temperature=0.75,
                            top_p=0.95,
                            repeat_penalty=1.25,
                            top_k=40,
                        )
                        reply = _sanitize_response(result["choices"][0]["message"]["content"].strip())
                        reply = strip_name_prefix(reply, args.name)
                    except Exception as e:
                        print(f"[{args.name}] ERROR during generation: {e}")
                        consecutive_fails += 1
                        if consecutive_fails >= 3 and not model_reloading:
                            model_reloading = True
                            asyncio.create_task(_reload_model())
                        speaking = False
                        continue

                    if not _is_valid_response(reply):
                        print(f"[{args.name}] WARNING: invalid response, retrying with safer params")
                        try:
                            result = await asyncio.to_thread(
                                llm.create_chat_completion,
                                messages=msgs,  # already windowed from first attempt
                                max_tokens=220,
                                temperature=0.5,
                                top_p=0.95,
                                repeat_penalty=1.3,
                                top_k=40,
                            )
                            reply = _sanitize_response(result["choices"][0]["message"]["content"].strip())
                            reply = strip_name_prefix(reply, args.name)
                        except Exception as e:
                            print(f"[{args.name}] ERROR during retry generation: {e}")
                            consecutive_fails += 1
                            if consecutive_fails >= 3 and not model_reloading:
                                model_reloading = True
                                asyncio.create_task(_reload_model())
                            speaking = False
                            continue
                        if not _is_valid_response(reply):
                            print(f"[{args.name}] WARNING: retry also invalid, skipping turn")
                            speaking = False
                            continue

                    consecutive_fails = 0
                    # Send reply immediately so coordinator can broadcast + pre-dispatch next
                    await ws.send(json.dumps({
                        "type": "reply",
                        "name": args.name,
                        "turn_id": turn_id,
                        "text": reply,
                    }))

                    # Wait for play_tts signal before playing audio (prevents overlap).
                    # Broadcasts may arrive in the meantime — process them.
                    play_deadline = asyncio.get_event_loop().time() + 60.0
                    try:
                        while asyncio.get_event_loop().time() < play_deadline:
                            remaining = play_deadline - asyncio.get_event_loop().time()
                            inner_raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                            try:
                                inner_msg = json.loads(inner_raw)
                            except json.JSONDecodeError:
                                continue
                            inner_type = inner_msg.get("type")
                            if inner_type == "play_tts" and inner_msg.get("turn_id") == turn_id:
                                break
                            elif inner_type == "broadcast":
                                bspeaker = inner_msg["speaker"]
                                btext = inner_msg["text"]
                                if _is_valid_response(btext):
                                    local_history.append(build_history_entry(bspeaker, btext, args.name))
                                else:
                                    print(f"[{args.name}] WARNING: dropping invalid broadcast from {bspeaker}")
                            elif inner_type == "error":
                                print(f"[{args.name}] ERROR from coordinator: {inner_msg.get('text')}")
                                speaking = False
                                break
                    except asyncio.TimeoutError:
                        print(f"[{args.name}] WARNING: play_tts timeout, proceeding anyway")

                    # Play TTS synchronously — blocks until audio finishes
                    if kokoro_pipeline is not None:
                        try:
                            wav_data, _ = await asyncio.to_thread(synthesize_wav, kokoro_pipeline, reply, voice_id)
                            if wav_data:
                                def _run_whisper():
                                    segs_gen, _ = whisper.transcribe(
                                        io.BytesIO(wav_data),
                                        word_timestamps=True,
                                        language=args.language,
                                    )
                                    return list(segs_gen)

                                segments = await asyncio.to_thread(_run_whisper)
                                word_times = []
                                for seg in segments:
                                    for w in (seg.words or []):
                                        word_times.append((w.start, w.word.strip()))

                                async def send_words():
                                    t0 = asyncio.get_event_loop().time()
                                    for start, word in word_times:
                                        wait = start - (asyncio.get_event_loop().time() - t0)
                                        if wait > 0:
                                            await asyncio.sleep(wait)
                                        if osc_sender is not None:
                                            osc_sender.send_message(f"/{args.name}/word", word)

                                if osc_sender is not None:
                                    osc_sender.send_message(f"/{args.name}/talking", "speaking")
                                await asyncio.gather(
                                    asyncio.to_thread(play_wav, wav_data),
                                    send_words(),
                                )
                        except Exception as e:
                            print(f"[{args.name}] TTS error: {e}")
                        finally:
                            if osc_sender is not None:
                                osc_sender.send_message(f"/{args.name}/talking", "silent")

                    # Signal coordinator that audio is done
                    await ws.send(json.dumps({
                        "type": "tts_done",
                        "name": args.name,
                        "turn_id": turn_id,
                    }))
                    speaking = False

                elif msg_type == "broadcast":
                    speaker = msg["speaker"]
                    text = msg["text"]
                    if _is_valid_response(text):
                        local_history.append(build_history_entry(speaker, text, args.name))
                    else:
                        print(f"[{args.name}] WARNING: dropping invalid broadcast from {speaker}")

                elif msg_type == "inject":
                    raw_text = msg["text"]
                    local_history.append({"role": "user", "content": f"[Human]: {raw_text}"})

                elif msg_type == "swap_model":
                    slot = msg.get("slot", "main")
                    if model_reloading:
                        print(f"[{args.name}] Model reloading — skipping swap to {slot}")
                        await ws.send(json.dumps({"type": "swap_done", "name": args.name}))
                        continue
                    if slot == current_slot:
                        print(f"[{args.name}] Already on slot {slot}, skipping swap")
                        await ws.send(json.dumps({"type": "swap_done", "name": args.name}))
                        continue
                    new_path = args.model if slot == "main" else small_model_path
                    print(f"[{args.name}] Swapping to slot={slot}: {os.path.basename(new_path)}")
                    llm = None  # clear before load so llm is never unbound on failure
                    gc.collect()
                    n_layers = n_gpu_layers if slot == "main" else 0
                    try:
                        llm = await asyncio.to_thread(
                            Llama,
                            model_path=new_path,
                            n_ctx=profile.n_ctx,
                            n_gpu_layers=n_layers,
                            n_batch=512,
                            use_mmap=True,
                            use_mlock=False,
                            verbose=False,
                        )
                        current_slot = slot
                        print(f"[{args.name}] Model swap complete → {slot}")
                    except Exception as e:
                        print(f"[{args.name}] Model swap failed: {e} — scheduling reload")
                        model_reloading = True
                        asyncio.create_task(_reload_model())
                    await ws.send(json.dumps({"type": "swap_done", "name": args.name}))

                elif msg_type == "sleep":
                    sleeping = True
                    print(f"[{args.name}] Going to sleep")
                    try:
                        extra = ""
                        if song_state_block:
                            extra += f"\n\n{song_state_block}"
                        if session_summary:
                            extra += f"\n\n=== SESSION SO FAR ===\n{session_summary}\n====================="
                        msgs = LLMEngine._build_messages(
                            system_prompt + "\n\nYou are wrapping up for the night. Say one short closing line." + extra,
                            local_history[-10:],
                        )
                        result = await asyncio.to_thread(
                            llm.create_chat_completion,
                            messages=msgs,
                            max_tokens=60,
                            temperature=0.7,
                            top_p=0.95,
                        )
                        closing = _sanitize_response(result["choices"][0]["message"]["content"].strip())
                        closing = strip_name_prefix(closing, args.name)
                        if closing:
                            await ws.send(json.dumps({
                                "type": "reply",
                                "name": args.name,
                                "turn_id": None,
                                "text": closing,
                            }))
                            if kokoro_pipeline is not None:
                                try:
                                    if osc_sender is not None:
                                        osc_sender.send_message(f"/{args.name}/talking", "speaking")
                                    wav_data, _ = await asyncio.to_thread(synthesize_wav, kokoro_pipeline, closing, voice_id)
                                    if wav_data:
                                        await asyncio.to_thread(play_wav, wav_data)
                                except Exception as e:
                                    print(f"[{args.name}] Sleep TTS error: {e}")
                                finally:
                                    if osc_sender is not None:
                                        osc_sender.send_message(f"/{args.name}/talking", "silent")
                    except Exception as e:
                        print(f"[{args.name}] Sleep line error: {e}")

                elif msg_type == "wake":
                    sleeping = False
                    print(f"[{args.name}] Waking up")

                elif msg_type == "mode":
                    mode = msg.get("mode", "conversation")
                    if mode == "performance":
                        performance_context = "\n\nYou are now in performance mode. React to the music being played."
                    else:
                        performance_context = ""
                    print(f"[{args.name}] Mode → {mode}")

                elif msg_type == "summary":
                    session_summary = msg.get("text", "")
                    print(f"[{args.name}] Session summary updated ({len(session_summary)} chars)")

                elif msg_type == "state_update":
                    song_state_block = msg.get("text", "")
                    print(f"[{args.name}] Song state updated ({len(song_state_block)} chars)")

                elif msg_type == "error":
                    print(f"[{args.name}] ERROR from coordinator: {msg.get('text')}")
                    break

                else:
                    print(f"[{args.name}] Unknown message type: {msg_type}")
    except websockets.ConnectionClosed as e:
        print(f"[{args.name}] Connection closed: {e}")
        raise  # reconnect loop handles all closes — no silent exits
    except Exception as e:
        print(f"[{args.name}] Unexpected error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bot client for four-bot conversation system")
    parser.add_argument("--name", required=True, help="Bot name")
    parser.add_argument("--prompt", required=True, help="Path to personality prompt file")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--small-model", default=None,
                        help="Path to smaller GGUF for off-peak hours (defaults to --model)")
    parser.add_argument("--voice", default=None,
                        help="Kokoro voice ID (e.g. af_heart). Overrides voice_config.json.")
    parser.add_argument("--coordinator", default="ws://localhost:8765",
                        help="Coordinator WebSocket URL")
    parser.add_argument("--gpu-layers", type=int, default=None,
                        help="Override auto-detected n_gpu_layers")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS (useful for testing without audio hardware)")
    parser.add_argument("--osc-ip", default=None,
                        help="OSC broadcast IP (omit to disable OSC)")
    parser.add_argument("--language", default="en",
                        help="Language code for LLM responses and Whisper (e.g. en, es, fr)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"[{args.name}] ERROR: Model file not found: {args.model}")
        return

    system_prompt, _ = parse_prompt_file(args.prompt)
    if args.language != "en":
        system_prompt += f"\n\nYou must respond only in {args.language}."

    kokoro_pipeline = None
    voice_id = None
    if not args.no_tts:
        voice_id = args.voice
        try:
            with open("voice_config.json") as f:
                voice_cfg = json.load(f)
            bot_cfg = voice_cfg.get(args.name.lower(), {})
            if voice_id is None:
                voice_id = bot_cfg.get("voice_id")
        except (FileNotFoundError, KeyError):
            pass

        if voice_id:
            try:
                import kokoro
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.rnn")
                    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.nn\.utils\.weight_norm")
                    try:
                        kokoro_pipeline = kokoro.KPipeline(lang_code="a", voice=voice_id, repo_id="hexgrad/Kokoro-82M", device="cpu")
                    except TypeError:
                        kokoro_pipeline = kokoro.KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
                        kokoro_pipeline.voice = voice_id
                print(f"[{args.name}] Kokoro voice loaded: {voice_id}")
            except Exception as e:
                print(f"[{args.name}] WARNING: Could not load Kokoro voice {voice_id!r}: {e}")
        else:
            print(f"[{args.name}] WARNING: No voice_id found, TTS disabled")

    import time
    _RECONNECT_DELAY = 10.0
    _RECONNECT_MAX = 50

    for attempt in range(_RECONNECT_MAX):
        print(f"[{args.name}] Connecting to {args.coordinator}"
              + (f" (attempt {attempt + 1})" if attempt else ""))
        try:
            asyncio.run(run_bot(args, kokoro_pipeline, system_prompt, voice_id=voice_id))
        except (OSError, websockets.exceptions.WebSocketException) as e:
            print(f"[{args.name}] Connection error: {e} — retrying in {_RECONNECT_DELAY}s")
            time.sleep(_RECONNECT_DELAY)
        except Exception as e:
            print(f"[{args.name}] Unexpected error: {e} — retrying in {_RECONNECT_DELAY}s")
            time.sleep(_RECONNECT_DELAY)
        else:
            break


if __name__ == "__main__":
    main()
