"""band_coordinator.py — single entry point: starts coordinator, DayArc, and bot subprocesses."""
import argparse
import asyncio
import json
import os
import subprocess
import sys

from coordinator import Coordinator
from day_arc import DayArc
from document_creator.song_document import SongDocument
from document_creator.ep_document import EPDocument
from resource_manager import ResourceManager
from hardware import detect_hardware_profile, register_cuda_dll_dirs

register_cuda_dll_dirs()
from llama_cpp import Llama

MODERATOR_N_CTX = 2048
COORDINATOR_HOST = "0.0.0.0"
COORDINATOR_PORT = 8765
REGISTER_TIMEOUT = 600
REPLY_TIMEOUT = 120
BOT_NAMES = ["singer", "guitarist", "bassist", "drummer"]


def _launch_bot(name, args):
    per_bot_model = getattr(args, f"{name}_model", None) or args.bot_model
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "bot.py"),
        "--name", name,
        "--prompt", f"prompts/{name}.txt",
        "--model", per_bot_model,
        "--coordinator", f"ws://localhost:{COORDINATOR_PORT}",
    ]
    if args.small_model:
        cmd += ["--small-model", args.small_model]
    if args.osc_ip:
        cmd += ["--osc-ip", args.osc_ip]
    if args.no_tts:
        cmd += ["--no-tts"]
    if args.language != "en":
        cmd += ["--language", args.language]
    print(f"[BandCoordinator] Launching: {name}")
    return subprocess.Popen(cmd)


BOT_RESTART_MAX = 10
BOT_RESTART_COOLDOWN = 15  # seconds to wait before restarting a crashed bot subprocess

async def _monitor_bots(bot_procs, args):
    restart_counts = {name: 0 for name in bot_procs}
    while True:
        await asyncio.sleep(15)
        for name in list(bot_procs.keys()):
            proc = bot_procs[name]
            ret = proc.poll()
            if ret is not None:
                count = restart_counts[name]
                if count >= BOT_RESTART_MAX:
                    print(f"[BandCoordinator] WARNING: {name} has exited {count} times — giving up")
                    continue
                print(f"[BandCoordinator] {name} exited (code {ret}) — "
                      f"restarting in {BOT_RESTART_COOLDOWN}s (attempt {count + 1}/{BOT_RESTART_MAX})")
                await asyncio.sleep(BOT_RESTART_COOLDOWN)
                new_proc = _launch_bot(name, args)
                bot_procs[name] = new_proc
                restart_counts[name] += 1


async def run(args):
    osc_sender = None
    if args.osc_ip:
        from osc_send import OSC_Sender
        osc_sender = OSC_Sender(args.osc_ip)

    # Load moderator LLM
    profile = detect_hardware_profile()
    n_gpu_layers = -1 if profile.name == "METAL" else 0
    print(f"[BandCoordinator] Hardware: {profile.name}")
    print(f"[BandCoordinator] Loading moderator: {os.path.basename(args.moderator_model)}")
    llm = await asyncio.to_thread(
        Llama,
        model_path=args.moderator_model,
        n_ctx=MODERATOR_N_CTX,
        n_gpu_layers=n_gpu_layers,
        n_batch=256,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )

    # Documents — load() creates fresh if file absent, auto-sets _save_path
    song_doc = SongDocument.load("song_state.json")
    ep_doc = EPDocument.load()

    # Resource manager
    resource_manager = ResourceManager(BOT_NAMES)

    # Coordinator
    coordinator = Coordinator(
        moderator_llm=llm,
        expected_bots=len(BOT_NAMES),
        reply_timeout=REPLY_TIMEOUT,
        register_timeout=REGISTER_TIMEOUT,
        osc_sender=osc_sender,
        conv_log_path=args.conv_log,
    )

    # DayArc inject callback — called from async context (same event loop)
    def inject_callback(text):
        asyncio.ensure_future(coordinator._inject(text))

    day_arc = DayArc(
        inject_callback=inject_callback,
        osc_sender=osc_sender,
        song_document=song_doc,
        ep_document=ep_doc,
        resource_manager=resource_manager,
        moderator_model_path=args.moderator_model,
        coordinator=coordinator,
        bot_names=BOT_NAMES,
        day_override=args.day,
        fast_mode=args.fast,
    )

    if args.remote_bots:
        print(f"[BandCoordinator] Remote-bot mode — waiting for {len(BOT_NAMES)} bots to connect on port {COORDINATOR_PORT}")
        tasks = [coordinator.run(COORDINATOR_HOST, COORDINATOR_PORT), day_arc.run()]
        await asyncio.gather(*tasks)
    else:
        bot_procs = {name: _launch_bot(name, args) for name in BOT_NAMES}
        try:
            await asyncio.gather(
                coordinator.run(COORDINATOR_HOST, COORDINATOR_PORT),
                day_arc.run(),
                _monitor_bots(bot_procs, args),
            )
        finally:
            for name, proc in bot_procs.items():
                if proc.poll() is None:
                    print(f"[BandCoordinator] Terminating {name}")
                    proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="Band Coordinator — launches all components")
    parser.add_argument("--moderator-model", required=True,
                        help="Path to moderator GGUF model")

    # Bot models — ignored when --remote-bots is set
    parser.add_argument("--bot-model", default=None,
                        help="Path to bot GGUF model (shared fallback if per-bot model not set)")
    for _name in BOT_NAMES:
        parser.add_argument(f"--{_name}-model", default=None,
                            help=f"Model for {_name} (overrides --bot-model)")
    parser.add_argument("--small-model", default=None,
                        help="Path to smaller GGUF for overnight use (passed to local bots only)")

    parser.add_argument("--remote-bots", action="store_true",
                        help="Don't launch local bot subprocesses — wait for bots to connect from remote machines")
    parser.add_argument("--osc-ip", default="192.168.1.163",
                        help="OSC broadcast IP address")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS on all bots (local mode only)")
    parser.add_argument("--language", default="en",
                        help="Language code (e.g. en, es, fr)")
    parser.add_argument("--day", default=None,
                        help="Override active day (sunday|monday|tuesday|wednesday|thursday)")
    parser.add_argument("--fast", action="store_true",
                        help="Compress each day to ~20 real minutes for testing")
    parser.add_argument("--conv-log", default="conv_log.jsonl",
                        help="Conversation log path")
    args = parser.parse_args()

    if not os.path.isfile(args.moderator_model):
        print(f"[BandCoordinator] ERROR: moderator model not found: {args.moderator_model}")
        sys.exit(1)

    if not args.remote_bots:
        for name in BOT_NAMES:
            model = getattr(args, f"{name}_model", None) or args.bot_model
            if not model:
                print(f"[BandCoordinator] ERROR: no model specified for {name} — use --{name}-model or --bot-model")
                sys.exit(1)
            if not os.path.isfile(model):
                print(f"[BandCoordinator] ERROR: {name} model not found: {model}")
                sys.exit(1)

        if not os.path.isdir("prompts"):
            print("[BandCoordinator] ERROR: prompts/ directory not found — run band_setup.py first")
            sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
