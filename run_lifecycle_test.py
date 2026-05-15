"""run_lifecycle_test.py — bot connection lifecycle test without loading real LLMs.

Exercises the full bot lifecycle in sequence:
  1. All 4 mock bots register and receive GPU load allocation
  2. Coordinator swaps all bots to small model
  3. Coordinator sends all bots to sleep
  4. Coordinator wakes all bots
  5. Coordinator swaps all bots back to main model

Usage:
    python run_lifecycle_test.py
"""
import asyncio
import json
import sys
import os

# Force UTF-8 stdout so hardware.py's → arrow prints without codec errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import websockets
from coordinator import Coordinator

PORT = 8766  # separate from prod port 8765
BOT_NAMES = ["singer", "guitarist", "bassist", "drummer"]

_pass: list[str] = []
_fail: list[str] = []


def ok(label: str) -> None:
    print(f"  [PASS] {label}")
    _pass.append(label)


def fail(label: str, detail: str = "") -> None:
    msg = f"  [FAIL] {label}" + (f" — {detail}" if detail else "")
    print(msg)
    _fail.append(label)


async def mock_bot(name: str, done: asyncio.Event) -> None:
    """Simulate one bot through the full lifecycle protocol."""
    try:
        async with websockets.connect(
            f"ws://localhost:{PORT}", ping_interval=None
        ) as ws:
            # 1. Register
            await ws.send(json.dumps({
                "type": "register",
                "name": name,
                "model_size_mb": 4096,
            }))

            # 2. Wait for load signal
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("type") == "load":
                ok(f"{name}: received load (n_gpu_layers={msg.get('n_gpu_layers')})")
            else:
                fail(f"{name}: expected load", f"got {msg.get('type')}")
                return

            # Simulate brief model-load delay
            await asyncio.sleep(0.1)

            # 3. Send ready
            await ws.send(json.dumps({"type": "ready", "name": name}))

            # 4. Listen and respond through the lifecycle
            swap_small = False
            slept = False
            woke = False
            swap_main = False

            async for raw in ws:
                msg = json.loads(raw)
                t = msg.get("type")

                if t == "swap_model":
                    slot = msg.get("slot")
                    await asyncio.sleep(0.05)  # simulate brief reload
                    await ws.send(json.dumps({"type": "swap_done", "name": name}))
                    if slot == "small" and not swap_small:
                        ok(f"{name}: swap→small ack'd")
                        swap_small = True
                    elif slot == "main" and not swap_main:
                        ok(f"{name}: swap→main ack'd")
                        swap_main = True

                elif t == "sleep":
                    ok(f"{name}: received sleep")
                    slept = True

                elif t == "wake":
                    ok(f"{name}: received wake")
                    woke = True

                if swap_small and slept and woke and swap_main:
                    break

            if not swap_small:
                fail(f"{name}: never received swap→small")
            if not slept:
                fail(f"{name}: never received sleep")
            if not woke:
                fail(f"{name}: never received wake")
            if not swap_main:
                fail(f"{name}: never received swap→main")

    except Exception as e:
        fail(f"{name}: exception", str(e))
    finally:
        done.set()


async def lifecycle_driver(coordinator: Coordinator, all_done: asyncio.Event) -> None:
    """Wait for all bots ready, then exercise each lifecycle stage."""
    await asyncio.wait_for(coordinator.ready_event.wait(), timeout=60)
    connected = list(coordinator.connections.keys())
    print(f"\n[Test] All {len(connected)} bots ready: {', '.join(connected)}")

    ok(f"registration: {len(connected)}/{len(BOT_NAMES)} bots ready")

    await asyncio.sleep(0.2)

    print("\n[Test] ── Stage: swap all → small model ──")
    await coordinator.broadcast_model_swap("small")

    print("\n[Test] ── Stage: sleep all bots ──")
    for name in list(coordinator.connections.keys()):
        await coordinator.set_bot_sleep(name)

    await asyncio.sleep(0.5)

    print("\n[Test] ── Stage: wake all bots ──")
    for name in list(coordinator.connections.keys()):
        await coordinator.set_bot_wake(name)

    await asyncio.sleep(0.2)

    print("\n[Test] ── Stage: swap all → main model ──")
    await coordinator.broadcast_model_swap("main")

    print("\n[Test] Lifecycle sequence complete — waiting for bots to disconnect...\n")
    await asyncio.wait_for(all_done.wait(), timeout=15)


async def run() -> bool:
    coordinator = Coordinator(
        moderator_llm=None,
        expected_bots=len(BOT_NAMES),
        reply_timeout=30,
        register_timeout=30,
    )

    # Initialize asyncio attributes normally set inside coordinator.run()
    coordinator.loop = asyncio.get_running_loop()
    coordinator._load_lock = asyncio.Lock()
    coordinator._all_registered = asyncio.Event()
    coordinator.pause_gate = asyncio.Event()
    coordinator.pause_gate.set()

    done_events = [asyncio.Event() for _ in BOT_NAMES]
    all_done = asyncio.Event()

    async def _wait_all():
        await asyncio.gather(*[e.wait() for e in done_events])
        all_done.set()

    print(f"[Test] Coordinator on ws://localhost:{PORT}")
    print(f"[Test] Bots: {', '.join(BOT_NAMES)}\n")

    async with websockets.serve(
        coordinator.handle_registration,
        "localhost",
        PORT,
        ping_interval=None,
    ):
        asyncio.create_task(_wait_all())
        bot_tasks = [
            asyncio.create_task(mock_bot(name, evt))
            for name, evt in zip(BOT_NAMES, done_events)
        ]

        try:
            await lifecycle_driver(coordinator, all_done)
        except asyncio.TimeoutError:
            fail("lifecycle_driver", "timed out waiting for bots")

        # Cancel any lingering bot tasks
        for t in bot_tasks:
            if not t.done():
                t.cancel()

    print("=" * 50)
    total = len(_pass) + len(_fail)
    print(f"Results: {len(_pass)}/{total} passed")
    if _fail:
        print("Failed checks:")
        for f in _fail:
            print(f"  - {f}")
    else:
        print("ALL PASSED")
    print("=" * 50)

    return len(_fail) == 0


if __name__ == "__main__":
    success = asyncio.run(run())
    sys.exit(0 if success else 1)
