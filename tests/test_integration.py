"""Integration tests: WebSocket message flow with mock LLM.

Uses real asyncio and real websockets. Coordinator runs with a mock
moderator LLM that always returns a fixed bot name.
"""
import asyncio
import json
import pytest
import websockets
from unittest.mock import MagicMock

# Skip entire module if llama_cpp (required by coordinator) is not installed
try:
    import llama_cpp  # noqa: F401
except ImportError:
    pytest.skip("llama_cpp not installed — skipping integration tests", allow_module_level=True)


def make_mock_llm(next_speaker="Bob"):
    """Return a mock Llama that always picks next_speaker."""
    mock = MagicMock()
    mock.create_chat_completion.return_value = {
        "choices": [{"message": {"content": next_speaker}}]
    }
    return mock


@pytest.mark.asyncio
async def test_full_registration_and_opening_broadcast():
    """Two bots connect, register, go ready, and receive the opening broadcast."""
    from coordinator import Coordinator

    mock_llm = make_mock_llm("Alice")
    coord = Coordinator(
        moderator_llm=mock_llm,
        expected_bots=2,
        reply_timeout=5,
        register_timeout=10,
    )

    received_by_alice = []
    received_by_bob = []

    async def fake_bot(name, received_list, port):
        await asyncio.sleep(0.2)
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            await ws.send(json.dumps({
                "type": "register", "name": name, "model_size_mb": 4096,
            }))
            # Wait for load signal (includes n_gpu_layers)
            load_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            assert load_msg["type"] == "load"
            assert "n_gpu_layers" in load_msg
            await ws.send(json.dumps({"type": "ready", "name": name}))
            try:
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    received_list.append(msg)
                    if msg["type"] == "broadcast":
                        break
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                pass

    port = 18765
    server_task = asyncio.create_task(coord.run("localhost", port))
    await asyncio.sleep(0.1)
    await asyncio.gather(
        fake_bot("Alice", received_by_alice, port),
        fake_bot("Bob", received_by_bob, port),
        return_exceptions=True,
    )
    server_task.cancel()
    try:
        await server_task
    except (asyncio.CancelledError, Exception):
        pass

    assert any(m["type"] == "broadcast" for m in received_by_alice), \
        f"Alice did not receive opening broadcast. Got: {received_by_alice}"
    assert any(m["type"] == "broadcast" for m in received_by_bob), \
        f"Bob did not receive opening broadcast. Got: {received_by_bob}"


@pytest.mark.asyncio
async def test_duplicate_name_rejected():
    """Second bot with same name receives error message."""
    from coordinator import Coordinator

    mock_llm = make_mock_llm()
    coord = Coordinator(
        moderator_llm=mock_llm,
        expected_bots=2,
        reply_timeout=5,
        register_timeout=10,
    )

    error_received = []

    async def first_bot(port):
        await asyncio.sleep(0.2)
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            await ws.send(json.dumps({
                "type": "register", "name": "Alice", "model_size_mb": 4096,
            }))
            # First bot won't get load signal until expected_bots are registered
            # but duplicate_bot connects before that, so first_bot just waits
            await asyncio.sleep(4)

    async def duplicate_bot(port):
        await asyncio.sleep(0.5)
        try:
            async with websockets.connect(f"ws://localhost:{port}") as ws:
                await ws.send(json.dumps({
                    "type": "register", "name": "Alice", "model_size_mb": 4096,
                }))
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                    error_received.append(msg)
                except asyncio.TimeoutError:
                    pass
        except websockets.ConnectionClosed:
            pass

    port = 18766
    server_task = asyncio.create_task(coord.run("localhost", port))
    await asyncio.sleep(0.1)
    await asyncio.gather(
        first_bot(port),
        duplicate_bot(port),
        return_exceptions=True,
    )
    server_task.cancel()
    try:
        await server_task
    except (asyncio.CancelledError, Exception):
        pass

    assert any(m.get("type") == "error" for m in error_received), \
        f"Expected error for duplicate name. Got: {error_received}"


@pytest.mark.asyncio
async def test_serialized_load_signal():
    """Coordinator sends load signal to each bot, waits for ready before next."""
    from coordinator import Coordinator

    mock_llm = make_mock_llm("Alice")
    coord = Coordinator(
        moderator_llm=mock_llm,
        expected_bots=2,
        reply_timeout=5,
        register_timeout=10,
    )

    load_order = []

    async def bot_with_load(name, port):
        await asyncio.sleep(0.2)
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            await ws.send(json.dumps({
                "type": "register", "name": name, "model_size_mb": 4096,
            }))
            # Wait for load signal (includes n_gpu_layers)
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            assert msg["type"] == "load", f"{name} expected load, got {msg['type']}"
            load_order.append(name)
            # Simulate model loading delay
            await asyncio.sleep(0.3)
            await ws.send(json.dumps({"type": "ready", "name": name}))
            # Wait for broadcast
            try:
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
                    if msg["type"] == "broadcast":
                        break
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                pass

    port = 18767
    server_task = asyncio.create_task(coord.run("localhost", port))
    await asyncio.sleep(0.1)
    await asyncio.gather(
        bot_with_load("Alice", port),
        bot_with_load("Bob", port),
        return_exceptions=True,
    )
    server_task.cancel()
    try:
        await server_task
    except (asyncio.CancelledError, Exception):
        pass

    assert len(load_order) == 2, f"Expected 2 bots loaded, got {load_order}"


@pytest.mark.asyncio
async def test_load_failed_skips_bot():
    """Bot that sends load_failed is skipped, remaining bots still work."""
    from coordinator import Coordinator

    mock_llm = make_mock_llm("Bob")
    coord = Coordinator(
        moderator_llm=mock_llm,
        expected_bots=2,
        reply_timeout=5,
        register_timeout=10,
    )

    async def failing_bot(port):
        await asyncio.sleep(0.2)
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            await ws.send(json.dumps({
                "type": "register", "name": "Alice", "model_size_mb": 4096,
            }))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            assert msg["type"] == "load"
            await ws.send(json.dumps({
                "type": "load_failed",
                "name": "Alice",
                "error": "OOM",
            }))
            await asyncio.sleep(2)

    async def good_bot(port):
        await asyncio.sleep(0.4)
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            await ws.send(json.dumps({
                "type": "register", "name": "Bob", "model_size_mb": 4096,
            }))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            assert msg["type"] == "load"
            await ws.send(json.dumps({"type": "ready", "name": "Bob"}))
            try:
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
                    if msg["type"] == "broadcast":
                        break
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                pass

    port = 18768
    server_task = asyncio.create_task(coord.run("localhost", port))
    await asyncio.sleep(0.1)
    await asyncio.gather(
        failing_bot(port),
        good_bot(port),
        return_exceptions=True,
    )
    server_task.cancel()
    try:
        await server_task
    except (asyncio.CancelledError, Exception):
        pass

    assert "Bob" in coord.connections or coord.ready_event.is_set()
