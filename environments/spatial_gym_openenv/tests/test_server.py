"""End-to-end test: run the FastAPI server and play a puzzle through the WebSocket client."""

import socket
import threading
import time
import urllib.request

import pytest
import uvicorn
from datasets import load_dataset
from spatial_gym_openenv import SpatialGymAction, SpatialGymEnv
from spatial_gym_openenv.server.app import app

MOVES = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield url
    server.should_exit = True
    thread.join(timeout=10)


def test_play_gold_solution_over_websocket(base_url):
    row = next(r for r in load_dataset("lkaesberg/SPaRC", "all", split="test") if r["difficulty_level"] == 3)
    path = [(p["x"], p["y"]) for p in row["solutions"][0]["path"]]
    with SpatialGymEnv(base_url=base_url).sync() as env:
        result = env.reset(puzzle_id=row["id"])
        assert result.observation.puzzle_id == row["id"] and not result.done
        assert result.observation.system_prompt
        for a, b in zip(path, path[1:]):
            result = env.step(SpatialGymAction(action=MOVES[(b[0] - a[0], b[1] - a[1])]))
        assert result.done and result.reward == 1.0 and result.observation.solved
        state = env.state()
        assert state.solved and state.step_count == len(path) - 1


def test_two_clients_get_separate_episodes(base_url):
    with SpatialGymEnv(base_url=base_url).sync() as one, SpatialGymEnv(base_url=base_url).sync() as two:
        first = one.reset(seed=3)
        two.reset(seed=3)
        one.step(SpatialGymAction(action=first.observation.legal_actions[0]))
        assert one.state().step_count == 1 and two.state().step_count == 0
