"""In-process tests of SpatialGymEnvironment (downloads the SPaRC dataset on first use)."""

import pytest
from datasets import load_dataset
from spatial_gym_openenv.models import SpatialGymAction
from spatial_gym_openenv.models import parse_action
from spatial_gym_openenv.server.spatial_gym_environment import SpatialGymEnvironment

MOVES = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}


@pytest.fixture(scope="module")
def test_rows():
    return list(load_dataset("lkaesberg/SPaRC", "all", split="test"))


def one_per_level(rows):
    seen = {}
    for row in rows:
        seen.setdefault(row["difficulty_level"], row)
    return [seen[level] for level in sorted(seen)]


def gold_moves(row):
    path = [(p["x"], p["y"]) for p in row["solutions"][0]["path"]]
    return [MOVES[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])]


def test_gold_solution_solves_each_level(test_rows):
    for row in one_per_level(test_rows):
        env = SpatialGymEnvironment()
        obs = env.reset(puzzle_id=row["id"])
        assert obs.puzzle_id == row["id"] and obs.difficulty == row["difficulty_level"]
        assert obs.system_prompt and obs.prompt.startswith("Step: 1\n")
        for move in gold_moves(row):
            assert not obs.done
            assert move in obs.legal_actions
            obs = env.step(SpatialGymAction(action=move))
        assert obs.done and obs.solved and obs.reached_end
        assert obs.reward == 1.0 and obs.prompt == ""
        assert env.state.solved and env.state.done
        assert env.state.path[-1] == obs.position


def test_wrong_line_scores_zero(test_rows):
    row = one_per_level(test_rows)[0]
    env = SpatialGymEnvironment()
    obs = env.reset(puzzle_id=row["id"], max_steps=3)
    illegal = next(a for a in range(4) if a not in obs.legal_actions)
    start = obs.position
    for _ in range(3):
        obs = env.step(SpatialGymAction(action=illegal))
    # An illegal move leaves the line where it is but still counts as a step.
    assert obs.position == start and obs.step == 3
    assert obs.done and not obs.solved and obs.reward == 0.0


def test_backtracking_undoes_a_move(test_rows):
    row = one_per_level(test_rows)[1]
    env = SpatialGymEnvironment()
    obs = env.reset(puzzle_id=row["id"], backtracking=True)
    first = gold_moves(row)[0]
    back = {0: 2, 1: 3, 2: 0, 3: 1}[first]
    obs = env.step(SpatialGymAction(action=first))
    assert len(env.state.path) == 2
    obs = env.step(SpatialGymAction(action=back))
    assert len(env.state.path) == 1 and obs.position == env.state.path[0]


def test_reset_options():
    env = SpatialGymEnvironment()
    a = env.reset(seed=7, difficulty=[4, 5])
    b = SpatialGymEnvironment().reset(seed=7, difficulty=[4, 5])
    assert a.puzzle_id == b.puzzle_id and a.difficulty in (4, 5)
    train = env.reset(split="train", difficulty=2, seed=1)
    assert env.state.split == "train" and train.difficulty == 2
    for bad in ({"split": "dev"}, {"difficulty": 6}, {"max_steps": 0}, {"puzle_id": "x"}, {"puzzle_id": "nope"}):
        with pytest.raises(ValueError):
            env.reset(**bad)


def test_step_needs_an_open_episode(test_rows):
    env = SpatialGymEnvironment()
    with pytest.raises(RuntimeError):
        env.step(SpatialGymAction(action=0))
    row = one_per_level(test_rows)[0]
    env.reset(puzzle_id=row["id"])
    for move in gold_moves(row):
        env.step(SpatialGymAction(action=move))
    with pytest.raises(RuntimeError):
        env.step(SpatialGymAction(action=0))


def test_sessions_are_independent(test_rows):
    row = one_per_level(test_rows)[0]
    one, two = SpatialGymEnvironment(), SpatialGymEnvironment()
    one.reset(puzzle_id=row["id"])
    two.reset(puzzle_id=row["id"])
    one.step(SpatialGymAction(action=gold_moves(row)[0]))
    assert len(one.state.path) == 2 and len(two.state.path) == 1


def test_action_parsing():
    assert parse_action("I go up.\nFinal: 1") == 1
    assert SpatialGymAction.from_reply("<think>maybe 0</think> Final: left").action == 2
    with pytest.raises(ValueError):
        SpatialGymAction.from_reply("no move here")
    with pytest.raises(ValueError):
        SpatialGymAction(action=4)
