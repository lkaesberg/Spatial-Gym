"""Spatial-Gym as a verifiers environment.

Each task is one SPaRC test puzzle. The model walks the grid one move per turn,
optionally with backtracking.

The reward comes from the existing Spatial-Gym environment: an episode scores
1.0 only if the walked path exactly matches one of the puzzle's stored
solutions (`outcome_reward == 1` in `Spatial_Gym.step`), otherwise 0.0.
"""

import asyncio
import copy
import json
from typing import Any

import gymnasium as gym
import verifiers.v1 as vf

import Spatial_Gym  # noqa: F401  (registers the "Spatial-Gym" gymnasium id)
from spatial_gym_env.prompts import (
    STEP_USER_TEMPLATE,
    parse_action,
    step_backtrack_prompt,
    step_prompt,
)

DATASET = "lkaesberg/SPaRC"
# Pinned Hugging Face commit of the dataset (2026-08-02).
DATASET_REVISION = "e2d908363190e9a244f931df2654a501facc90a7"
ACTION_NAMES = {0: "RIGHT", 1: "UP", 2: "LEFT", 3: "DOWN"}
MOVES = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}
RETRY_MESSAGE = (
    "Your last reply did not end with a valid action. Reply again and end with:\n"
    "Final: <action>\nWhere <action> is one of 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN."
)

# Building a Spatial-Gym env loads and preprocesses the whole dataset (~2 s), and
# reusing one env across episodes leaks visited cells (see NOTES.md). So we keep
# one untouched template per (backtracking, max_steps) and deep-copy it per episode.
_TEMPLATES: dict[tuple[bool, int], gym.Env] = {}


def _fresh_env(backtracking: bool, max_steps: int) -> gym.Env:
    key = (backtracking, max_steps)
    if key not in _TEMPLATES:
        _TEMPLATES[key] = gym.make(
            "Spatial-Gym",
            df_name=DATASET,
            df_split="all",
            df_set="test",
            render_mode=None,
            observation="SPaRC",
            traceback=backtracking,
            max_steps=max_steps,
        )
    return copy.deepcopy(_TEMPLATES[key])


def _reset(env: gym.Env, puzzle_id: str) -> tuple[Any, dict]:
    obs, info = env.reset(options={"puzzle_id": puzzle_id})
    inner = env.unwrapped
    if inner.puzzles[inner.current_puzzle_index]["id"] != puzzle_id:
        raise ValueError(f"puzzle {puzzle_id} is not in the Spatial-Gym dataset")
    return obs, info


def _step_message(step: int, obs: str, info: dict) -> str:
    x, y = (int(v) for v in info["agent_location"])
    legal = ", ".join(f"{a}={ACTION_NAMES[a]}" for a in info["legal_actions"])
    grid = "\n".join(str(row) for row in json.loads(obs))
    return STEP_USER_TEMPLATE.format(
        step=step, agent_loc=f"({x}, {y})", legal_str=legal, obs_str=grid
    )


def _interleave_by_level(rows: list[dict]) -> list[dict]:
    """Round-robin over difficulty levels so `-n k` covers all levels evenly."""
    by_level: dict[int, list[dict]] = {}
    for row in rows:
        by_level.setdefault(int(row["difficulty_level"]), []).append(row)
    queues = [by_level[k] for k in sorted(by_level)]
    longest = max((len(q) for q in queues), default=0)
    return [q[i] for i in range(longest) for q in queues if i < len(q)]


class SpatialGymData(vf.TaskData):
    puzzle_id: str
    difficulty: int
    puzzle: dict
    """The SPaRC record fields the prompts read (grid_size, puzzle_array, polyshapes)."""


class SpatialGymTask(vf.Task[SpatialGymData]):
    """Scored by the env: it records the `solved` reward after each episode."""


class SpatialGymConfig(vf.TasksetConfig):
    backtracking: bool = False
    """Allow stepping back onto the previous cell to undo a move."""
    max_steps: int = 100
    """The episode ends after this many moves (paper: 100)."""
    difficulty: list[int] | None = None
    """Only use puzzles of these difficulty levels (1-5). Default: all 500 test puzzles."""
    max_invalid_retries: int = 2
    """How often to re-ask when a reply has no parsable action."""


class SpatialGymTaskset(vf.Taskset[SpatialGymTask, SpatialGymConfig]):
    def load(self) -> list[SpatialGymTask]:
        from datasets import load_dataset

        cfg = self.config
        rows = load_dataset(DATASET, "all", split="test", revision=DATASET_REVISION)
        rows = [
            r
            for r in rows
            if cfg.difficulty is None or r["difficulty_level"] in cfg.difficulty
        ]
        tasks = []
        for row in _interleave_by_level(rows):
            puzzle = {k: row[k] for k in ("grid_size", "puzzle_array", "polyshapes")}
            prompt = (step_backtrack_prompt if cfg.backtracking else step_prompt)(row)
            data = SpatialGymData(
                id=row["id"],
                name=f"{row['id']} (level {row['difficulty_level']})",
                # The env opens the chat itself with the first board state.
                prompt=None,
                system_prompt=prompt["system"],
                puzzle_id=row["id"],
                difficulty=int(row["difficulty_level"]),
                puzzle=puzzle,
            )
            tasks.append(SpatialGymTask(data, cfg.task))
        return tasks


class SpatialGymEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"}, runtime=vf.SubprocessConfig()
    )
    """The model under test: a plain chat program (no tools), run locally."""


class SpatialGymEnv(vf.Env[SpatialGymEnvConfig]):
    async def run(self, task: SpatialGymTask, agents: vf.Agents) -> None:
        cfg = self.taskset.config
        env = await asyncio.to_thread(_fresh_env, cfg.backtracking, cfg.max_steps)
        obs, info = _reset(env, task.data.puzzle_id)
        steps = retries = 0
        invalid_reply = False
        async with agents.agent.interaction(task) as interaction:
            segment = await interaction.turn(_step_message(1, obs, info))
            while not segment.terminated:
                try:
                    action = parse_action(
                        (segment.last_reply or "").split("</think>")[-1].strip()
                    )
                except ValueError:
                    # SPaRC's harness re-asked the model when no action could be parsed.
                    if retries >= cfg.max_invalid_retries:
                        invalid_reply = True
                        break
                    retries += 1
                    segment = await interaction.turn(RETRY_MESSAGE)
                    continue
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
                segment = await interaction.turn(_step_message(steps + 1, obs, info))
            inner = env.unwrapped
            trace = interaction.trace
            trace.record_metric("steps", float(steps))
            trace.record_metric("retries", float(retries))
            trace.record_metric("invalid_reply", float(invalid_reply))
            at = tuple(int(v) for v in inner._agent_location)
            trace.record_metric(
                "reached_end", float(at == tuple(int(v) for v in inner.target_location))
            )
            _record(trace, task, inner.outcome_reward == 1)


def _record(trace: vf.Trace, task: SpatialGymTask, solved: bool) -> None:
    # Recorded inside the interaction so the reward is final when the rollout closes.
    trace.record_reward("solved", 1.0 if solved else 0.0, 1.0)
    trace.record_metric("difficulty", float(task.data.difficulty))
    trace.info["puzzle_id"] = task.data.puzzle_id
