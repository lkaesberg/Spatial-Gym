---
title: Spatial-Gym
emoji: 🧩
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
license: mit
short_description: SPaRC grid pathfinding puzzles, one move per step
tags:
  - openenv
  - reinforcement-learning
  - spatial-reasoning
  - puzzles
---

# Spatial-Gym for OpenEnv

[Spatial-Gym](https://github.com/lkaesberg/Spatial-Gym) turns the [SPaRC](https://sparc.gipplab.org/) grid puzzles into a step-by-step environment. The agent draws one line from the start node to the exit, one move at a time, and the line has to follow every rule on the board. In the [Spatial-Gym paper](https://arxiv.org/abs/2604.09338), humans solve 98.0% of the test puzzles, and the best model, GPT-OSS 120B, solves 16.0%.

This folder serves Spatial-Gym through [OpenEnv](https://github.com/meta-pytorch/OpenEnv), so it works with any OpenEnv client or trainer. It uses the same prompts and the same scoring as the paper.

## One episode

- `reset()` loads one puzzle. The observation holds the board, the legal moves, the paper's rules prompt (`system_prompt`) and the paper's message for the first move (`prompt`).
- `step(SpatialGymAction(action=...))` makes one move: `0`=RIGHT, `1`=UP, `2`=LEFT, `3`=DOWN. A move that is not legal leaves the line where it is but still counts as a step.
- The episode ends when the line reaches the exit, when no legal move is left, or after `max_steps` moves.
- The reward is `1.0` when the episode ends with a line that exactly matches one of the puzzle's stored solutions, and `0.0` otherwise. `metadata["rule_status"]` shows which rules the line meets so far.

## Quick start

Install the client:

```bash
pip install "git+https://github.com/lkaesberg/Spatial-Gym.git#subdirectory=environments/spatial_gym_openenv"
```

Connect to a running server, then let a model play one puzzle:

```python
from spatial_gym_openenv import SpatialGymAction, SpatialGymEnv

with SpatialGymEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(difficulty=2, seed=0)
    messages = [
        {"role": "system", "content": result.observation.system_prompt},
        {"role": "user", "content": result.observation.prompt},
    ]
    while not result.done:
        reply = my_model(messages)  # your LLM call; the reply ends with "Final: <action>"
        messages.append({"role": "assistant", "content": reply})
        result = env.step(SpatialGymAction.from_reply(reply))
        if not result.done:
            messages.append({"role": "user", "content": result.observation.prompt})
    print("solved" if result.reward == 1.0 else "not solved")
```

`SpatialGymAction.from_reply` reads the move the way the paper did and raises `ValueError` when the reply has none. To run the environment from its Hugging Face Space image with Docker, use `SpatialGymEnv.from_env("<space-id>")` instead of `base_url`.

## Reset options

| Option | Default | Meaning |
|---|---|---|
| `puzzle_id` | none | Play this SPaRC puzzle |
| `difficulty` | all levels | A level from 1 to 5, or a list of levels, to draw the puzzle from |
| `seed` | none | Makes the draw repeatable |
| `split` | `"test"` | `"test"` holds the 500 benchmark puzzles, `"train"` 500 more for training |
| `backtracking` | `False` | Allow stepping back onto the previous node to undo a move |
| `max_steps` | `100` | The episode ends after this many moves, as in the paper |

## Observation

| Field | Meaning |
|---|---|
| `puzzle_id`, `difficulty` | The puzzle and its level |
| `step` | Moves taken so far |
| `position` | Current node as `[x, y]` |
| `legal_actions` | Moves allowed from here |
| `grid` | Board rows in SPaRC notation; `V` marks visited nodes and `L` the current one |
| `system_prompt` | Rules and symbol legend from the paper, sent on reset only |
| `prompt` | The paper's message for the next move, empty once the episode is over |
| `solved`, `reached_end` | The line matches a stored solution; the line ended on the exit |
| `done`, `reward` | Episode over; 1.0 for a solved puzzle, else 0.0 |

`state()` returns the episode's puzzle, options, step count and the full path.

## Run it yourself

```bash
cd environments/spatial_gym_openenv
uv run --project . server          # http://localhost:8000, web UI at /web with ENABLE_WEB_INTERFACE=true
uv run --extra dev pytest          # tests; they download the SPaRC dataset
openenv build                      # Docker image
openenv push --repo-id <user>/spatial_gym --exclude .gitignore   # Hugging Face Space
```

The server keeps one environment per WebSocket session, up to `MAX_CONCURRENT_ENVS` (default 8). It downloads the SPaRC dataset from Hugging Face on the first reset of each split.

## Citation

```bibtex
@misc{kaesberg2026spatialgym,
  title  = {Mind the Gap Between Spatial Reasoning and Acting! Step-by-Step Evaluation of Agents With Spatial-Gym},
  author = {Kaesberg, Lars Benedikt and Yang, Tianyu and Bauer, Niklas and Ruas, Terry and Wahle, Jan Philip and Gipp, Bela},
  year   = {2026},
  eprint = {2604.09338},
  archivePrefix = {arXiv}
}

@inproceedings{kaesberg-etal-2025-sparc,
  title     = "{SP}a{RC}: A Spatial Pathfinding Reasoning Challenge",
  author    = "Kaesberg, Lars Benedikt and Wahle, Jan Philip and Ruas, Terry and Gipp, Bela",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
  month     = nov,
  year      = "2025",
  address   = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2025.emnlp-main.526/",
  doi       = "10.18653/v1/2025.emnlp-main.526",
  pages     = "10359--10390"
}
```
