# spatial-gym-env

Spatial-Gym tests whether a model can act on a spatial plan, not just describe one. Each episode is a SPaRC
grid puzzle: the model must draw a line from the start node to the end node that obeys every rule on the board,
moving one cell per turn. The Spatial-Gym paper evaluates eight models in three settings (one-shot,
step-by-step, and step-by-step with backtracking) on 500 episodes. Giving vision models an image of the board
instead of text lowers their solve rate by 73%, and extended chain-of-thought keeps a 3-5x accuracy advantage
over standard inference even in the step-by-step setting.

## Settings

| Setting | Config | What the model does |
|---|---|---|
| Step-by-step | default | One move per turn. Each turn shows the step number, the current position, the legal moves and the board with the path so far. The reply must end with `Final: <action>` (0=RIGHT, 1=UP, 2=LEFT, 3=DOWN). |
| Step-by-step with backtracking | `backtracking=true` | Same, but the model may step back onto the previous cell to undo a move. |

Options (`--env.taskset.<name>`):

* `backtracking`: `true` or `false`.
* `max_steps`: move limit per episode, default 100 as in the paper.
* `difficulty`: list of levels 1 to 5, for example `[1]` for the easiest puzzles. Default: all 500 test puzzles.
* `max_invalid_retries`: how often to ask again when a reply has no action, default 2.

Tasks are ordered round-robin by difficulty level, so `-n 10` gives 2 puzzles per level.

## Reward

The reward is 1.0 if the final path exactly matches one of the puzzle's stored solutions, and 0.0 otherwise. The
check is the one Spatial-Gym already uses (`outcome_reward` in `Spatial_Gym.step`). Metrics: `steps`, `reached_end`, `invalid_reply`, `retries`,
`difficulty`.

## Difference from the paper

The paper's harness sent each step as a new request: the system prompt plus the current state only. This
environment keeps the whole episode in one conversation, which is how verifiers runs multi-turn episodes. The
model therefore sees its earlier moves and replies. Scores are not directly comparable to the paper's
step-by-step numbers.

## Run

```bash
uv pip install -e environments/spatial_gym
vf-eval spatial-gym-env -m <model> --client.base-url <OpenAI-compatible URL> --client.api-key-var <ENV_VAR> \
  -n 20 --env.taskset.difficulty '[1]' --no-push
vf-eval spatial-gym-env -m <model> ... --env.taskset.backtracking true
```

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

Code: https://github.com/lkaesberg/Spatial-Gym · Dataset: https://huggingface.co/datasets/lkaesberg/SPaRC
