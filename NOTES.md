# NOTES: citation and benchmark-listing work (Spatial-Gym repo)

Work log for the SPaRC / Spatial-Gym citation handoff, written on 2026-10-13. The SPaRC repo has a matching
`NOTES.md` that covers Tasks B, C and E.

## Branches

The branches are stacked: each one starts from the one above it, so `NOTES.md` grows along the stack.

| Branch | Base | Content |
|---|---|---|
| `task-a-citations` | `main` | Task A: README citation (also the PyPI long description) and related-work block |
| `task-d-verifiers` | `task-a-citations` | Task D: verifiers environment in `environments/spatial_gym/` |

Both branches are pushed to `origin`. No PRs were opened.

## Code map (handoff §3)

### Spatial-Gym (`lkaesberg/Spatial-Gym`)
- **README**: `README.md`. `pyproject.toml` sets `readme = "README.md"`, so the README is also the PyPI long
  description. There is no `setup.py`.
- **Puzzle data loader**: `Spatial_Gym/Spatial_Gym.py`. `__init__` calls `load_dataset(df_name, df_split,
  split=df_set)` at line 77, and `_process_puzzles` (line 219) and `_load_puzzle` (line 95) build the arrays for
  each puzzle.
- **Path validator**: `Spatial_Gym.step` (line 1111). When an episode ends, it sets `outcome_reward = 1`
  (line 1207) only if the walked path exactly equals one of `solution_paths`. This is the same rule as SPaRC's
  `validate_solution`. `_is_on_solution_path` (line 1244) gives the +0.01 shaping reward.
  `_run_rule_validators` and `_validate_rules` (lines 901 and 941) fill `info["rule_status"]`. That output is
  informational only and is not used for the reward (see Discrepancies).
- **Renderers**: `Spatial_Gym/render/llm_renderer.py` (a pygame window, see Discrepancies) and
  `Spatial_Gym/render/human_renderer.py`.
- **Existing LLM evaluation script**: `llm_testing/llm_host.py`. Its `run_episode` (line 62) uses
  `observation='SPaRC'`, `max_steps=100` and `temperature=0.0`. The paper's runs used SPaRC's `sparc --gym`
  path instead (`sparc/process_puzzle.py:207` in the SPaRC repo). That path sends only the system prompt and the
  current state at each step, with no chat history.
- **Tests**: `tests/`. All 43 pass on `main` (checked 2026-10-13).

### SPaRC (`lkaesberg/SPaRC`), for reference
- The validator is `sparc/validation.py`: `validate_solution` (exact match against stored solutions) and
  `analyze_path` (structural checks).
- The step-by-step prompts are `sparc/prompts/gym_step.py` and `gym_step_traceback.py`.

## Task D: verifiers environment

- **Location**: `environments/spatial_gym/` (package `spatial_gym_env`, distribution `spatial-gym-env`).
- **Settings**: `mode="step"` (default), `backtracking=true`, and `mode="one_shot"`. Options: `max_steps`
  (default 100), `difficulty` (list of levels), `max_invalid_retries` (default 2).
- **Observation**: the paper's format. Each turn holds the step number, the position, the legal moves and the
  grid as Python-list rows, built from `observation='SPaRC'` with SPaRC's step prompt.
- **Reward**: 1.0 or 0.0 from the env's own exact-match check (`outcome_reward == 1`). In one-shot mode the
  answer is replayed move by move in the env. On all 500 test puzzles, plus 2,500 broken paths, this replay
  agreed with SPaRC's `validate_solution` in 3,000 of 3,000 cases.
- **Checks done**: `load_environment(backtracking=False)` and `(backtracking=True)` map to
  `SpatialGymTaskset(SpatialGymConfig(backtracking=...))`, and both build 500 tasks. The env also works with the
  PyPI release `Spatial-Gym==0.1.1` (checked in a fresh venv), which is what Hub users will install.

## OpenEnv environment (2026-10-14)

`environments/spatial_gym_openenv/` is a hand-written OpenEnv environment, scaffolded with `openenv init`
(openenv 0.8.0). `openenv import` can wrap verifiers environments now, but it turns them into MCP-style tool
actions. A native environment keeps Spatial-Gym's own action space: one move per step.

- Same prompts and parser as the verifiers environment (`server/prompts.py` copies its step prompts; the move
  parser sits in `models.py`, because clients use it too), same
  deep-copied env templates, same reward (1.0 only when `outcome_reward == 1`).
- `reset()` options: `puzzle_id`, `difficulty`, `seed`, `split` (`test` or `train`), `backtracking`,
  `max_steps` (default 100). Unknown options raise, because the OpenEnv server would otherwise drop them quietly.
- The package is `spatial_gym_openenv`, not `spatial_gym`: on macOS's case-insensitive file system a
  `spatial_gym` folder would clash with `Spatial_Gym` in site-packages. `env_name` in `server/app.py` is the
  same name, because the web UI builds its Quick Start import line from it.
- The dataset is not pinned to a revision, because `Spatial_Gym.__init__` loads the latest version itself.
- Checks: 9 pytest tests (gold solutions solve one puzzle per level, wrong lines score 0, backtracking, reset
  options, independent sessions, a WebSocket end-to-end run). `openenv validate` passes at the static level and
  against a running server. `openenv build` makes a 1.44 GB image; the container solves a level-5 puzzle in 46
  moves for reward 1.0 and passes `openenv validate --url`.
- Hugging Face needs a PRO plan for Docker Spaces on a personal account (402 on `openenv push`), so the env is
  also proposed to OpenEnv itself as `envs/spatial_gym_env` (huggingface/OpenEnv#1430); merged envs get deployed
  under the `openenv` org. Both copies have the same code layout: the server's `main()` parses its own
  arguments, as OpenEnv's `tests/envs/test_server_main_args.py` requires.

## Reproduced numbers

`vf-eval` runs on 2026-10-13 with `openai/gpt-oss-120b` through OpenRouter, step-by-step mode, level-1 puzzles
only (`--env.taskset.difficulty '[1]'`), `--sampling.max-tokens 8192`:

| Run | Episodes | Solved | Rewards | Notes |
|---|---|---|---|---|
| `-n 5` (acceptance run) | 5 | 3 | [0, 1, 1, 1, 0] | Finished. One episode had an empty reply (from the code before the retry was added), and one ended with a provider error. |
| `-n 20` (D4) | 20 | 10 | 10 × 1.0, 10 × 0.0 | 3 episodes ended with a provider error. The other 17 ended normally, and 10 of those were solved. |

All rewards are exactly 0.0 or 1.0, and easy puzzles get non-zero rewards. For comparison, the Spatial-Gym paper
reports 52.3% for GPT-OSS 120B on level 1 in the step-by-step setting (Table 3). This environment keeps the chat
history (see the guide differences), so the numbers are not directly comparable.

**Provider errors.** OpenRouter sometimes returns `finish_reason: "error"`. verifiers 0.4.0 rejects that value
when it parses the response, and the episode ends with `stop_condition = provider_error` and reward 0. The
environment does not retry these.

Outputs: `paper-boost/runs/vf/outputs/spatial-gym-env--openai--gpt-oss-120b--null--559343b0` (`-n 5`) and
`paper-boost/runs/vf/outputs/spatial-gym-env--openai--gpt-oss-120b--null--948f1d1e` (`-n 20`), each with
`traces.jsonl` and `logs/`.

Commands used:
```bash
vf-eval spatial-gym-env -m openai/gpt-oss-120b \
  --client.base-url https://openrouter.ai/api/v1 --client.api-key-var OPENROUTER_API_KEY \
  -n 20 -r 1 --env.taskset.difficulty '[1]' --sampling.max-tokens 8192 --no-push --no-rich
```

## Commands left for a human

- **PyPI release (Task A5).** The PyPI page for 0.1.1 still shows the old citation with the wrong SPaRC title.
  After `task-a-citations` is merged into `main`, cut a release. The publish workflow takes the version from the
  tag:
  ```bash
  gh release create v0.1.2 -R lkaesberg/Spatial-Gym --target main --title v0.1.2 --notes "Add citation (Spatial-Gym arXiv 2604.09338 + SPaRC EMNLP 2025), related-work section, verifiers environment"
  ```
  The version in `pyproject.toml` was intentionally not changed.
- **Prime Intellect Environments Hub (Task D6).** Log in once, then push from the repo root:
  ```bash
  prime login
  prime env push --path environments/spatial_gym --visibility PUBLIC
  ```
  The CLI reads the `verifiers>=0.4.0` pin and lists the env as v1 (not "Legacy"). The Hub name is the
  distribution name, `spatial-gym-env`. Use `--name spatial-gym` if you want the shorter name on the Hub.
- **Spatial-Gym paper, next arXiv version.** Fix the SPaRC page numbers in the bibliography (10359–10390), and
  check the step-by-step numbers listed under Discrepancies 8 and 9.

## Discrepancies found

1. **The old citation block was already gone.** The handoff says the README's "If you use Spatial-Gym, please
   cite" block uses a different SPaRC title. That block was already removed on `main` in commit `32034a0`
   (2026-03-31). The PyPI page for 0.1.1 still shows it, including the wrong title "SPaRC: Spatial Reasoning
   Challenges for Large Language Models". Task A adds the correct two-entry block. A release is needed.
2. **SPaRC page numbers.** The Spatial-Gym paper's bibliography gives SPaRC as pp. 10370–10401. The ACL
   Anthology says 10359–10390.
3. **The rule checker disagrees with the stored solutions.** Replaying each test puzzle's first stored solution
   gives `outcome_reward == 1` on 500 of 500 puzzles. But `rule_status["all_rules_satisfied"]` passes on only 242:
   `poly_ylop_area` fails on 251 and `star_pairing_exact` on 24. The reward uses exact match, which agrees with the
   paper, so the reward is not affected. The rule checker looks wrong for polyshape and star regions. It was not
   changed, per the handoff rules.
4. **`render_mode='llm'` is not a text rendering.** `Spatial_Gym/render/llm_renderer.py` opens a pygame window
   and `render()` returns `None`. The text the paper's agents saw is the `observation='SPaRC'` grid. Task D uses
   that grid instead.
5. **Resetting to an already played puzzle keeps its old state.** `_load_puzzle` reuses the puzzle's stored
   arrays, and `step()` writes "visited" marks into them. After 3 moves on a puzzle, `reset()` to the same puzzle
   still shows 4 visited cells instead of 1, and the legal moves change from `[2, 3]` to `[3]`. The text
   observation looks fresh, so the bug is easy to miss. SPaRC's eval builds a new env for every puzzle, so the
   paper's numbers are not affected. Task D copies an untouched env for every episode. The env code was not
   changed.
6. **Prompt text differs between settings.** `gym_step.py` (no backtracking) and `single_shot.py` say polyshapes
   may NOT be rotated. `gym_step_traceback.py` (backtracking) says rotation "is generally allowed". Task D keeps
   the paper's prompts unchanged.
7. **Paper text vs tables.** §4.7 (p. 9) says Gym accuracy rises "from 2.6% at 0.6B". Table 3 and Figures 2 and 8
   give 2.7% for Qwen 3 0.6B; 2.6% is Magistral Small's score. The "73%" image-input drop appears only in the
   abstract. It is derived from 10.6% (Qwen3 32B, text) to 2.8% (Qwen3-VL 32B, vision).
8. **Two step-by-step results in the paper come from mixed files.** These files are in
   `~/Documents/Cursor/sparc-gym-analysis/results/spatial_gym/`. On 2026-01-30, backtracking jobs were started
   without `--run-name traceback`, so they resumed the plain `_gym.jsonl` files:
   - `allenai_Olmo-3.1-32B-Think_gym.jsonl`: the last 40 of 500 rows were run with backtracking. The paper
     reports 11.4% (Table 3). The first 460 rows give 12.2%, and the log of the original full run gives 61/500 =
     12.2%.
   - `nvidia_Llama-3_3-Nemotron-Super-49B-v1_5_gym.jsonl`: the last 80 rows were run with backtracking. The
     paper reports 11.0%. The first 420 rows give 11.4%, and the original log gives 58/500 = 11.6%.
   - `rebuttal/allenai_Olmo-3.1-32B-Instruct_gym.jsonl` shows the same jump at row 400 (not in the paper
     tables).

   The reported numbers were not changed anywhere, including on the leaderboard.
9. **Qwen 3 0.6B with backtracking.** The result file has 257 rows. The paper (Table 4) reports 3.8% of 500,
   which counts the 243 missing puzzles as failures. On the 257 rows that exist, the rate is 7.4%.
10. **Versioning.** `pyproject.toml` says `version = "1.0.0"`, but releases and PyPI are at `v0.1.1`. The publish
    workflow overwrites the version from the git tag, so the value in the file is ignored.

## Third-party guide differences

- **verifiers v1 API.** verifiers 0.4.0 (released 2026-10-12) removed `load_environment`, `MultiTurnEnv`,
  `SingleTurnEnv`, `Rubric` and the old `vf-eval -a/-b/-k` flags. Its docs say not to add a `load_environment()`.
  The environment therefore exports a `Taskset` (`SpatialGymTaskset`) and an `Env` (`SpatialGymEnv`). The
  handoff's `load_environment(backtracking=...)` check maps to `SpatialGymConfig(backtracking=...)`, and both
  values were checked.
- **Name.** `vf-init spatial-gym` would create a distribution called `spatial-gym`. That normalises to the same
  name as `Spatial-Gym`, the package it depends on, so both cannot be installed together. The environment is
  `spatial-gym-env` (`vf-eval spatial-gym-env`). It still lives in `environments/spatial_gym/` as the handoff
  asks.
- **Chat history.** The paper's harness sent a fresh request at each step (system prompt plus current state).
  verifiers keeps one conversation per episode, so the model sees its earlier turns. The README says that scores
  are therefore not directly comparable to the paper.
- **Runtime.** The environment defaults to the `null` harness (plain chat) and the `subprocess` runtime, so
  `vf-eval` runs locally without a Prime sandbox.
- **Model access.** The handoff says "an API model". Only an OpenRouter key was available, so the runs use
  `--client.base-url https://openrouter.ai/api/v1 --client.api-key-var OPENROUTER_API_KEY`.
