# NOTES — citation & benchmark-listing work (Spatial-Gym repo)

Work log for the SPaRC / Spatial-Gym citation-infrastructure handoff, written 2026-10-13. A matching
`NOTES.md` exists in the SPaRC repo.

## Branches

The branches are stacked so `NOTES.md` accumulates:

| Branch | Base | Content |
|---|---|---|
| `task-a-citations` | `main` | Task A: README citation (also the PyPI long description), related-work block |
| `task-d-verifiers` | `task-a-citations` | Task D: `environments/spatial_gym/` verifiers environment |

## Code map (handoff §3)

### SPaRC (`lkaesberg/SPaRC`)
- **README**: `README.md`
- **Puzzle data loader**: there is no dedicated loader module. The data is loaded inline with
  `load_dataset("lkaesberg/SPaRC", "all", split="test")` in `sparc/sparc.py:745` (`main()`).
  Each row has `id`, `difficulty_level` (1–5), `difficulty_score`, `grid_size`, `polyshapes` (a JSON string),
  `puzzle_array`, `solution_count`, `solutions` (each with a `path` list of `{x, y}`), and `text_visualization`.
- **Path validator**: `sparc/validation.py`
  - `extract_solution_path` (`:5`) takes the text after the last `####`, regex-matches `(x,y)` pairs,
    and stops at the end node.
  - `validate_solution` (`:64`) is **the validator**. It does an exact match of the extracted path against
    any entry of `puzzle["solutions"]`. All 500 test rows have `len(solutions) == solution_count` (1–48 solutions).
  - `analyze_path` (`:98`) does structural checks: start/end, connected, non-intersecting, no rule-cell crossing.
    `fully_valid_path == False` is the paper's "Invalid Path" (Table 7).
- **Prompts**: the paper's single-shot prompt is in `sparc/prompts/single_shot.py:22` (`get_prompt`). The
  registry is `sparc/prompts/__init__.py`, and backward-compatible wrappers are in `sparc/prompt.py`.
- **Existing LLM evaluation script**: the `sparc` CLI in `sparc/sparc.py` (`main()`). Per-puzzle logic is in
  `sparc/process_puzzle.py`:
  - `process_puzzle` (`:66`) is single-shot.
  - `process_puzzle_step_by_step` (`:207`) is gym mode: `observation='SPaRC'`, 100 steps, and no chat history
    (each step sends system + current state only).
  - Tables are in `sparc/tables.py`.
  - The CLI default is `--temperature 1.0`.
- `sparc/process_puzzle.py:10` imports `SPaRC_Gym` (PyPI `sparc-gym<0.3`), which is the pre-rename name of
  Spatial-Gym, not `Spatial_Gym`.

### Spatial-Gym (`lkaesberg/Spatial-Gym`)
- **README**: `README.md`. `pyproject.toml` sets `readme = "README.md"`, so the PyPI long description is the
  README. There is no `setup.py`.
- **Puzzle data loader**: `Spatial_Gym/Spatial_Gym.py`
  - `load_dataset(df_name, df_split, split=df_set)` is at `:77` in `__init__`.
  - `_process_puzzles` (`:219`) and `_load_puzzle` (`:95`) build the per-puzzle arrays.
- **Path validator**:
  - The success check is in `Spatial_Gym.step` (`:1111`). On termination or truncation it sets
    `outcome_reward = 1` (`:1207`) only if `self.path` exactly equals one of `solution_paths`. This is the
    same semantics as SPaRC's `validate_solution`.
  - `_is_on_solution_path` (`:1244`) gives the +0.01 shaping reward.
  - `_run_rule_validators` / `_validate_rules` (`:901`, `:941`) fill `info["rule_status"]`. This is
    informational only and not used for reward (see Discrepancies).
- **Renderers**: `Spatial_Gym/render/llm_renderer.py` (pygame window, see Discrepancies) and
  `Spatial_Gym/render/human_renderer.py`.
- **Existing LLM evaluation script**: `llm_testing/llm_host.py`. `run_episode` (`:62`) uses
  `observation='SPaRC'`, `max_steps=100` and `temperature=0.0`. The paper's runs appear to have used SPaRC's
  `sparc --gym` path above.
- **Tests**: `tests/` (43 tests, all pass on `main` as of 2026-10-13).


## Discrepancies found

1. **The old citation block was already gone.** The handoff says the README's "If you use Spatial-Gym, please
   cite" block references a different SPaRC title. That block was already removed on `main` in commit `32034a0`
   (2026-03-31). However, **the PyPI page for 0.1.1 still shows it**, including the wrong title "SPaRC: Spatial
   Reasoning Challenges for Large Language Models". Task A adds the correct two-entry block; a release is needed
   (see below).
2. **SPaRC page numbers.** The Spatial-Gym paper's bibliography cites SPaRC as pp. 10370–10401. The ACL
   Anthology value is 10359–10390. Fix it in the next arXiv version (human task).
3. **The Spatial-Gym rule checker disagrees with the stored solutions.** All 500 test puzzles' ground-truth
   solution 0 gets `outcome_reward == 1` (exact match, `Spatial_Gym.py:1207`). Only 242/500 get
   `info["rule_status"]["all_rules_satisfied"]["passed"] == True`: `poly_ylop_area` fails on 251 and
   `star_pairing_exact` on 24. The validator semantics used by the paper and by SPaRC's `validate_solution` is
   exact match, so Task D uses that path (outcome reward). The rule checker is left unchanged, per the handoff
   rules; it looks buggy for polyshape/ylop and star regions.
4. **`render_mode='llm'` is not a text rendering.** `Spatial_Gym/render/llm_renderer.py` opens a pygame window
   and draws the grid, and `render()` returns `None`. The text the paper's step-by-step agents see is the
   `observation='SPaRC'` grid, printed one row per line as Python lists (`str(row)`), plus position and legal
   actions. That is what SPaRC's `process_puzzle_step_by_step` sends. Task D uses that, not `render_mode='llm'`.
5. **Paper text vs tables.** §4.7 (p. 9) says Gym accuracy rises "from 2.6% at 0.6B". Table 3 and Figs. 2 and 8
   give 2.7% for Qwen 3 0.6B (2.6% is Magistral Small). The "73%" image-input drop appears only in the abstract
   and is derived from 10.6% → 2.8% (Qwen3 text vs Qwen3-VL vision).
6. **Versioning.** `pyproject.toml` says `version = "1.0.0"`, but releases and PyPI are at `v0.1.1`. The
   publish workflow overwrites the version from the git tag, so the file value is ignored.

## Reproduced numbers

_(filled in by Task D)_

## Commands left for a human

### Task A
- **PyPI release (A5).** The PyPI long description is `README.md`, so the PyPI page only picks up the corrected
  citation after a new release. After `task-a-citations` is merged into `main`:
  ```bash
  gh release create v0.1.2 -R lkaesberg/Spatial-Gym --target main --title v0.1.2 --notes "Add citation (Spatial-Gym arXiv 2604.09338 + SPaRC EMNLP 2025) and related-work section"
  ```
  The version was intentionally not bumped in this task.

## Third-party guide differences

_(filled in by Task D)_
