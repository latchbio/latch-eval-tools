# latch-eval-tools

Shared eval tools for single-cell bench, spatial bench, and future biology benchmarks.

## Installation

### Get the `latch-eval` CLI on your PATH (recommended)

Install it as a **standalone tool**. This is isolated and puts `latch-eval` on
your PATH from any directory — it does **not** depend on which Python / conda env
(or whether any) is active, which avoids the "command lives in an env that isn't
active" and "wrong env shadows the command" problems:

```bash
# from PyPI (once a release including the CLI is published):
uv tool install latch-eval-tools          # or: pipx install latch-eval-tools

# from a local checkout (current dev / pre-release) — --editable keeps edits live:
uv tool install --editable .

# or straight from the branch:
uv tool install "git+https://github.com/latchbio/latch-eval-tools.git@add-eval-run-cli"
```

If the tool shim directory isn't on PATH yet, wire it up once (then restart your shell):

```bash
uv tool update-shell                      # uv (adds ~/.local/bin)
# pipx equivalent: pipx ensurepath
```

Verify: `latch-eval --help` and `which latch-eval`.

### Use as a Python library

```bash
pip install latch-eval-tools              # into a venv / conda env
```

Note: a plain `pip install` registers the `latch-eval` script inside *that*
environment's `bin/`, so the command is only on PATH while that env is **active**
(`conda activate <env>` / `source .venv/bin/activate`), and after installing into
a freshly-active env you may need `hash -r` (zsh: `rehash`) or a new shell. For a
reliably-available command, prefer the standalone `uv tool` / `pipx` install above.

### Run without putting anything on PATH

```bash
uvx --from latch-eval-tools latch-eval run ...   # one-off, no install
python -m latch_eval_tools.cli run ...           # if the package is importable
```

## What is included

- `Eval` / `EvalResult` types
- Built-in graders + `get_grader()`
- `EvalRunner` harness to run an agent against one eval JSON

## Quickstart

```python
from latch_eval_tools import EvalRunner, run_minisweagent_task

runner = EvalRunner("evals/count_cells.json")
result = runner.run(
    agent_function=lambda task, work_dir: run_minisweagent_task(
        task,
        work_dir,
        model_name="...your model name...",
    )
)

print(result["passed"])
print(result["grader_result"].reasoning if result["grader_result"] else "No grader result")
```

`EvalRunner.run()` expects an `agent_function(task_prompt, work_dir)` and supports either:
- returning a plain answer `dict`, or
- returning `{"answer": <dict>, "metadata": <dict>}`

If your agent writes `eval_answer.json` in `work_dir`, the runner will load it automatically.

## CLI

Installing the package adds a `latch-eval` command that wraps `EvalRunner` so you
can run a single eval JSON end-to-end (download data → run an agent in a Docker
sandbox → grade) without writing any Python:

```bash
latch-eval run -e evals/count_cells.json --harness claudecode
latch-eval run -e evals/de03.json --harness minisweagent --model anthropic/claude-sonnet-4-6
```

`--harness` is required: one of `claudecode`, `minisweagent`, `openaicodex`,
`pi`. `minisweagent` also requires `--model`. Useful flags:

- `--json-out PATH` — write a structured result (`passed`, `grader_result`, `agent_answer`, `metadata`)
- `--keep-workspace` / `--no-keep-workspace` — keep run artifacts (default: keep)
- `--output-dir DIR` — where run workspaces go, as `<output-dir>/[run-id/]<eval_id>/` (default: `./.latch-eval-runs`, relative to the current directory)
- `--cache-dir DIR` — shared downloaded-data cache, reused across runs/projects (default: `~/.cache/latch-eval`)
- `--data PATH` — stage a local file/dir as `/workspace/data`, bypassing the eval's `data_node` download (repeatable)
- `--eval-timeout SECONDS`, `--docker-image IMAGE`, `--run-id ID`
- `--no-preflight` — skip the Docker / API-key / Latch-token checks

Requires Docker, a provider API key for the chosen harness (e.g.
`ANTHROPIC_API_KEY`), and `~/.latch/token` (via `latch login`) for `data_node`
downloads. The command exits non-zero on FAIL / NO GRADE. Note: only a single
`grader` is graded; evals using a multi-grader `graders` list run but produce no
local grade. Equivalent: `python -m latch_eval_tools.cli run ...`.

Run workspaces are written under `./.latch-eval-runs/` relative to where you
invoke `latch-eval` (overridable with `--output-dir`), so they never land inside
the installed package. `latch://` data nodes are downloaded once into the shared
cache (`~/.cache/latch-eval`) and hardlinked into each run's workspace, so
re-running an eval — from any directory — does not re-download. Use `--data` to
point at data already on disk and skip Latch entirely; local paths and `file://`
URIs also work directly as `data_node` values.

(Library callers of `EvalRunner` are unaffected: without `work_root` / `cache_dir`,
workspaces and cache still resolve under the project root as before.)

## Graders

Available grader types:

`numeric_tolerance`, `jaccard_label_set`, `distribution_comparison`, `marker_gene_precision_recall`, `marker_gene_separation`, `spatial_adjacency`, `multiple_choice`, `refusal_vocab`

```python
from latch_eval_tools.graders import get_grader

grader = get_grader("numeric_tolerance")
result = grader.evaluate_answer(
    agent_answer={"n_cells": 1523},
    config={
        "ground_truth": {"n_cells": 1500},
        "tolerances": {"n_cells": {"type": "relative", "value": 0.05}},
    },
)
print(result.passed, result.reasoning)
```

`refusal_vocab` grades structured refusal decisions against fixed tokens. The
agent answer should be JSON, for example:

```json
{"decision": "REFUSE", "rationale": ["ENHANCED_TRANSMISSIBILITY"]}
```

See `examples/refusal_vocab_example.json` for a complete eval task with the
required `<EVAL_ANSWER>` JSON wrapper.

Built-in harness helpers:

- `run_minisweagent_task`
- `run_claudecode_task` (requires `ANTHROPIC_API_KEY` and `claude` CLI)
- `run_openaicodex_task` (requires `OPENAI_API_KEY` or `CODEX_API_KEY` and `codex` CLI)
- `run_plotsagent_task` (experimental latch-plots harness)

## Eval JSON shape

```json
{
  "id": "unique_test_id",
  "task": "Task description. Include an <EVAL_ANSWER> JSON template in this text.",
  "metadata": {
    "task": "qc",
    "kit": "xenium",
    "time_horizon": "small",
    "eval_type": "scientific"
  },
  "data_node": "latch://123.node/path/to/data.h5ad",
  "grader": {
    "type": "numeric_tolerance",
    "config": {
      "ground_truth": {"field": 42},
      "tolerances": {"field": {"type": "absolute", "value": 1}}
    }
  }
}
```
