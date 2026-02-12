# latch-eval-tools

Shared eval tools for single-cell bench, spatial bench, and future biology benchmarks.

## Installation

```bash
pip install latch-eval-tools
```

## What is included

- `Eval` / `EvalResult` types
- Built-in graders + `get_grader()`
- `EvalRunner` harness to run an agent against one eval JSON
- `eval-lint` CLI and Python linter APIs

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

## Graders

Available grader types:

`numeric_tolerance`, `jaccard_label_set`, `distribution_comparison`, `marker_gene_precision_recall`, `marker_gene_separation`, `spatial_adjacency`, `multiple_choice`

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

Built-in harness helpers:

- `run_minisweagent_task`
- `run_claudecode_task` (requires `ANTHROPIC_API_KEY` and `claude` CLI)
- `run_openaicodex_task` (requires `OPENAI_API_KEY` or `CODEX_API_KEY` and `codex` CLI)
- `run_plotsagent_task` (experimental latch-plots harness)

### Linter

Validate eval JSON files:

```bash
eval-lint evals/my_dataset/
eval-lint evals/ --format json
```

```python
from latch_eval_tools.linter import lint_eval, lint_directory

result = lint_eval("evals/test.json")
print(result.passed, result.issues)
```

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
