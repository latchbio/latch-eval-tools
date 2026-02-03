# latch-eval-tools

Shared eval tools for single-cell bench, spatial bench, and future biology benchmarks.

## Installation

```bash
pip install latch-eval-tools
```

## Components

### Types

```python
from latch_eval_tools import Eval, EvalResult

eval_case = Eval(
    id="test_001",
    task="Count cells in the dataset",
    data_node="latch:///data/sample.h5ad",
    grader={"type": "numeric_tolerance", "config": {...}}
)
```

### Graders

Available graders: `numeric_tolerance`, `label_set_jaccard`, `distribution_comparison`, `marker_gene_precision_recall`, `marker_gene_separation`, `spatial_adjacency`, `multiple_choice`

```python
from latch_eval_tools.graders import get_grader, NumericToleranceGrader

grader = get_grader("numeric_tolerance")
result = grader.evaluate(
    agent_answer={"n_cells": 1523},
    config={
        "ground_truth": {"n_cells": 1500},
        "tolerances": {"n_cells": {"type": "relative", "value": 0.05}}
    }
)
print(result.passed)
print(result.reasoning)
```

### Harness

Run evaluations with different agents:

```python
from latch_eval_tools.harness import EvalRunner, run_minisweagent_task

runner = EvalRunner("evals/count_cells.json", cache_name=".scbench")
result = runner.run(agent_function=lambda task, work_dir: 
    run_minisweagent_task(task, work_dir, model_name="anthropic/claude-sonnet-4")
)

def my_agent(task_prompt: str, work_dir: Path) -> dict:
    return {"answer": json.loads((work_dir / "eval_answer.json").read_text())}

runner.run(agent_function=my_agent)
```

Built-in agents: `run_minisweagent_task`, `run_claudecode_task`, `run_plotsagent_task`

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

## Eval JSON Schema

```json
{
  "id": "unique_test_id",
  "task": "Task description for the agent",
  "data_node": "latch:///path/to/data.h5ad",
  "grader": {
    "type": "numeric_tolerance",
    "config": {
      "ground_truth": {"field": 42},
      "tolerances": {"field": {"type": "absolute", "value": 1}}
    }
  }
}
```
