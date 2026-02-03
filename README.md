# latch-eval-tools

Shared harness tools for singlecell bench, spatial bench, and future biology benchmarks.

## Installation

```bash
pip install latch-eval-tools
```

## Linting Eval Files

```bash
# Lint a single eval file
eval-lint path/to/eval.json

# Lint a directory
eval-lint evals/my_dataset/

# JSON output for CI/CD
eval-lint evals/ --format json
```

## Python API

```python
from latch_eval_tools import Eval, EvalResult, lint_eval

result = lint_eval("path/to/eval.json")
if result.passed:
    print("Validation passed!")
```

## License

Proprietary
