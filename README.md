# latch-eval-tools

Shared evaluation harness tools for biology AI benchmarks.

## Installation

```bash
pip install latch-eval-tools
```

## Features

- **Core Types**: `Eval`, `EvalResult`, `TestCase`, `TestResult` - Pydantic models for evaluation data
- **Linter**: Validate eval JSON files with `eval-lint` CLI tool
- **Servers**: Evaluation servers for running benchmarks (eval_server, headless_eval_server)
- **Utilities**: Answer extraction, FAAS utilities, wrapper entrypoint

## Usage

### Linting Eval Files

```bash
# Lint a single eval file
eval-lint path/to/eval.json

# Lint a directory
eval-lint evals/my_dataset/

# JSON output for CI/CD
eval-lint evals/ --format json
```

### Python API

```python
from latch_eval_tools import Eval, EvalResult, lint_eval

# Load and validate an eval
result = lint_eval("path/to/eval.json")
if result.passed:
    print("Validation passed!")
```

## Optional Dependencies

Install additional features:

```bash
# Scientific computing (numpy, scipy, scikit-learn, matplotlib)
pip install latch-eval-tools[scientific]

# OpenAI/Anthropic support
pip install latch-eval-tools[openai]

# Latch platform integration
pip install latch-eval-tools[latch]
```

## License

Proprietary
