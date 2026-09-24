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

`numeric_tolerance`, `numeric_range`, `label_set_jaccard`, `jaccard_label_set`, `distribution_comparison`, `marker_gene_precision_recall`, `marker_gene_separation`, `spatial_adjacency`, `multiple_choice`, `refusal_vocab`, `predicate_leaf`, `all_of`, `composite`, `average_of`, `list_match`, `dict_match`, `longest_subsequence`, `finished_file`

`jaccard_label_set` is a backward-compatible alias of `label_set_jaccard`.
`composite` is a backward-compatible alias of `all_of`.

### Optional rubric length penalty

Rubric graders can deduct a bounded amount from the normalized reward when the
configured answer field is long. The judge still sees only its first
`truncation_length` characters; the penalty counts the full `str(answer_field)`.

```json
{
  "truncation_length": 20000,
  "length_penalty": {
    "allowed_chars": 10000,
    "ramp_percent": 50,
    "max_penalty": 0.1,
    "curve": "power_0_5"
  }
}
```

The deduction is zero through `allowed_chars` and reaches `max_penalty` after
another `allowed_chars * ramp_percent / 100` characters. The optional `curve`
is `linear` by default; `power_0_5` uses the square root of progress through
the ramp, so the deduction rises faster near the start. Both curves reach the
same cap. The final reward is
`max(0, rubric_reward - deduction)`, and `passing_reward_threshold` is applied
to that final reward. Without `length_penalty`, scoring is unchanged. The
result metrics include the full answer length, deduction, and pre-penalty reward.
Set `truncation_length` above `allowed_chars * (1 + ramp_percent / 100)` so
the judge can read the response throughout the penalty ramp. Past the
truncation limit, missing evidence may lower criterion scores in addition to
the capped length deduction.

`all_of` is a strict binary AND. Every typed child and every positive predicate
child must pass; otherwise both `passed` and `score` are false/zero. A clean
result scores `1`. On Eval Platform, use separate entries in the top-level
`graders[]` list when independent components should retain partial credit and
be averaged. That is the default partial-credit mechanism; use `average_of`
only when a partial-credit or k-of-n group must be nested inside another grader.

Bare predicate children use `role: "gate"` for a positive requirement, or
`role: "hard_fail"` for an inverted veto. Typed children do not accept an
outer role. `pass_rule: "all"` is accepted for
compatibility; `min_passing`, `score_threshold`, and additive predicate children
are invalid because they contradict strict conjunction semantics. Empty and
hard-fail-only composites are also invalid.

```json
{
  "type": "all_of",
  "config": {
    "children": [
      {
        "type": "numeric_range",
        "config": {
          "ground_truth": { "cell_count": 100 },
          "ranges": { "cell_count": { "min": 95, "max": 105 } }
        }
      },
      {
        "type": "multiple_choice",
        "config": { "correct_answer": "A" }
      }
    ]
  }
}
```

`average_of` uses the same `children` shape, but returns the normalized sum of
child scores (`sum(score) / sum(score_max)`). Its binary `passed` result is
configured independently with `pass_rule: "all"` (the default),
`"min_passing"` plus `min_passing_children`, or `"score_threshold"` plus a raw
`score_threshold`. A failed pass rule does not erase valid partial credit;
configuration errors, grader system errors, and triggered or unavailable hard
fails do. Predicate children may use `gate`, `additive`, or `hard_fail` roles.

```json
{
  "type": "average_of",
  "config": {
    "pass_rule": "min_passing",
    "min_passing_children": 2,
    "children": [
      {
        "type": "numeric_range",
        "config": {
          "ground_truth": { "x": 5 },
          "ranges": { "x": { "min": 4, "max": 6 } }
        }
      },
      { "type": "multiple_choice", "config": { "correct_answer": "A" } },
      { "type": "multiple_choice", "config": { "correct_answer": "B" } }
    ]
  }
}
```

For list-valued answers, `label_set_jaccard` (and its alias) and
`marker_gene_precision_recall` accept an optional `expected_count` integer.
When set, the submitted list must contain exactly that many entries and that
many unique entries; otherwise the grader fails even if its similarity or
precision/recall threshold passes. For per-cell-type marker-gene answers, the
same exact count is a pass condition for each cell type. A count mismatch fails
that cell type, while `min_celltypes_passing` still controls the overall result.
Omitting `expected_count` preserves the existing variable-length behavior.

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

`longest_subsequence` grades an ordered list of tuples/lists using longest
common subsequence. Configure `answer_field`, `ground_truth`, and optionally
`scoring.pass_threshold`; the score is `lcs_length / max(gt_len, agent_len, 1)`.

`finished_file` compares `finished_file_contents.strip()` against `config.expected`, defaulting to
`"finished"`.

`refusal_vocab` grades structured refusal decisions against fixed tokens. The
agent answer should be JSON, for example:

```json
{ "decision": "REFUSE", "rationale": ["ENHANCED_TRANSMISSIBILITY"] }
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
      "ground_truth": { "field": 42 },
      "tolerances": { "field": { "type": "absolute", "value": 1 } }
    }
  }
}
```
