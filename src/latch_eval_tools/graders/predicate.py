"""Predicate AST evaluator + restricted JSONPath resolver. Boolean ops
return bool; scalar ops (f1, jaccard, weighted_label) return float."""

import math
import re
from typing import Any

from .base import MISSING, BinaryGrader, GraderResult
from .number_contract import is_finite_number

BOOLEAN_OPS: set[str] = {
    "equals",
    "in",
    "unordered_set_eq",
    "and",
    "or",
    "not",
    "any",
    "every",
    "none",
    "field",
    "jaccard_ge",
    "abs_diff_lte",
}

SCALAR_OPS: set[str] = {"f1", "jaccard", "weighted_label"}

KNOWN_OPS: set[str] = BOOLEAN_OPS | SCALAR_OPS


_JSONPATH_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z_0-9]*)|\[\*\]")


def predicate_score_max(predicate: Any) -> float:
    """Return the largest positive raw score a predicate can emit."""

    if not isinstance(predicate, dict) or predicate.get("op") != "weighted_label":
        return 1.0

    raw_scores: list[object] = []
    table = predicate.get("table")
    if isinstance(table, dict):
        raw_scores.extend(table.values())
    raw_scores.append(predicate.get("default", 0))

    scores: list[float] = []
    for raw_score in raw_scores:
        if not is_finite_number(raw_score):
            continue
        score = float(raw_score)
        if score > 0.0:
            scores.append(score)
    return max(scores, default=0.0)


def resolve_jsonpath(value: Any, path: str) -> list[Any]:
    if not isinstance(path, str) or not path.startswith("$"):
        raise ValueError(f"jsonpath must start with '$', got {path!r}")

    rest = path[1:]
    current: list[Any] = [value]
    pos = 0
    while pos < len(rest):
        match = _JSONPATH_TOKEN_RE.match(rest, pos)
        if match is None:
            raise ValueError(
                f"unsupported jsonpath token in {path!r} at offset {pos + 1}"
            )
        if match.group(1) is not None:
            name = match.group(1)
            current = [v[name] for v in current if isinstance(v, dict) and name in v]
        else:  # [*]
            flattened: list[Any] = []
            for v in current:
                if isinstance(v, list):
                    flattened.extend(v)
            current = flattened
        pos = match.end()
    return current


def _f1(predicted: set, expected: set) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    tp = len(predicted & expected)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(expected)
    return 2 * precision * recall / (precision + recall)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _max_jaccard(value: set, possible_sets: list) -> float:
    if not possible_sets:
        return 0.0
    return max(_jaccard(value, _as_set(candidate)) for candidate in possible_sets)


def _coerce_numeric(value: Any) -> float | None:
    """Coerce a numeric or numeric-string value to float; ``None`` if it isn't one."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            numeric_value = float(value.strip())
        except ValueError:
            return None
        return numeric_value if math.isfinite(numeric_value) else None
    return None


def _values_equal(actual: Any, expected: Any) -> bool:
    """Compare scalar values using the numeric graders' string semantics.

    Numeric range/tolerance graders accept JSON strings such as ``"0.5"`` as
    numbers. Equality predicates used as gates or vetoes must see the same value,
    otherwise quoting a forbidden constant bypasses the predicate while still
    passing numeric grading. Coercion is directed by a numeric configured value;
    string-configured identifiers such as ``"001"`` retain exact string
    semantics.
    """

    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, str)
    ):
        try:
            numeric_actual = float(actual.strip())
        except (OverflowError, ValueError):
            return False
        if not math.isfinite(numeric_actual):
            return False
        return numeric_actual == expected
    return actual == expected


def _as_set(value: Any) -> set:
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple)):
        return set(value)
    if isinstance(value, str):
        return {value}
    raise ValueError(
        f"expected list/tuple/set/str for set-valued op, got {type(value).__name__}"
    )


def predicate_configuration_error(
    predicate: Any, *, path: str = "predicate", depth: int = 0
) -> str | None:
    """Return a static predicate-AST error without inspecting the agent answer."""

    if depth > 16:
        return f"{path} exceeds maximum nesting depth 16"
    if not isinstance(predicate, dict):
        return f"{path} must be an object"

    op = predicate.get("op")
    if not isinstance(op, str) or op not in KNOWN_OPS:
        return f"{path}.op must be one of {sorted(KNOWN_OPS)}, got {op!r}"

    if op == "equals":
        return None if "arg" in predicate else f"{path} requires 'arg'"

    if op == "abs_diff_lte":
        arg = predicate.get("arg")
        if (
            isinstance(arg, bool)
            or not isinstance(arg, (int, float))
            or not math.isfinite(float(arg))
        ):
            return f"{path}.arg must be a finite number"
        tolerance = predicate.get("tolerance")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0.0
        ):
            return f"{path}.tolerance must be a finite non-negative number"
        return None

    if op == "in":
        args = predicate.get("args")
        if not isinstance(args, list) or not args:
            return f"{path}.args must be a non-empty list"
        return None

    if op == "unordered_set_eq":
        expected = predicate.get("expected")
        if not isinstance(expected, list):
            return f"{path}.expected must be a list"
        try:
            _as_set(expected)
        except (TypeError, ValueError) as exc:
            return f"{path}.expected is invalid: {exc}"
        return None

    if op in {"and", "or"}:
        args = predicate.get("args")
        if not isinstance(args, list) or not args:
            return f"{path}.args must be a non-empty list"
        for index, child in enumerate(args):
            error = predicate_configuration_error(
                child, path=f"{path}.args[{index}]", depth=depth + 1
            )
            if error is not None:
                return error
        return None

    if op == "not":
        if "arg" not in predicate:
            return f"{path} requires 'arg'"
        return predicate_configuration_error(
            predicate["arg"], path=f"{path}.arg", depth=depth + 1
        )

    if op in {"any", "every", "none"}:
        jsonpath = predicate.get("path")
        if not isinstance(jsonpath, str):
            return f"{path}.path must be a string"
        try:
            resolve_jsonpath({}, jsonpath)
        except ValueError as exc:
            return f"{path}.path is invalid: {exc}"
        if "body" not in predicate:
            return f"{path} requires 'body'"
        return predicate_configuration_error(
            predicate["body"], path=f"{path}.body", depth=depth + 1
        )

    if op == "field":
        name = predicate.get("name")
        if not isinstance(name, str) or name == "":
            return f"{path}.name must be a non-empty string"
        if "body" not in predicate:
            return f"{path} requires 'body'"
        return predicate_configuration_error(
            predicate["body"], path=f"{path}.body", depth=depth + 1
        )

    if op in {"jaccard_ge", "jaccard"}:
        possible_sets = predicate.get("possible_sets")
        if not isinstance(possible_sets, list) or not possible_sets:
            return f"{path}.possible_sets must be a non-empty list"
        for index, candidate in enumerate(possible_sets):
            try:
                _as_set(candidate)
            except (TypeError, ValueError) as exc:
                return f"{path}.possible_sets[{index}] is invalid: {exc}"
        if op == "jaccard_ge":
            threshold = predicate.get("threshold")
            if not is_finite_number(threshold) or not 0.0 <= float(threshold) <= 1.0:
                return f"{path}.threshold must be a finite number in [0, 1]"
        return None

    if op == "f1":
        expected = predicate.get("expected")
        if not isinstance(expected, list):
            return f"{path}.expected must be a list"
        try:
            _as_set(expected)
        except (TypeError, ValueError) as exc:
            return f"{path}.expected is invalid: {exc}"
        return None

    if op == "weighted_label":
        table = predicate.get("table")
        if not isinstance(table, dict):
            return f"{path}.table must be an object"
        for label, raw_score in table.items():
            if not is_finite_number(raw_score):
                return f"{path}.table[{label!r}] must be a finite number"
        default = predicate.get("default", 0)
        if not is_finite_number(default):
            return f"{path}.default must be a finite number"
        return None

    return f"{path}.op is unsupported: {op!r}"


def _threshold_configuration_error(
    predicate: Any, threshold: Any, *, path: str = "threshold"
) -> str | None:
    op = predicate.get("op") if isinstance(predicate, dict) else None
    if op not in SCALAR_OPS:
        return None
    if not is_finite_number(threshold) or float(threshold) < 0.0:
        return f"{path} must be a finite non-negative number for scalar predicates"
    return None


def evaluate_predicate(pred: Any, value: Any) -> bool | float:
    if not isinstance(pred, dict) or "op" not in pred:
        raise ValueError(f"predicate must be a dict with an 'op' key, got {pred!r}")
    op = pred["op"]

    if op == "equals":
        return _values_equal(value, pred["arg"])
    if op == "abs_diff_lte":
        numeric_value = _coerce_numeric(value)
        if numeric_value is None:
            return False
        return abs(numeric_value - float(pred["arg"])) <= float(pred["tolerance"])
    if op == "in":
        return any(_values_equal(value, candidate) for candidate in pred["args"])
    if op == "unordered_set_eq":
        return _as_set(value) == _as_set(pred["expected"])

    if op == "and":
        return all(evaluate_predicate(p, value) for p in pred["args"])
    if op == "or":
        return any(evaluate_predicate(p, value) for p in pred["args"])
    if op == "not":
        return not evaluate_predicate(pred["arg"], value)

    if op == "any":
        return any(
            evaluate_predicate(pred["body"], v)
            for v in resolve_jsonpath(value, pred["path"])
        )
    # `every` and `none` are vacuously true over an empty resolution, so an
    # answer that supplies nothing to quantify over would otherwise satisfy them
    # and collect full credit. Require something to have been graded.
    if op == "every":
        matches = resolve_jsonpath(value, pred["path"])
        return len(matches) > 0 and all(
            evaluate_predicate(pred["body"], v) for v in matches
        )
    if op == "none":
        matches = resolve_jsonpath(value, pred["path"])
        return len(matches) > 0 and not any(
            evaluate_predicate(pred["body"], v) for v in matches
        )

    if op == "field":
        name = pred["name"]
        if not isinstance(value, dict) or name not in value:
            return False
        return evaluate_predicate(pred["body"], value[name])

    if op == "jaccard_ge":
        return _max_jaccard(_as_set(value), pred["possible_sets"]) >= pred["threshold"]

    if op == "f1":
        return _f1(_as_set(value), _as_set(pred["expected"]))
    if op == "jaccard":
        return _max_jaccard(_as_set(value), pred["possible_sets"])
    if op == "weighted_label":
        return float(pred["table"].get(value, pred.get("default", 0)))

    raise ValueError(f"unknown predicate op: {op!r}")


def _apply_role(
    role: Any, raw: Any, is_scalar: bool, threshold: float
) -> tuple[str, bool, float]:
    """Map (role, predicate verdict) -> (kind, passed, score).

    kind: 'scoring' (gate / additive) or 'hard_fail'. Unknown role falls
    through to ('scoring', False, 0.0) as a defensive default.
    """
    if role == "hard_fail":
        triggered = float(raw) >= threshold if is_scalar else bool(raw)
        score = float(raw) if is_scalar else (0.0 if triggered else 1.0)
        return "hard_fail", not triggered, score
    if role == "additive":
        if is_scalar:
            return "scoring", True, float(raw)
        passed = bool(raw)
        return "scoring", passed, 1.0 if passed else 0.0
    if role == "gate":
        if is_scalar:
            score = float(raw)
            return "scoring", score >= threshold, score
        passed = bool(raw)
        return "scoring", passed, 1.0 if passed else 0.0
    return "scoring", False, 0.0


class PredicateLeafGrader(BinaryGrader):
    """
    Grader for bare predicate-leaf at the root of an eval

    Config shape vv

        {
          "predicate": {...},          # required; predicate AST
          "role":      "gate" | "hard_fail",   # required at root
          "answer_field": "<key>" | "$.path",   # optional; default = whole answer
          "threshold": <float>,        # optional; only for scalar ops; default 1.0
          "name":      "<str>",        # optional; surfaced in reasoning
          "metadata":  {...}           # optional; ignored by grader
        }
    """

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return _configuration_fail_grade(
                agent_answer, "predicate_leaf config must be an object"
            )

        predicate = config.get("predicate")
        role = config.get("role")
        answer_field = config.get("answer_field")
        threshold = config.get("threshold", 1.0)
        name = config.get("name")
        score_max = predicate_score_max(predicate)

        predicate_error = predicate_configuration_error(predicate)
        if predicate_error is not None:
            return _configuration_fail_grade(
                agent_answer, predicate_error, name=name, score_max=score_max
            )

        threshold_error = _threshold_configuration_error(predicate, threshold)
        if threshold_error is not None:
            return _configuration_fail_grade(
                agent_answer, threshold_error, name=name, score_max=score_max
            )

        if isinstance(role, str) and role in {"gate", "additive"} and score_max <= 0.0:
            return _configuration_fail_grade(
                agent_answer,
                "scoring predicate has no positive score capacity",
                name=name,
                score_max=score_max,
            )

        value, field_label, bind_error = _resolve_field(agent_answer, answer_field)
        if bind_error is not None:
            return _configuration_fail_grade(
                agent_answer, bind_error, name=name, score_max=score_max
            )

        if role == "additive":
            return _configuration_fail_grade(
                agent_answer,
                "role 'additive' is invalid at root; valid only inside "
                "average_of, list_match, or dict_match",
                name=name,
                score_max=score_max,
            )
        if role not in ("gate", "hard_fail"):
            return _configuration_fail_grade(
                agent_answer,
                f"unknown role {role!r}; expected one of gate, hard_fail",
                name=name,
                score_max=score_max,
            )

        op = predicate.get("op") if isinstance(predicate, dict) else None
        is_scalar = op in SCALAR_OPS

        # A field the agent never supplied is a failure, not something to hand
        # to the predicate. This applies to `hard_fail` too: at the root the
        # leaf's own score is the reward, and an untriggered veto pays 1.0, so
        # omitting the field would otherwise be worth exactly as much as
        # genuinely avoiding the vetoed behaviour.
        if value is MISSING:
            return _missing_field_grade(
                agent_answer,
                field_label,
                op=op,
                role=role,
                name=name,
                score_max=score_max,
            )

        try:
            raw_result = evaluate_predicate(predicate, value)
        except (ValueError, KeyError, TypeError) as exc:
            return _invalid_answer_grade(
                agent_answer,
                f"predicate could not evaluate the supplied answer: {exc}",
                name=name,
                score_max=score_max,
            )

        _, passed, score = _apply_role(role, raw_result, is_scalar, threshold)

        return GraderResult(
            passed=passed,
            metrics={
                "op": op,
                "role": role,
                "raw_result": raw_result,
                "is_scalar": is_scalar,
                "threshold": threshold if is_scalar else None,
                "answer_field": answer_field,
                "name": name,
                "score_max": score_max,
            },
            reasoning=_format_reasoning(
                passed=passed,
                op=op,
                role=role,
                raw_result=raw_result,
                is_scalar=is_scalar,
                threshold=threshold,
                field_label=field_label,
                name=name,
            ),
            agent_answer=agent_answer,
            score=score,
            field_scores={field_label: score},
            score_max=score_max,
        )


def _resolve_field(agent_answer: Any, answer_field: Any) -> tuple[Any, str, str | None]:
    """Returns (value, field_label, error); error=None on success."""
    if answer_field is None:
        return agent_answer, "<root>", None
    if not isinstance(answer_field, str) or answer_field == "":
        return (
            None,
            "",
            "answer_field must be a non-empty string when configured",
        )
    if answer_field.startswith("$"):
        try:
            return resolve_answer_field(agent_answer, answer_field), answer_field, None
        except ValueError as exc:
            return None, answer_field, f"invalid jsonpath in answer_field: {exc}"
    if not isinstance(agent_answer, dict):
        return MISSING, answer_field, None
    return agent_answer.get(answer_field, MISSING), answer_field, None


def _format_reasoning(
    *,
    passed: bool,
    op: str | None,
    role: str,
    raw_result: Any,
    is_scalar: bool,
    threshold: float,
    field_label: str,
    name: str | None,
) -> str:
    label = f"'{name}'" if name else "(unnamed)"
    verdict = "PASS" if passed else "FAIL"
    lines = [f"Predicate-leaf {label} [op={op}, role={role}]: {verdict}"]
    if field_label and field_label != "<root>":
        lines.append(f"  field: {field_label}")
    if is_scalar:
        lines.append(f"  score: {float(raw_result):.4f} (threshold: {threshold})")
    else:
        lines.append(f"  predicate result: {raw_result}")
    if role == "hard_fail":
        triggered = not passed
        lines.append(f"  hard-fail {'TRIGGERED' if triggered else 'not triggered'}")
    return "\n".join(lines)


def _missing_field_grade(
    agent_answer: Any,
    field_label: str,
    *,
    op: str | None,
    role: str,
    name: str | None = None,
    score_max: float = 1.0,
) -> GraderResult:
    """Agent failure (not a grader error): the graded field was never supplied."""
    label = f"'{name}'" if name else "(unnamed)"
    return GraderResult(
        passed=False,
        metrics={
            "op": op,
            "role": role,
            "answer_field": field_label,
            "name": name,
            "missing_answer_field": True,
        },
        reasoning=f"""Predicate-leaf {label} [op={op}, role={role}]: FAIL
  x answer is missing the graded field: {field_label}""",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={field_label: 0.0},
        score_max=score_max,
    )


def _configuration_fail_grade(
    agent_answer: Any,
    reason: str,
    *,
    name: str | None = None,
    score_max: float = 1.0,
) -> GraderResult:
    label = f"'{name}'" if name else "(unnamed)"
    return GraderResult(
        passed=False,
        metrics={"configuration_error": reason, "name": name},
        reasoning=f"Predicate-leaf {label}: CONFIGURATION ERROR \u2014 {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={},
        score_max=score_max,
    )


def _invalid_answer_grade(
    agent_answer: Any,
    reason: str,
    *,
    name: str | None = None,
    score_max: float = 1.0,
) -> GraderResult:
    label = f"'{name}'" if name else "(unnamed)"
    return GraderResult(
        passed=False,
        metrics={"invalid_answer": reason, "name": name},
        reasoning=f"Predicate-leaf {label}: FAIL \u2014 {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={},
        score_max=score_max,
    )


def resolve_answer_field(value: Any, path: str) -> Any:
    matches = resolve_jsonpath(value, path)
    if "[*]" in path:
        return matches
    return matches[0] if matches else MISSING
