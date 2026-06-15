"""Predicate AST evaluator + restricted JSONPath resolver. Boolean ops
return bool; scalar ops (f1, jaccard, weighted_label) return float."""

import re
from typing import Any

from .base import BinaryGrader, GraderResult

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
}

SCALAR_OPS: set[str] = {"f1", "jaccard", "weighted_label"}

KNOWN_OPS: set[str] = BOOLEAN_OPS | SCALAR_OPS


_JSONPATH_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z_0-9]*)|\[\*\]")


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


def evaluate_predicate(pred: Any, value: Any) -> bool | float:
    if not isinstance(pred, dict) or "op" not in pred:
        raise ValueError(f"predicate must be a dict with an 'op' key, got {pred!r}")
    op = pred["op"]

    if op == "equals":
        return value == pred["arg"]
    if op == "in":
        return value in pred["args"]
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
    if op == "every":
        return all(
            evaluate_predicate(pred["body"], v)
            for v in resolve_jsonpath(value, pred["path"])
        )
    if op == "none":
        return not any(
            evaluate_predicate(pred["body"], v)
            for v in resolve_jsonpath(value, pred["path"])
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
        predicate = config.get("predicate")
        role = config.get("role")
        answer_field = config.get("answer_field")
        threshold = config.get("threshold", 1.0)
        name = config.get("name")

        value, field_label, bind_error = _resolve_field(agent_answer, answer_field)
        if bind_error is not None:
            return _fail_grade(agent_answer, bind_error, name=name)

        try:
            raw_result = evaluate_predicate(predicate, value)
        except (ValueError, KeyError, TypeError) as exc:
            return _fail_grade(
                agent_answer,
                f"predicate evaluation failed: {exc}",
                name=name,
            )

        op = predicate.get("op") if isinstance(predicate, dict) else None
        is_scalar = op in SCALAR_OPS

        if role == "additive":
            return _fail_grade(
                agent_answer,
                "role 'additive' is invalid at root; valid only inside "
                "list_match or dict_match GT entries",
                name=name,
            )
        if role not in ("gate", "hard_fail"):
            return _fail_grade(
                agent_answer,
                f"unknown role {role!r}; expected one of gate, hard_fail",
                name=name,
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
        )


def _resolve_field(agent_answer: Any, answer_field: Any) -> tuple[Any, str, str | None]:
    """Returns (value, field_label, error); error=None on success."""
    if answer_field is None:
        return agent_answer, "<root>", None
    if not isinstance(answer_field, str):
        return (
            None,
            "",
            f"answer_field must be string, got {type(answer_field).__name__}",
        )
    if answer_field.startswith("$"):
        try:
            return resolve_answer_field(agent_answer, answer_field), answer_field, None
        except ValueError as exc:
            return None, answer_field, f"invalid jsonpath in answer_field: {exc}"
    if not isinstance(agent_answer, dict):
        return None, answer_field, None
    return agent_answer.get(answer_field), answer_field, None


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


def _fail_grade(
    agent_answer: Any, reason: str, *, name: str | None = None
) -> GraderResult:
    label = f"'{name}'" if name else "(unnamed)"
    return GraderResult(
        passed=False,
        metrics={"error": reason, "name": name},
        reasoning=f"Predicate-leaf {label}: GRADER ERROR \u2014 {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={},
    )


def resolve_answer_field(value: Any, path: str) -> Any:
    matches = resolve_jsonpath(value, path)
    if "[*]" in path:
        return matches
    return matches[0] if matches else None
