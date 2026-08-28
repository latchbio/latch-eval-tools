"""Composite graders: all_of, average_of, list_match, dict_match. Recursive over
predicate-leaves and nested composites."""

import math
from typing import Any

from .base import MISSING, BinaryGrader, GraderResult, normalize_score
from .number_contract import is_finite_number
from .predicate import (
    SCALAR_OPS,
    _apply_role,
    _threshold_configuration_error,
    evaluate_predicate,
    predicate_configuration_error,
    predicate_score_max,
    resolve_answer_field,
)

# shared helper functions vv


def _is_leaf(node: Any) -> bool:
    """A node is a leaf iff it is a bare ``{predicate, role, ...}`` envelope."""
    return isinstance(node, dict) and "predicate" in node and "type" not in node


def _all_of_child_kind(child: Any) -> tuple[str, str | None]:
    """Classify one strict-AND child and return any configuration error.

    Every positive child of ``all_of`` is required. Predicate leaves retain a
    role only to distinguish a positive condition from an inverted
    ``hard_fail`` veto. Partial-credit ``additive`` leaves and outer roles on
    typed children are deliberately unsupported: independent scoring belongs
    in the eval's top-level ``graders`` list.
    """

    if not isinstance(child, dict):
        return "invalid", "all_of child must be an object"

    if _is_leaf(child):
        role = child.get("role")
        if role == "gate":
            return "required", None
        if role == "hard_fail":
            return "hard_fail", None
        if role == "additive":
            return (
                "invalid",
                "role 'additive' is invalid inside strict all_of; use a "
                "top-level graders[] entry or average_of for partial credit",
            )
        return (
            "invalid",
            f"bare predicate child role must be one of gate/hard_fail, got {role!r}",
        )

    if "role" in child:
        return (
            "invalid",
            "typed all_of children do not accept an outer role; every typed "
            "child is required",
        )

    child_type = child.get("type")
    child_config = child.get("config")
    if child_type is not None and not isinstance(child_config, dict):
        return "invalid", "typed all_of child config must be an object"

    if child_type != "predicate_leaf":
        return "required", None

    role = child_config.get("role") if isinstance(child_config, dict) else None
    if role == "gate":
        return "required", None
    if role == "hard_fail":
        return "hard_fail", None
    if role == "additive":
        return (
            "invalid",
            "role 'additive' is invalid inside strict all_of; use a "
            "top-level graders[] entry or average_of for partial credit",
        )
    return (
        "invalid",
        f"predicate_leaf config role must be one of gate/hard_fail, got {role!r}",
    )


def _all_of_child_label(child: Any, index: int) -> str:
    if not isinstance(child, dict):
        return f"children[{index}]"
    if _is_leaf(child):
        return str(child.get("name") or f"children[{index}]")
    child_type = child.get("type")
    child_config = child.get("config")
    if child_type == "predicate_leaf" and isinstance(child_config, dict):
        return str(child_config.get("name") or f"children[{index}].predicate_leaf")
    return str(child.get("name") or f"children[{index}].{child_type or 'unknown'}")


def _duplicate_child_labels(children: list[Any]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for index, child in enumerate(children):
        label = _all_of_child_label(child, index)
        if label in seen and label not in duplicates:
            duplicates.append(label)
        seen.add(label)
    return duplicates


def _child_payload_configuration_error(
    passed: object,
    metrics: object,
    score: object,
    score_max: object,
    *,
    require_positive_score_max: bool,
) -> str | None:
    if not isinstance(passed, bool):
        return "child grader passed value must be boolean"
    if not isinstance(metrics, dict):
        return "child grader metrics must be an object"
    if not is_finite_number(score):
        return "child grader score must be a finite number"
    if not is_finite_number(score_max) or float(score_max) < 0.0:
        return "child grader score_max must be a finite non-negative number"
    if require_positive_score_max and float(score_max) <= 0.0:
        return "average_of scoring child must have positive score capacity"
    return None


def _child_info_has_configuration_error(info: dict[str, Any]) -> bool:
    if info.get("configuration_error") is not None:
        return True
    sub_metrics = info.get("sub_metrics")
    return isinstance(sub_metrics, dict) and (
        sub_metrics.get("configuration_error") is not None
    )


def _child_info_has_system_error(info: dict[str, Any]) -> bool:
    if info.get("grader_system_error") is True or info.get("grader_error") is not None:
        return True
    sub_metrics = info.get("sub_metrics")
    return isinstance(sub_metrics, dict) and (
        sub_metrics.get("grader_system_error") is True
        or sub_metrics.get("grader_error") is not None
    )


def _average_of_child_kind(child: Any) -> tuple[str, str | None]:
    """Classify one partial-credit child and return any configuration error."""

    if not isinstance(child, dict):
        return "invalid", "average_of child must be an object"

    if _is_leaf(child):
        role = child.get("role")
        if isinstance(role, str) and role in {"gate", "additive"}:
            return "scoring", None
        if role == "hard_fail":
            return "hard_fail", None
        return (
            "invalid",
            "bare predicate child role must be one of gate/additive/hard_fail, "
            f"got {role!r}",
        )

    if "role" in child:
        return (
            "invalid",
            "typed average_of children do not accept an outer role; every typed "
            "child is a scoring component",
        )

    child_type = child.get("type")
    child_config = child.get("config")
    if not isinstance(child_type, str):
        return "invalid", "average_of child missing string 'type'"
    if not isinstance(child_config, dict):
        return "invalid", "typed average_of child config must be an object"

    if child_type != "predicate_leaf":
        return "scoring", None

    role = child_config.get("role")
    if isinstance(role, str) and role in {"gate", "additive"}:
        return "scoring", None
    if role == "hard_fail":
        return "hard_fail", None
    return (
        "invalid",
        "predicate_leaf config role must be one of gate/additive/hard_fail, "
        f"got {role!r}",
    )


def _hashable(value: Any) -> Any:
    """Recursively convert lists/dicts into hashable tuples."""
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_hashable(v) for v in value), key=repr))
    if isinstance(value, dict):
        items = [(_hashable(k), _hashable(v)) for k, v in value.items()]
        return tuple(sorted(items, key=repr))
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    return repr(value)


def _normalize_match_key(value: Any, mode: str) -> Any:
    """``match_key_normalize: sort`` sorts a list-valued key before lookup."""
    if mode == "sort" and isinstance(value, list):
        return tuple(sorted((_hashable(v) for v in value), key=repr))
    return _hashable(value)


def _leaf_score_max(leaf: Any) -> float:
    if not isinstance(leaf, dict):
        return 0.0
    if leaf.get("role") == "hard_fail":
        return 0.0
    return predicate_score_max(leaf.get("predicate"))


def _list_match_additive_score_denominator(gt_entries: Any, k: Any = None) -> float:
    """Largest additive score an answer can actually reach.

    ``k`` truncates the agent list before grading, so at most ``k`` ground-truth
    rows are ever consumed. Normalizing against every row would then cap a
    flawless answer at ``k / len(ground_truth)``, so sum only the ``k``
    highest-capacity rows: "report any k of these rows" scores 1.0 when all k
    submitted rows are right. Without ``k`` every row stays in the denominator.
    """
    per_entry_score_max: list[float] = []
    if isinstance(gt_entries, list):
        for gt_entry in gt_entries:
            if not isinstance(gt_entry, dict):
                continue
            fields = gt_entry.get("fields")
            if not isinstance(fields, dict):
                continue
            per_entry_score_max.append(
                sum(
                    _leaf_score_max(leaf)
                    for leaf in fields.values()
                    if isinstance(leaf, dict) and leaf.get("role") == "additive"
                )
            )

    if isinstance(k, int) and not isinstance(k, bool) and k >= 0:
        per_entry_score_max.sort(reverse=True)
        del per_entry_score_max[k:]

    return sum(per_entry_score_max)


def _dict_match_entry_score_denominator(gt_entry: Any) -> float:
    if not isinstance(gt_entry, dict):
        return 0.0
    if "predicate" in gt_entry:
        return _leaf_score_max(gt_entry)
    fields = gt_entry.get("fields")
    if not isinstance(fields, dict):
        return 0.0
    return sum(_leaf_score_max(leaf) for leaf in fields.values())


def _bind_field(value: Any, field: Any) -> Any:
    """Same binding rules as :class:`PredicateLeafGrader`: plain-key or JSONPath.

    Yields ``MISSING`` when the answer does not supply the field at all, so the
    leaf is graded as a failure instead of letting the predicate see ``None``.
    """
    if field is None:
        return value
    if isinstance(field, str) and field.startswith("$"):
        try:
            return resolve_answer_field(value, field)
        except ValueError:
            return MISSING
    if isinstance(value, dict) and isinstance(field, str):
        return value.get(field, MISSING)
    return MISSING


def _evaluate_leaf(leaf: dict, value: Any) -> tuple[str, bool, float, float, str, dict]:
    """Returns (kind, passed, score, score_max, label, info)."""
    role = leaf.get("role")
    predicate = leaf.get("predicate")
    threshold = leaf.get("threshold", 1.0)
    op = predicate.get("op") if isinstance(predicate, dict) else None
    is_scalar = isinstance(op, str) and op in SCALAR_OPS
    label = leaf.get("name") or (f"{op}-leaf" if op else "(unnamed)")
    kind = "hard_fail" if role == "hard_fail" else "scoring"
    score_max = _leaf_score_max(leaf)

    answer_field = leaf.get("answer_field")
    if answer_field is not None:
        if not isinstance(answer_field, str) or answer_field == "":
            return (
                kind,
                False,
                0.0,
                score_max,
                label,
                {
                    "configuration_error": (
                        "answer_field must be a non-empty string when configured"
                    )
                },
            )
        if answer_field.startswith("$"):
            try:
                resolve_answer_field({}, answer_field)
            except ValueError as exc:
                return (
                    kind,
                    False,
                    0.0,
                    score_max,
                    label,
                    {"configuration_error": f"invalid answer_field JSONPath: {exc}"},
                )

    if not isinstance(role, str) or role not in {
        "gate",
        "additive",
        "hard_fail",
    }:
        return (
            kind,
            False,
            0.0,
            score_max,
            label,
            {"configuration_error": f"unknown predicate role {role!r}"},
        )

    predicate_error = predicate_configuration_error(predicate)
    if predicate_error is not None:
        return (
            kind,
            False,
            0.0,
            score_max,
            label,
            {"configuration_error": predicate_error, "role": role, "op": op},
        )

    threshold_error = _threshold_configuration_error(predicate, threshold)
    if threshold_error is not None:
        return (
            kind,
            False,
            0.0,
            score_max,
            label,
            {"configuration_error": threshold_error, "role": role, "op": op},
        )

    if role in {"gate", "additive"} and score_max <= 0.0:
        return (
            kind,
            False,
            0.0,
            score_max,
            label,
            {
                "configuration_error": (
                    "scoring predicate has no positive score capacity"
                ),
                "role": role,
                "op": op,
            },
        )

    if value is MISSING:
        # An explicitly bound answer field is part of the grading contract, so
        # skipping it is scored exactly like getting it wrong. For a hard-fail
        # leaf, ``passed=False`` triggers the veto and fails closed.
        #
        # A genuinely optional behavioural veto can instead omit
        # ``answer_field`` and inspect the whole answer with a ``field``
        # predicate. In that shape, an absent optional field evaluates to false
        # and leaves the veto untriggered.
        info = {"op": op, "role": role, "missing_answer_field": True}
        return kind, False, 0.0, score_max, label, info

    try:
        raw = evaluate_predicate(predicate, value)
    except (KeyError, TypeError, ValueError) as exc:
        return (
            kind,
            False,
            0.0,
            score_max,
            label,
            {
                "invalid_answer": str(exc),
                "role": role,
                "op": op,
            },
        )

    kind, passed, score = _apply_role(role, raw, is_scalar, threshold)
    return (
        kind,
        passed,
        score,
        score_max,
        label,
        {
            "op": op,
            "role": role,
            "raw": raw,
            "is_scalar": is_scalar,
        },
    )


def evaluate_composite_predicate_leaf(agent_answer: dict, config: dict) -> GraderResult:
    """Evaluate a predicate leaf with composite roles, including ``additive``."""

    value = _bind_field(agent_answer, config.get("answer_field"))
    kind, passed, score, score_max, label, info = _evaluate_leaf(config, value)
    return GraderResult(
        passed=passed,
        metrics={"kind": kind, "name": label, **info},
        reasoning=f"Composite predicate leaf {label!r}: {'PASS' if passed else 'FAIL'}",
        agent_answer=agent_answer,
        score=score,
        field_scores={label: score},
        score_max=score_max,
    )


def _evaluate_all_of_child(
    child: Any, agent_answer: Any, index: int
) -> tuple[str, bool, float, float, str, dict]:
    """Dispatch one ``all_of`` child: bare leaf vs composite envelope."""
    child_kind, role_error = _all_of_child_kind(child)
    label = _all_of_child_label(child, index)
    if role_error is not None:
        return (
            "invalid",
            False,
            0.0,
            1.0,
            label,
            {"configuration_error": role_error},
        )

    if _is_leaf(child):
        result = evaluate_composite_predicate_leaf(agent_answer, child)
        return (
            child_kind,
            result.passed,
            result.score,
            result.score_max,
            label,
            result.metrics,
        )

    from . import get_grader  # Lazy import is required to avoid a cycle.

    child_type = child.get("type") if isinstance(child, dict) else None
    child_config = child.get("config", {}) if isinstance(child, dict) else {}
    if not isinstance(child_type, str):
        return (
            "invalid",
            False,
            0.0,
            1.0,
            label,
            {
                "configuration_error": (
                    "composite child missing 'type' and is not a bare predicate leaf"
                )
            },
        )
    if child_type == "predicate_leaf" and isinstance(child_config, dict):
        sub = evaluate_composite_predicate_leaf(agent_answer, child_config)
        return (
            child_kind,
            sub.passed,
            sub.score,
            sub.score_max,
            label,
            sub.metrics,
        )

    try:
        grader = get_grader(child_type)
    except ValueError as exc:
        return (
            "invalid",
            False,
            0.0,
            1.0,
            label,
            {"configuration_error": str(exc)},
        )
    try:
        sub = grader.evaluate_answer(agent_answer, child_config)
    except Exception as exc:  # noqa: BLE001 -- isolate a broken child grader
        return (
            child_kind,
            False,
            0.0,
            1.0,
            label,
            {"grader_error": str(exc), "grader_system_error": True},
        )
    return (
        child_kind,
        sub.passed,
        sub.score,
        sub.score_max,
        label,
        {
            "sub_metrics": sub.metrics,
            "sub_reasoning": sub.reasoning,
        },
    )


def _evaluate_average_of_child(
    child: Any, agent_answer: Any, index: int
) -> tuple[str, bool, float, float, str, dict]:
    """Dispatch one ``average_of`` scoring component or hard-fail veto."""

    child_kind, role_error = _average_of_child_kind(child)
    label = _all_of_child_label(child, index)
    if role_error is not None:
        return (
            "invalid",
            False,
            0.0,
            1.0,
            label,
            {"configuration_error": role_error},
        )

    if _is_leaf(child):
        result = evaluate_composite_predicate_leaf(agent_answer, child)
        return (
            child_kind,
            result.passed,
            result.score,
            result.score_max,
            label,
            result.metrics,
        )

    from . import get_grader  # Lazy import is required to avoid a cycle.

    child_type = child["type"]
    child_config = child["config"]
    if child_type == "predicate_leaf":
        sub = evaluate_composite_predicate_leaf(agent_answer, child_config)
        return (
            child_kind,
            sub.passed,
            sub.score,
            sub.score_max,
            label,
            sub.metrics,
        )

    try:
        grader = get_grader(child_type)
    except ValueError as exc:
        return (
            "invalid",
            False,
            0.0,
            1.0,
            label,
            {"configuration_error": str(exc)},
        )
    try:
        sub = grader.evaluate_answer(agent_answer, child_config)
    except Exception as exc:  # noqa: BLE001 -- isolate a broken child grader
        return (
            child_kind,
            False,
            0.0,
            1.0,
            label,
            {"grader_error": str(exc), "grader_system_error": True},
        )
    return (
        child_kind,
        sub.passed,
        sub.score,
        sub.score_max,
        label,
        {
            "sub_metrics": sub.metrics,
            "sub_reasoning": sub.reasoning,
        },
    )


def _composite_fail(agent_answer: Any, reason: str) -> GraderResult:
    return GraderResult(
        passed=False,
        metrics={"error": reason},
        reasoning=f"composite grader: {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={},
    )


def _composite_configuration_fail(agent_answer: Any, reason: str) -> GraderResult:
    return GraderResult(
        passed=False,
        metrics={"configuration_error": reason},
        reasoning=f"composite grader configuration error: {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={},
    )


def _evaluate_average_of_pass_rule(
    config: dict, scoring_count: int, scoring_passed: int, scoring_total_score: float
) -> tuple[str, bool, str | None]:
    pass_rule = config.get("pass_rule", "all")
    if pass_rule == "all":
        if "min_passing_children" in config or "score_threshold" in config:
            return (
                pass_rule,
                False,
                "min_passing_children and score_threshold are only valid with "
                "their matching average_of pass_rule",
            )
        return pass_rule, scoring_passed == scoring_count, None

    if pass_rule == "min_passing":
        if "score_threshold" in config:
            return (
                pass_rule,
                False,
                "score_threshold is invalid with pass_rule='min_passing'",
            )
        minimum = config.get("min_passing_children")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or minimum < 0
            or minimum > scoring_count
        ):
            return (
                pass_rule,
                False,
                "min_passing_children must be an integer between 0 and the "
                "number of scoring children",
            )
        return pass_rule, scoring_passed >= minimum, None

    if pass_rule == "score_threshold":
        if "min_passing_children" in config:
            return (
                pass_rule,
                False,
                "min_passing_children is invalid with pass_rule='score_threshold'",
            )
        threshold = config.get("score_threshold")
        if not is_finite_number(threshold) or threshold < 0:
            return (
                pass_rule,
                False,
                "score_threshold must be a finite non-negative number",
            )
        return pass_rule, scoring_total_score >= float(threshold), None

    return (
        str(pass_rule),
        False,
        "average_of pass_rule must be one of all/min_passing/score_threshold",
    )


def _leaf_configuration_error(leaf: object) -> str | None:
    if not isinstance(leaf, dict):
        return "predicate leaf must be an object"
    _, _, _, _, _, info = _evaluate_leaf(leaf, MISSING)
    error = info.get("configuration_error")
    return str(error) if error is not None else None


def _list_match_configuration_error(config: object) -> str | None:
    if not isinstance(config, dict):
        return "list_match config must be an object"

    answer_field = config.get("answer_field")
    if not isinstance(answer_field, str) or answer_field == "":
        return "list_match answer_field must be a non-empty string"
    match_key = config.get("match_key")
    if not isinstance(match_key, str) or match_key == "":
        return "list_match match_key must be a non-empty string"

    normalize_mode = config.get("match_key_normalize", "none")
    if not isinstance(normalize_mode, str):
        return "list_match match_key_normalize must be a string"
    deduplicate_by = config.get("deduplicate_by")
    if deduplicate_by is not None and (
        not isinstance(deduplicate_by, str) or deduplicate_by == ""
    ):
        return "list_match deduplicate_by must be a non-empty string when configured"
    per_tuple_rule = config.get("per_tuple_rule", "gates_all_pass")
    if not isinstance(per_tuple_rule, str) or per_tuple_rule == "":
        return "list_match per_tuple_rule must be a non-empty string"

    k = config.get("k")
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k <= 0):
        return "list_match k must be a positive integer when configured"
    tuple_pass_min = config.get("tuple_pass_min", 0)
    if (
        isinstance(tuple_pass_min, bool)
        or not isinstance(tuple_pass_min, int)
        or tuple_pass_min < 0
    ):
        return "list_match tuple_pass_min must be a non-negative integer"
    if isinstance(k, int) and tuple_pass_min > k:
        return "list_match tuple_pass_min cannot exceed k"
    additive_score_min = config.get("additive_score_min", 0)
    if not is_finite_number(additive_score_min) or float(additive_score_min) < 0.0:
        return "list_match additive_score_min must be a finite non-negative number"

    ground_truth = config.get("ground_truth")
    if not isinstance(ground_truth, list) or not ground_truth:
        return "list_match ground_truth must be a non-empty list"
    if tuple_pass_min > len(ground_truth):
        return "list_match tuple_pass_min cannot exceed ground_truth size"

    seen_match_keys: set[Any] = set()
    positive_leaf_count = 0
    for index, entry in enumerate(ground_truth):
        if not isinstance(entry, dict):
            return f"list_match ground_truth[{index}] must be an object"
        if match_key not in entry:
            return (
                f"list_match ground_truth[{index}] is missing match_key {match_key!r}"
            )
        try:
            normalized_key = _normalize_match_key(entry[match_key], normalize_mode)
        except (TypeError, ValueError) as exc:
            return f"list_match ground_truth[{index}] has invalid match key: {exc}"
        if normalized_key in seen_match_keys:
            return f"list_match ground_truth contains duplicate match key {entry[match_key]!r}"
        seen_match_keys.add(normalized_key)

        fields = entry.get("fields")
        if not isinstance(fields, dict) or not fields:
            return f"list_match ground_truth[{index}].fields must be a non-empty object"
        for field_name, leaf in fields.items():
            error = _leaf_configuration_error(leaf)
            if error is not None:
                return (
                    f"list_match ground_truth[{index}].fields[{field_name!r}]: {error}"
                )
            if isinstance(leaf, dict) and leaf.get("role") != "hard_fail":
                positive_leaf_count += 1

    if positive_leaf_count == 0:
        return "list_match must contain at least one positive predicate leaf"
    return None


def _dict_match_configuration_error(config: object) -> str | None:
    if not isinstance(config, dict):
        return "dict_match config must be an object"

    answer_field = config.get("answer_field")
    if not isinstance(answer_field, str) or answer_field == "":
        return "dict_match answer_field must be a non-empty string"
    per_entry_rule = config.get("per_entry_rule", "gates_all_pass")
    if not isinstance(per_entry_rule, str) or per_entry_rule == "":
        return "dict_match per_entry_rule must be a non-empty string"
    all_keys_required = config.get("all_keys_required", True)
    if not isinstance(all_keys_required, bool):
        return "dict_match all_keys_required must be a boolean"

    ground_truth = config.get("ground_truth")
    if not isinstance(ground_truth, dict) or not ground_truth:
        return "dict_match ground_truth must be a non-empty object"

    positive_leaf_count = 0
    for key, entry in ground_truth.items():
        if not isinstance(entry, dict):
            return f"dict_match ground_truth[{key!r}] must be an object"
        if "predicate" in entry:
            error = _leaf_configuration_error(entry)
            if error is not None:
                return f"dict_match ground_truth[{key!r}]: {error}"
            if entry.get("role") != "hard_fail":
                positive_leaf_count += 1
            continue

        fields = entry.get("fields")
        if not isinstance(fields, dict) or not fields:
            return f"dict_match ground_truth[{key!r}].fields must be a non-empty object"
        for field_name, leaf in fields.items():
            error = _leaf_configuration_error(leaf)
            if error is not None:
                return (
                    f"dict_match ground_truth[{key!r}].fields[{field_name!r}]: {error}"
                )
            if isinstance(leaf, dict) and leaf.get("role") != "hard_fail":
                positive_leaf_count += 1

    if positive_leaf_count == 0:
        return "dict_match must contain at least one positive predicate leaf"
    return None


# composites vv


class AllOfGrader(BinaryGrader):
    """Strict conjunction: every positive child must pass for binary credit."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return _composite_configuration_fail(
                agent_answer, "all_of/composite config must be an object"
            )
        children = config.get("children")
        if not isinstance(children, list):
            return _composite_configuration_fail(
                agent_answer, "all_of/composite config must contain a children list"
            )
        duplicate_labels = _duplicate_child_labels(children)
        if duplicate_labels:
            return _composite_configuration_fail(
                agent_answer,
                f"all_of/composite child labels must be unique: {duplicate_labels}",
            )

        required: list[tuple[bool, float, float, str, dict]] = []
        hard_fails: list[tuple[bool, str, dict]] = []
        misconfigured_children: list[str] = []
        system_error_children: list[str] = []

        for index, child in enumerate(children):
            kind, passed, score, score_max, label, info = _evaluate_all_of_child(
                child, agent_answer, index
            )
            payload_error = _child_payload_configuration_error(
                passed,
                info.get("sub_metrics", info),
                score,
                score_max,
                require_positive_score_max=False,
            )
            if payload_error is not None:
                passed = False
                score = 0.0
                score_max = 0.0
                info = {**info, "configuration_error": payload_error}
            has_configuration_error = _child_info_has_configuration_error(info)
            has_system_error = _child_info_has_system_error(info)
            if (kind == "invalid" or has_configuration_error) and (
                label not in misconfigured_children
            ):
                misconfigured_children.append(label)
            if has_system_error and label not in system_error_children:
                system_error_children.append(label)
            if kind == "hard_fail":
                hard_fails.append((passed, label, info))
            elif kind != "invalid" and not has_configuration_error:
                required.append((passed, score, score_max, label, info))

        scoring_total_score = sum(
            score
            for _, score, _, _, info in required
            if not _child_info_has_system_error(info)
        )
        score_denominator = sum(s_max for _, _, s_max, *_ in required)
        scoring_count = len(required)
        scoring_passed = sum(
            1
            for passed, _, _, _, info in required
            if passed and not _child_info_has_system_error(info)
        )
        failed_children = [
            label
            for passed, _, _, label, info in required
            if not passed and not _child_info_has_system_error(info)
        ]

        hard_fail_triggered: list[Any] = []
        hard_fail_unavailable: list[Any] = []
        for passed, label, info in hard_fails:
            if passed:
                continue
            sub_metrics = info.get("sub_metrics")
            if (
                info.get("error") is not None
                or info.get("configuration_error") is not None
                or _child_info_has_system_error(info)
                or (
                    isinstance(sub_metrics, dict)
                    and (sub_metrics.get("configuration_error") is not None)
                )
            ):
                hard_fail_unavailable.append(label)
            else:
                hard_fail_triggered.append(label)

        for _, _, _, label, info in required:
            sub_metrics = info.get("sub_metrics")
            if not isinstance(sub_metrics, dict):
                continue
            nested_triggered = sub_metrics.get("hard_fail_triggered")
            if isinstance(nested_triggered, list) and nested_triggered:
                hard_fail_triggered.append(
                    {"child": label, "children": nested_triggered}
                )
            nested_unavailable = sub_metrics.get("hard_fail_unavailable")
            if isinstance(nested_unavailable, list) and nested_unavailable:
                hard_fail_unavailable.append(
                    {"child": label, "children": nested_unavailable}
                )

        pass_rule = config.get("pass_rule", "all")
        unsupported_pass_config: str | None = None
        if pass_rule != "all":
            unsupported_pass_config = (
                "strict all_of only supports pass_rule='all'; use top-level "
                "graders[] or average_of for partial credit"
            )
        elif "min_passing_children" in config or "score_threshold" in config:
            unsupported_pass_config = (
                "min_passing_children and score_threshold are invalid for strict all_of"
            )

        # A hard-fail-only composite can veto an answer but cannot establish
        # that any requested work was completed. Keep at least one positive
        # required child so a clean veto does not earn ungrounded full credit.
        no_positive_children = scoring_count == 0
        blocked = (
            bool(failed_children)
            or bool(hard_fail_triggered)
            or bool(hard_fail_unavailable)
            or bool(misconfigured_children)
            or bool(system_error_children)
            or unsupported_pass_config is not None
            or no_positive_children
        )
        passed = not blocked
        score = 1.0 if passed else 0.0
        configuration_error: str | None = None
        if misconfigured_children:
            configuration_error = (
                "all_of contains child graders that could not be configured"
            )
        elif no_positive_children:
            configuration_error = "all_of has no positive required children"
        elif unsupported_pass_config is not None:
            configuration_error = unsupported_pass_config

        return GraderResult(
            passed=passed,
            score=score,
            metrics={
                "type": "all_of",
                "pass_rule": pass_rule,
                "scoring_count": scoring_count,
                "scoring_passed": scoring_passed,
                "scoring_total_score": scoring_total_score,
                "score_denominator": score_denominator,
                "failed_children": failed_children,
                "hard_fail_triggered": hard_fail_triggered,
                **(
                    {"hard_fail_unavailable": hard_fail_unavailable}
                    if hard_fail_unavailable
                    else {}
                ),
                **(
                    {"configuration_error": configuration_error}
                    if configuration_error is not None
                    else {}
                ),
                **(
                    {"misconfigured_children": misconfigured_children}
                    if misconfigured_children
                    else {}
                ),
                **(
                    {
                        "grader_system_error": True,
                        "system_error_children": system_error_children,
                    }
                    if system_error_children
                    else {}
                ),
            },
            reasoning=_format_all_of(required, hard_fails, passed),
            agent_answer=agent_answer,
            field_scores={name: s for _, s, _, name, _ in required},
        )


class AverageOfGrader(BinaryGrader):
    """Normalized partial credit with an explicit binary passing policy."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return _composite_configuration_fail(
                agent_answer, "average_of config must be an object"
            )
        children = config.get("children")
        if not isinstance(children, list):
            return _composite_configuration_fail(
                agent_answer, "average_of config must contain a children list"
            )
        duplicate_labels = _duplicate_child_labels(children)
        if duplicate_labels:
            return _composite_configuration_fail(
                agent_answer,
                f"average_of child labels must be unique: {duplicate_labels}",
            )

        scoring: list[tuple[bool, float, float, str, dict]] = []
        hard_fails: list[tuple[bool, str, dict]] = []
        misconfigured_children: list[str] = []
        system_error_children: list[str] = []

        for index, child in enumerate(children):
            kind, passed, score, score_max, label, info = _evaluate_average_of_child(
                child, agent_answer, index
            )
            payload_error = _child_payload_configuration_error(
                passed,
                info.get("sub_metrics", info),
                score,
                score_max,
                require_positive_score_max=kind == "scoring",
            )
            if payload_error is not None:
                passed = False
                score = 0.0
                score_max = 0.0
                info = {**info, "configuration_error": payload_error}
            has_configuration_error = _child_info_has_configuration_error(info)
            has_system_error = _child_info_has_system_error(info)
            if (kind == "invalid" or has_configuration_error) and (
                label not in misconfigured_children
            ):
                misconfigured_children.append(label)
            if has_system_error and label not in system_error_children:
                system_error_children.append(label)
            if kind == "hard_fail":
                hard_fails.append((passed, label, info))
            elif kind != "invalid" and not has_configuration_error:
                scoring.append((passed, score, score_max, label, info))

        scoring_total_score = sum(
            score
            for _, score, _, _, info in scoring
            if not _child_info_has_system_error(info)
        )
        score_denominator = sum(score_max for _, _, score_max, *_ in scoring)
        scoring_count = len(scoring)
        scoring_passed = sum(
            1
            for passed, _, _, _, info in scoring
            if passed and not _child_info_has_system_error(info)
        )
        failed_children = [
            label
            for passed, _, _, label, info in scoring
            if not passed and not _child_info_has_system_error(info)
        ]
        pass_rule, scoring_ok, pass_rule_error = _evaluate_average_of_pass_rule(
            config, scoring_count, scoring_passed, scoring_total_score
        )

        hard_fail_triggered: list[Any] = []
        hard_fail_unavailable: list[Any] = []
        for passed, label, info in hard_fails:
            if passed:
                continue
            sub_metrics = info.get("sub_metrics")
            if (
                info.get("error") is not None
                or info.get("configuration_error") is not None
                or _child_info_has_system_error(info)
                or (
                    isinstance(sub_metrics, dict)
                    and (sub_metrics.get("configuration_error") is not None)
                )
            ):
                hard_fail_unavailable.append(label)
            else:
                hard_fail_triggered.append(label)

        for _, _, _, label, info in scoring:
            sub_metrics = info.get("sub_metrics")
            if not isinstance(sub_metrics, dict):
                continue
            nested_triggered = sub_metrics.get("hard_fail_triggered")
            if isinstance(nested_triggered, list) and nested_triggered:
                hard_fail_triggered.append(
                    {"child": label, "children": nested_triggered}
                )
            nested_unavailable = sub_metrics.get("hard_fail_unavailable")
            if isinstance(nested_unavailable, list) and nested_unavailable:
                hard_fail_unavailable.append(
                    {"child": label, "children": nested_unavailable}
                )

        no_scoring_children = scoring_count == 0
        invalid_score_values = (
            not math.isfinite(scoring_total_score)
            or not math.isfinite(score_denominator)
            or score_denominator <= 0.0
        )
        blocked = (
            bool(hard_fail_triggered)
            or bool(hard_fail_unavailable)
            or bool(misconfigured_children)
            or bool(system_error_children)
            or pass_rule_error is not None
            or no_scoring_children
            or invalid_score_values
        )
        score = (
            0.0 if blocked else normalize_score(scoring_total_score, score_denominator)
        )
        passed = scoring_ok and not blocked

        configuration_error: str | None = None
        if misconfigured_children:
            configuration_error = (
                "average_of contains child graders that could not be configured"
            )
        elif no_scoring_children:
            configuration_error = "average_of has no scoring children"
        elif invalid_score_values:
            configuration_error = (
                "average_of scoring children must provide a positive finite "
                "score denominator and finite total score"
            )
        elif pass_rule_error is not None:
            configuration_error = pass_rule_error

        return GraderResult(
            passed=passed,
            score=score,
            metrics={
                "type": "average_of",
                "pass_rule": pass_rule,
                "scoring_count": scoring_count,
                "scoring_passed": scoring_passed,
                "scoring_total_score": scoring_total_score,
                "score_denominator": score_denominator,
                "failed_children": failed_children,
                "hard_fail_triggered": hard_fail_triggered,
                **(
                    {"hard_fail_unavailable": hard_fail_unavailable}
                    if hard_fail_unavailable
                    else {}
                ),
                **(
                    {"configuration_error": configuration_error}
                    if configuration_error is not None
                    else {}
                ),
                **(
                    {"misconfigured_children": misconfigured_children}
                    if misconfigured_children
                    else {}
                ),
                **(
                    {
                        "grader_system_error": True,
                        "system_error_children": system_error_children,
                    }
                    if system_error_children
                    else {}
                ),
            },
            reasoning=_format_average_of(scoring, hard_fails, pass_rule, passed, score),
            agent_answer=agent_answer,
            field_scores={
                label: child_score for _, child_score, _, label, _ in scoring
            },
        )


class ListMatchGrader(BinaryGrader):
    """Pair tuples against GT by ``match_key``; per-tuple gate + additive scoring."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        configuration_error = _list_match_configuration_error(config)
        if configuration_error is not None:
            return _composite_configuration_fail(agent_answer, configuration_error)

        answer_field = config.get("answer_field")
        match_key = config.get("match_key")
        normalize_mode = config.get("match_key_normalize", "none")
        deduplicate_by = config.get("deduplicate_by")
        gt_entries = config.get("ground_truth", [])
        per_tuple_rule = config.get("per_tuple_rule", "gates_all_pass")
        tuple_pass_min = config.get("tuple_pass_min", 0)
        additive_score_min = config.get("additive_score_min", 0)
        k = config.get("k")

        agent_list = (
            agent_answer.get(answer_field) if isinstance(agent_answer, dict) else None
        )
        if not isinstance(agent_list, list):
            return _composite_fail(
                agent_answer, f"agent_answer.{answer_field!r} must be a list"
            )

        if deduplicate_by:
            seen: set = set()
            deduped: list = []
            for tup in agent_list:
                if not isinstance(tup, dict):
                    deduped.append(tup)
                    continue
                hk = _hashable(tup.get(deduplicate_by))
                if hk in seen:
                    continue
                seen.add(hk)
                deduped.append(tup)
            agent_list = deduped

        if isinstance(k, int) and not isinstance(k, bool) and k >= 0:
            agent_list = agent_list[:k]

        gt_by_key: dict = {
            _normalize_match_key(gt.get(match_key), normalize_mode): gt
            for gt in gt_entries
            if isinstance(gt, dict)
        }

        tuple_summaries: list[dict] = []
        tuple_pass_count = 0
        additive_score = 0.0
        field_scores: dict = {}
        consumed_gt_keys: set[Any] = set()
        hard_fail_triggered: list[str] = []

        for i, tup in enumerate(agent_list):
            if not isinstance(tup, dict):
                tuple_summaries.append(
                    {"index": i, "passed": False, "reason": "tuple is not a dict"}
                )
                continue
            key_val = tup.get(match_key)
            normalized = _normalize_match_key(key_val, normalize_mode)
            gt = gt_by_key.get(normalized)
            if gt is None:
                tuple_summaries.append(
                    {
                        "index": i,
                        "match_key_value": key_val,
                        "passed": False,
                        "reason": f"match_key {key_val!r} not in GT",
                    }
                )
                continue
            if normalized in consumed_gt_keys:
                tuple_summaries.append(
                    {
                        "index": i,
                        "match_key_value": key_val,
                        "passed": False,
                        "reason": f"duplicate match_key {key_val!r}",
                    }
                )
                continue
            consumed_gt_keys.add(normalized)

            per_field: dict = {}
            all_pass = True
            gate_pass = True
            for fname, leaf in gt.get("fields", {}).items():
                fvalue = tup.get(fname, MISSING)
                kind, passed, score, _, _, _ = _evaluate_leaf(leaf, fvalue)
                role = leaf.get("role") if isinstance(leaf, dict) else None
                per_field[fname] = {"passed": passed, "score": score, "role": role}
                field_scores[f"{key_val}.{fname}"] = score
                if kind == "hard_fail":
                    if not passed:
                        hard_fail_triggered.append(f"{key_val}.{fname}")
                    continue
                if role == "gate" and not passed:
                    gate_pass = False
                if not passed:
                    all_pass = False
                if role == "additive":
                    additive_score += score

            tuple_passed = gate_pass if per_tuple_rule == "gates_all_pass" else all_pass
            if tuple_passed:
                tuple_pass_count += 1
            tuple_summaries.append(
                {
                    "index": i,
                    "match_key_value": key_val,
                    "passed": tuple_passed,
                    "per_field": per_field,
                }
            )

        veto = len(hard_fail_triggered) > 0
        # An answer that matched no ground-truth entry graded nothing, so it must
        # not report a pass: `tuple_pass_min`/`additive_score_min` of 0 would
        # otherwise mark an empty list as passing, and that verdict propagates
        # when this grader is nested as an `all_of` child.
        nothing_graded = len(gt_by_key) > 0 and len(consumed_gt_keys) == 0
        passed = (
            (tuple_pass_count >= tuple_pass_min)
            and (additive_score >= additive_score_min)
            and not veto
            and not nothing_graded
        )
        score_denominator = _list_match_additive_score_denominator(
            list(gt_by_key.values()), k
        )
        score = 0.0 if veto else normalize_score(additive_score, score_denominator)

        return GraderResult(
            passed=passed,
            score=score,
            metrics={
                "k": k,
                "tuple_pass_count": tuple_pass_count,
                "tuple_pass_min": tuple_pass_min,
                "additive_score": additive_score,
                "additive_score_min": additive_score_min,
                "additive_score_denominator": score_denominator,
                "n_tuples_evaluated": len(agent_list),
                "hard_fail_triggered": hard_fail_triggered,
                **(
                    {
                        "composite_error": "answer matched none of the ground-truth entries"
                    }
                    if nothing_graded
                    else {}
                ),
            },
            reasoning=_format_list_match(
                answer_field,
                tuple_summaries,
                tuple_pass_count,
                additive_score,
                tuple_pass_min,
                additive_score_min,
                passed,
            ),
            agent_answer=agent_answer,
            field_scores=field_scores,
            score_max=1.0 if score_denominator > 0.0 else 0.0,
        )


class DictMatchGrader(BinaryGrader):
    """Key-by-key dispatch: scalar predicate-leaf or object-form per-field leaves."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        configuration_error = _dict_match_configuration_error(config)
        if configuration_error is not None:
            return _composite_configuration_fail(agent_answer, configuration_error)

        answer_field = config.get("answer_field")
        gt = config.get("ground_truth", {})
        per_entry_rule = config.get("per_entry_rule", "gates_all_pass")
        # Retained only to flag configs that still set it: omitted keys are now
        # graded as failures either way, so the flag can no longer buy credit.
        all_keys_required = config.get("all_keys_required", True)

        agent_dict = (
            agent_answer.get(answer_field) if isinstance(agent_answer, dict) else None
        )
        if not isinstance(agent_dict, dict):
            return _composite_fail(
                agent_answer, f"agent_answer.{answer_field!r} must be an object"
            )

        entry_results: list[dict] = []
        field_scores: dict = {}
        all_pass = True
        raw_score = 0.0
        score_denominator = 0.0
        hard_fail_triggered: list[str] = []

        for gt_key, gt_entry in gt.items():
            entry_score_denominator = _dict_match_entry_score_denominator(gt_entry)
            if gt_key not in agent_dict:
                # Omitting a graded key is scored exactly like answering it
                # wrong: it stays in the denominator and earns nothing.
                #
                # `all_keys_required: false` used to drop the key from both the
                # numerator and the denominator, so an answer that supplied one
                # easy key and skipped the rest normalised to full credit --
                # strictly better than attempting the rest and missing them.
                score_denominator += entry_score_denominator
                entry_results.append(
                    {"key": gt_key, "passed": False, "reason": "missing"}
                )
                field_scores[gt_key] = 0.0
                all_pass = False
                continue
            agent_val = agent_dict[gt_key]
            score_denominator += entry_score_denominator

            if isinstance(gt_entry, dict) and "predicate" in gt_entry:
                kind, passed, score, _, _, info = _evaluate_leaf(gt_entry, agent_val)
                entry_results.append(
                    {"key": gt_key, "passed": passed, "shape": "scalar", "info": info}
                )
                field_scores[gt_key] = score
                if kind != "hard_fail":
                    raw_score += score
                elif not passed:
                    hard_fail_triggered.append(gt_key)
                if not passed:
                    all_pass = False
                continue

            if isinstance(gt_entry, dict) and "fields" in gt_entry:
                per_field: dict = {}
                ok = True
                gate_ok = True
                for fname, leaf in gt_entry["fields"].items():
                    fvalue = (
                        agent_val.get(fname, MISSING)
                        if isinstance(agent_val, dict)
                        else MISSING
                    )
                    kind, passed, score, _, _, _ = _evaluate_leaf(leaf, fvalue)
                    role = leaf.get("role") if isinstance(leaf, dict) else None
                    per_field[fname] = {"passed": passed, "score": score, "role": role}
                    field_scores[f"{gt_key}.{fname}"] = score
                    if kind != "hard_fail":
                        raw_score += score
                    elif not passed:
                        hard_fail_triggered.append(f"{gt_key}.{fname}")
                    if not passed:
                        ok = False
                    if role == "gate" and not passed:
                        gate_ok = False
                entry_passed = gate_ok if per_entry_rule == "gates_all_pass" else ok
                entry_results.append(
                    {
                        "key": gt_key,
                        "passed": entry_passed,
                        "shape": "object",
                        "per_field": per_field,
                    }
                )
                if not entry_passed:
                    all_pass = False
                continue

            entry_results.append(
                {"key": gt_key, "passed": False, "reason": "malformed GT entry"}
            )
            field_scores[gt_key] = 0.0
            all_pass = False

        entries_total = len(entry_results)
        entries_passed = sum(1 for r in entry_results if r.get("passed"))
        veto = len(hard_fail_triggered) > 0
        # score_denominator is 0 only when the answer supplied none of the graded
        # keys (every entry was an omitted optional key). That is a non-answer, so
        # fail closed rather than paying full credit for ungraded work.
        nothing_graded = score_denominator <= 0.0
        if veto or nothing_graded:
            score = 0.0
        else:
            score = normalize_score(raw_score, score_denominator)
        passed = all_pass and not veto and not nothing_graded

        return GraderResult(
            passed=passed,
            score=score,
            metrics={
                "entries_total": entries_total,
                "entries_passed": entries_passed,
                "raw_score": raw_score,
                "score_denominator": score_denominator,
                "hard_fail_triggered": hard_fail_triggered,
                "failing_keys": [
                    r["key"] for r in entry_results if not r.get("passed")
                ],
                **(
                    {"composite_error": "answer supplied none of the graded keys"}
                    if nothing_graded
                    else {}
                ),
                **(
                    {"all_keys_required_ignored": "omitted keys are graded as failures"}
                    if not all_keys_required
                    else {}
                ),
            },
            reasoning=_format_dict_match(answer_field, entry_results, passed),
            agent_answer=agent_answer,
            field_scores=field_scores,
        )


# ----- reasoning formatters --------------------------------------------------


def _format_all_of(required: list, hard_fails: list, passed: bool) -> str:
    verdict = "PASS" if passed else "FAIL"
    lines = [f"all_of [strict AND]: {verdict}"]
    for p, score, _, label, info in required:
        marker = "+" if p else "x"
        sub = info.get("sub_reasoning")
        if sub:
            first = sub.splitlines()[0] if sub else ""
            lines.append(f"  {marker} {label}: {first}  (score={score})")
        else:
            lines.append(f"  {marker} {label}: passed={p}, score={score}")
    for p, label, _ in hard_fails:
        marker = "+" if p else "x"
        state = "not triggered" if p else "TRIGGERED"
        lines.append(f"  {marker} hard_fail {label!r}: {state}")
    return "\n".join(lines)


def _format_average_of(
    scoring: list,
    hard_fails: list,
    pass_rule: str,
    passed: bool,
    score: float,
) -> str:
    verdict = "PASS" if passed else "FAIL"
    lines = [
        f"average_of [pass_rule={pass_rule}]: {verdict} (normalized score={score})"
    ]
    for child_passed, child_score, score_max, label, info in scoring:
        marker = "+" if child_passed else "x"
        sub = info.get("sub_reasoning")
        if sub:
            first = sub.splitlines()[0]
            lines.append(
                f"  {marker} {label}: {first}  (score={child_score}/{score_max})"
            )
        else:
            lines.append(
                f"  {marker} {label}: passed={child_passed}, "
                f"score={child_score}/{score_max}"
            )
    for child_passed, label, _ in hard_fails:
        marker = "+" if child_passed else "x"
        state = "not triggered" if child_passed else "TRIGGERED/UNAVAILABLE"
        lines.append(f"  {marker} hard_fail {label!r}: {state}")
    return "\n".join(lines)


def _format_list_match(
    answer_field: Any,
    tuple_summaries: list,
    tuple_pass_count: int,
    additive_score: float,
    tuple_pass_min: int,
    additive_score_min: float,
    passed: bool,
) -> str:
    verdict = "PASS" if passed else "FAIL"
    lines = [f"list_match (answer_field={answer_field!r}): {verdict}"]
    for s in tuple_summaries:
        marker = "+" if s.get("passed") else "x"
        key_val = s.get("match_key_value", f"tuple[{s['index']}]")
        if "per_field" in s:
            inner = ", ".join(
                f"{fn}={'P' if fr['passed'] else 'F'}"
                for fn, fr in s["per_field"].items()
            )
            lines.append(f"  {marker} {key_val}: {inner}")
        else:
            lines.append(f"  {marker} {key_val}: {s.get('reason', '')}")
    lines.append(
        f"  tuple_pass: {tuple_pass_count}/{tuple_pass_min}  "
        f"additive_score: {additive_score}/{additive_score_min}"
    )
    return "\n".join(lines)


def _format_dict_match(answer_field: Any, entry_results: list, passed: bool) -> str:
    verdict = "PASS" if passed else "FAIL"
    lines = [f"dict_match (answer_field={answer_field!r}): {verdict}"]
    for r in entry_results:
        marker = "+" if r.get("passed") else "x"
        reason = r.get("reason") or r.get("shape", "")
        lines.append(f"  {marker} {r['key']}: {reason}")
    return "\n".join(lines)
