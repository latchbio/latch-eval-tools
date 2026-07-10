"""Composite graders: all_of, list_match, dict_match. Recursive over
predicate-leaves and nested composites."""

from typing import Any

from .base import BinaryGrader, GraderResult
from .predicate import (
    SCALAR_OPS,
    _apply_role,
    evaluate_predicate,
    resolve_answer_field,
)

# shared helper functions vv


def _is_leaf(node: Any) -> bool:
    """A node is a leaf iff it is a bare ``{predicate, role, ...}`` envelope."""
    return isinstance(node, dict) and "predicate" in node and "type" not in node


def _hashable(value: Any) -> Any:
    """Recursively convert lists/dicts into hashable tuples."""
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _normalize_match_key(value: Any, mode: str) -> Any:
    """``match_key_normalize: sort`` sorts a list-valued key before lookup."""
    if mode == "sort" and isinstance(value, list):
        return tuple(sorted(_hashable(v) for v in value))
    return _hashable(value)


def _as_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _predicate_score_max(predicate: Any) -> float:
    if not isinstance(predicate, dict):
        return 1.0

    if predicate.get("op") != "weighted_label":
        return 1.0

    scores: list[float] = []
    table = predicate.get("table")
    if isinstance(table, dict):
        for raw_score in table.values():
            score = _as_numeric(raw_score)
            if score is not None:
                scores.append(score)

    default_score = _as_numeric(predicate.get("default", 0))
    if default_score is not None:
        scores.append(default_score)

    positive_scores = [score for score in scores if score > 0.0]
    if not positive_scores:
        return 0.0
    return max(positive_scores)


def _leaf_score_max(leaf: Any) -> float:
    if not isinstance(leaf, dict):
        return 0.0
    if leaf.get("role") == "hard_fail":
        return 0.0
    return _predicate_score_max(leaf.get("predicate"))


def _normalize_score(score: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0

    normalized = score / denominator
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def _list_match_additive_score_denominator(gt_entries: Any) -> float:
    additive_score_max = 0.0
    if isinstance(gt_entries, list):
        for gt_entry in gt_entries:
            if not isinstance(gt_entry, dict):
                continue
            fields = gt_entry.get("fields")
            if not isinstance(fields, dict):
                continue
            for leaf in fields.values():
                if isinstance(leaf, dict) and leaf.get("role") == "additive":
                    additive_score_max += _leaf_score_max(leaf)

    return additive_score_max


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
    """Same binding rules as :class:`PredicateLeafGrader`: plain-key or JSONPath."""
    if field is None:
        return value
    if isinstance(field, str) and field.startswith("$"):
        try:
            return resolve_answer_field(value, field)
        except ValueError:
            return None
    if isinstance(value, dict) and isinstance(field, str):
        return value.get(field)
    return None


def _evaluate_leaf(leaf: dict, value: Any) -> tuple[str, bool, float, float, str, dict]:
    """Returns (kind, passed, score, score_max, label, info)."""
    role = leaf.get("role")
    predicate = leaf.get("predicate")
    threshold = leaf.get("threshold", 1.0)
    op = predicate.get("op") if isinstance(predicate, dict) else None
    is_scalar = op in SCALAR_OPS
    label = leaf.get("name") or (f"{op}-leaf" if op else "(unnamed)")
    kind = "hard_fail" if role == "hard_fail" else "scoring"
    score_max = _leaf_score_max(leaf)

    try:
        raw = evaluate_predicate(predicate, value)
    except (ValueError, TypeError) as exc:
        return (
            kind,
            False,
            0.0,
            score_max,
            label,
            {
                "error": str(exc),
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


def _evaluate_all_of_child(
    child: Any, agent_answer: Any
) -> tuple[str, bool, float, float, str, dict]:
    """Dispatch one ``all_of`` child: bare leaf vs composite envelope."""
    if _is_leaf(child):
        value = _bind_field(agent_answer, child.get("answer_field"))
        return _evaluate_leaf(child, value)

    from . import get_grader  # noqa: PLC0415 -- lazy to avoid circular import

    child_type = child.get("type") if isinstance(child, dict) else None
    child_config = child.get("config", {}) if isinstance(child, dict) else {}
    if not isinstance(child_type, str):
        return (
            "scoring",
            False,
            0.0,
            1.0,
            "?",
            {"error": "composite child missing 'type' and not a bare predicate-leaf"},
        )
    try:
        sub = get_grader(child_type).evaluate_answer(agent_answer, child_config)
    except (ValueError, TypeError) as exc:
        return "scoring", False, 0.0, 1.0, child_type, {"error": str(exc)}
    score_max = 1.0
    if child_type == "predicate_leaf" and isinstance(child_config, dict):
        score_max = _leaf_score_max(child_config)
    return (
        "scoring",
        sub.passed,
        sub.score,
        score_max,
        child_type,
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


# composites vv


class AllOfGrader(BinaryGrader):
    """Conjunction over scoring children plus veto from hard_fail children."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        scoring: list[tuple[bool, float, float, str, dict]] = []
        hard_fails: list[tuple[bool, str, dict]] = []

        for child in config.get("children", []):
            kind, passed, score, score_max, label, info = _evaluate_all_of_child(
                child, agent_answer
            )
            if kind == "hard_fail":
                hard_fails.append((passed, label, info))
            else:
                scoring.append((passed, score, score_max, label, info))

        scoring_total_score = sum(s for _, s, *_ in scoring)
        score_denominator = sum(s_max for _, _, s_max, *_ in scoring)
        scoring_count = len(scoring)
        scoring_passed = sum(1 for p, *_ in scoring if p)
        veto = any(not p for p, _, _ in hard_fails)
        pass_rule = config.get("pass_rule", "all")
        if pass_rule == "all":
            scoring_ok = all(p for p, *_ in scoring) if scoring else True
        elif pass_rule == "min_passing":
            scoring_ok = scoring_passed >= config.get("min_passing_children", 0)
        elif pass_rule == "score_threshold":
            scoring_ok = scoring_total_score >= config.get("score_threshold", 0.0)
        else:
            scoring_ok = False

        score = _normalize_score(scoring_total_score, score_denominator)

        passed = scoring_ok and not veto
        return GraderResult(
            passed=passed,
            score=score,
            metrics={
                "pass_rule": pass_rule,
                "scoring_count": scoring_count,
                "scoring_passed": scoring_passed,
                "scoring_total_score": scoring_total_score,
                "score_denominator": score_denominator,
                "hard_fail_triggered": [name for p, name, _ in hard_fails if not p],
            },
            reasoning=_format_all_of(scoring, hard_fails, pass_rule, passed),
            agent_answer=agent_answer,
            field_scores={name: s for _, s, _, name, _ in scoring},
        )


class ListMatchGrader(BinaryGrader):
    """Pair tuples against GT by ``match_key``; per-tuple gate + additive scoring."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
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

        if isinstance(k, int) and k >= 0:
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
                fvalue = tup.get(fname)
                _, passed, score, _, _, _ = _evaluate_leaf(leaf, fvalue)
                role = leaf.get("role") if isinstance(leaf, dict) else None
                per_field[fname] = {"passed": passed, "score": score, "role": role}
                field_scores[f"{key_val}.{fname}"] = score
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

        passed = (tuple_pass_count >= tuple_pass_min) and (
            additive_score >= additive_score_min
        )
        score_denominator = _list_match_additive_score_denominator(
            list(gt_by_key.values())
        )
        score = _normalize_score(additive_score, score_denominator)

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
        )


class DictMatchGrader(BinaryGrader):
    """Key-by-key dispatch: scalar predicate-leaf or object-form per-field leaves."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        answer_field = config.get("answer_field")
        gt = config.get("ground_truth", {})
        per_entry_rule = config.get("per_entry_rule", "gates_all_pass")
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

        for gt_key, gt_entry in gt.items():
            entry_score_denominator = _dict_match_entry_score_denominator(gt_entry)
            if gt_key not in agent_dict:
                if all_keys_required:
                    score_denominator += entry_score_denominator
                    entry_results.append(
                        {"key": gt_key, "passed": False, "reason": "missing"}
                    )
                    field_scores[gt_key] = 0.0
                    all_pass = False
                else:
                    raw_score += entry_score_denominator
                    score_denominator += entry_score_denominator
                    entry_results.append(
                        {
                            "key": gt_key,
                            "passed": True,
                            "reason": "missing-but-not-required",
                        }
                    )
                    field_scores[gt_key] = 0.0
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
                if not passed:
                    all_pass = False
                continue

            if isinstance(gt_entry, dict) and "fields" in gt_entry:
                per_field: dict = {}
                ok = True
                gate_ok = True
                for fname, leaf in gt_entry["fields"].items():
                    fvalue = (
                        agent_val.get(fname) if isinstance(agent_val, dict) else None
                    )
                    kind, passed, score, _, _, _ = _evaluate_leaf(leaf, fvalue)
                    role = leaf.get("role") if isinstance(leaf, dict) else None
                    per_field[fname] = {"passed": passed, "score": score, "role": role}
                    field_scores[f"{gt_key}.{fname}"] = score
                    if kind != "hard_fail":
                        raw_score += score
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
        score = _normalize_score(raw_score, score_denominator)

        return GraderResult(
            passed=all_pass,
            score=score,
            metrics={
                "entries_total": entries_total,
                "entries_passed": entries_passed,
                "raw_score": raw_score,
                "score_denominator": score_denominator,
                "failing_keys": [
                    r["key"] for r in entry_results if not r.get("passed")
                ],
            },
            reasoning=_format_dict_match(answer_field, entry_results, all_pass),
            agent_answer=agent_answer,
            field_scores=field_scores,
        )


# ----- reasoning formatters --------------------------------------------------


def _format_all_of(
    scoring: list, hard_fails: list, pass_rule: str, passed: bool
) -> str:
    verdict = "PASS" if passed else "FAIL"
    lines = [f"all_of [pass_rule={pass_rule}]: {verdict}"]
    for p, score, _, label, info in scoring:
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
