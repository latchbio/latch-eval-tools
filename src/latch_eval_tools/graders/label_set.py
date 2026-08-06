import math

from .base import BinaryGrader, GraderResult
from .list_contract import check_list_cardinality


def _normalize_labels(
    value: object, *, label: str
) -> tuple[list[str] | None, str | None]:
    if not isinstance(value, list):
        return None, f"{label} must be a list, got {type(value).__name__}"

    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, (str, int, float, bool)):
            return None, (
                f"{label}[{index}] must be a string or number, "
                f"got {type(item).__name__}"
            )
        if isinstance(item, float) and not math.isfinite(item):
            return None, f"{label}[{index}] must be finite, got {item!r}"
        normalized.append(str(item))
    return normalized, None


def _failed_result(
    agent_answer: object, reasoning: str, *, configuration_error: bool = False
) -> GraderResult:
    metrics = {"configuration_error": reasoning} if configuration_error else {}
    return GraderResult(
        passed=False,
        metrics=metrics,
        reasoning=reasoning,
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
    )


class LabelSetJaccardGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return _failed_result(
                agent_answer, "config must be an object", configuration_error=True
            )
        ground_truth_labels, ground_truth_error = _normalize_labels(
            config.get("ground_truth_labels"), label="ground_truth_labels"
        )
        if ground_truth_labels is None:
            return _failed_result(
                agent_answer,
                ground_truth_error or "Invalid ground_truth_labels",
                configuration_error=True,
            )
        if len(ground_truth_labels) == 0:
            return _failed_result(
                agent_answer,
                "ground_truth_labels must be a non-empty list",
                configuration_error=True,
            )

        scoring = config.get("scoring", {})
        if not isinstance(scoring, dict):
            return _failed_result(
                agent_answer,
                f"scoring must be an object, got {type(scoring).__name__}",
                configuration_error=True,
            )
        pass_threshold = scoring.get("pass_threshold", 0.90)
        if (
            not isinstance(pass_threshold, (int, float))
            or isinstance(pass_threshold, bool)
            or not 0 <= pass_threshold <= 1
        ):
            return _failed_result(
                agent_answer,
                f"pass_threshold must be a number in [0, 1], got {pass_threshold!r}",
                configuration_error=True,
            )

        answer_field = config.get("answer_field", "cell_types_predicted")
        if not isinstance(answer_field, str) or answer_field.strip() == "":
            return _failed_result(
                agent_answer,
                f"answer_field must be a non-empty string, got {answer_field!r}",
                configuration_error=True,
            )

        configured_cardinality = check_list_cardinality(
            [], config.get("expected_count")
        )
        if configured_cardinality.configuration_error is not None:
            return _failed_result(
                agent_answer,
                configured_cardinality.configuration_error,
                configuration_error=True,
            )

        if not isinstance(agent_answer, dict):
            return _failed_result(agent_answer, "agent answer must be an object")

        if answer_field not in agent_answer:
            return _failed_result(
                agent_answer, f"Agent answer missing required field: {answer_field}"
            )

        predicted_labels, predicted_error = _normalize_labels(
            agent_answer[answer_field], label=f"agent_answer.{answer_field}"
        )
        if predicted_labels is None:
            return _failed_result(
                agent_answer, predicted_error or "Invalid predicted labels"
            )

        cardinality = check_list_cardinality(
            predicted_labels, configured_cardinality.expected_count
        )

        ground_truth_set = set(ground_truth_labels)
        predicted_set = set(predicted_labels)
        intersection = ground_truth_set & predicted_set
        union = ground_truth_set | predicted_set

        jaccard_index = len(intersection) / len(union) if union else 0.0
        similarity_passed = jaccard_index >= float(pass_threshold)
        passed = similarity_passed and cardinality.passed

        true_positives = intersection
        false_positives = predicted_set - ground_truth_set
        false_negatives = ground_truth_set - predicted_set

        metrics = {
            "jaccard_index": jaccard_index,
            "pass_threshold": pass_threshold,
            "answer_field": answer_field,
            "true_positives": sorted(true_positives),
            "false_positives": sorted(false_positives),
            "false_negatives": sorted(false_negatives),
            "submitted_count": cardinality.submitted_count,
            "unique_count": cardinality.unique_count,
            "expected_count": cardinality.expected_count,
            "cardinality_pass": cardinality.passed,
            # Backward-compatible name for callers that consumed this metric
            # before raw/unique cardinality diagnostics were added.
            "predicted_count": len(predicted_set),
            "ground_truth_count": len(ground_truth_set),
        }

        lines = [
            f"Label Set Comparison: {'PASS' if passed else 'FAIL'}",
            "",
            (
                f"  {'+' if similarity_passed else 'x'} Jaccard Index: "
                f"{jaccard_index:.3f} (threshold: {float(pass_threshold):.3f})"
            ),
        ]
        if cardinality.expected_count is not None:
            lines.append(
                f"  {'+' if cardinality.passed else 'x'} Exact unique count: "
                f"submitted={cardinality.submitted_count}, "
                f"unique={cardinality.unique_count}, "
                f"expected={cardinality.expected_count}"
            )

        lines.extend(["", f"Correct Labels ({len(true_positives)}):"])
        if true_positives:
            lines.extend(f"  + {label}" for label in sorted(true_positives))
        else:
            lines.append("  None")

        lines.extend(["", f"Missing Labels ({len(false_negatives)}):"])
        if false_negatives:
            lines.extend(f"  - {label}" for label in sorted(false_negatives))
        else:
            lines.append("  None")

        lines.extend(["", f"Extra Labels ({len(false_positives)}):"])
        if false_positives:
            lines.extend(f"  ? {label}" for label in sorted(false_positives))
        else:
            lines.append("  None")

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning="\n".join(lines),
            agent_answer=agent_answer,
            score=1.0 if passed else 0.0,
        )
