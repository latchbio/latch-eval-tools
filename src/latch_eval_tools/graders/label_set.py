from .base import BinaryGrader, GraderResult


class LabelSetJaccardGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        ground_truth_labels = set(config.get("ground_truth_labels", []))
        scoring = config.get("scoring", {})
        pass_threshold = scoring.get("pass_threshold", 0.90)
        answer_field = config.get("answer_field", "cell_types_predicted")

        if answer_field not in agent_answer:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning=f"Agent answer missing required field: {answer_field}",
                agent_answer=agent_answer
            )

        predicted_labels = set(agent_answer[answer_field])

        intersection = ground_truth_labels & predicted_labels
        union = ground_truth_labels | predicted_labels

        jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0.0
        passed = jaccard_index >= pass_threshold

        true_positives = intersection
        false_positives = predicted_labels - ground_truth_labels
        false_negatives = ground_truth_labels - predicted_labels

        metrics = {
            "jaccard_index": jaccard_index,
            "pass_threshold": pass_threshold,
            "answer_field": answer_field,
            "true_positives": sorted(list(true_positives)),
            "false_positives": sorted(list(false_positives)),
            "false_negatives": sorted(list(false_negatives)),
            "predicted_count": len(predicted_labels),
            "ground_truth_count": len(ground_truth_labels),
        }

        lines = [
            f"Label Set Comparison: {'PASS' if passed else 'FAIL'}",
            "",
            f"  {'+'if passed else 'x'} Jaccard Index: {jaccard_index:.3f} (threshold: {pass_threshold:.3f})",
            "",
            f"Correct Labels ({len(true_positives)}):"
        ]

        if true_positives:
            for label in sorted(true_positives):
                lines.append(f"  + {label}")
        else:
            lines.append("  None")

        lines.extend(["", f"Missing Labels ({len(false_negatives)}):"])
        if false_negatives:
            for label in sorted(false_negatives):
                lines.append(f"  - {label}")
        else:
            lines.append("  None")

        lines.extend(["", f"Extra Labels ({len(false_positives)}):"])
        if false_positives:
            for label in sorted(false_positives):
                lines.append(f"  ? {label}")
        else:
            lines.append("  None")

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning="\n".join(lines),
            agent_answer=agent_answer
        )
