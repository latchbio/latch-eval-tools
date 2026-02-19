from .base import BinaryGrader, GraderResult


class DistributionComparisonGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        ground_truth = config.get("ground_truth", {})
        tolerances = config.get("tolerances", {})

        gt_total_cells = ground_truth.get("total_cells")
        gt_distribution = ground_truth.get("cell_type_distribution", {})

        total_cells_tolerance = tolerances.get("total_cells", {})
        pct_tolerance_config = tolerances.get("cell_type_percentages", {})
        pct_tolerance = pct_tolerance_config.get("value", 3.0)

        if "cell_type_distribution" not in agent_answer:
            return GraderResult(
                passed=False,
                metrics={"score": 0.0},
                reasoning="Agent answer missing required field: cell_type_distribution",
                agent_answer=agent_answer,
                score=0.0,
            )

        agent_total_cells = agent_answer.get("total_cells")
        agent_distribution = agent_answer["cell_type_distribution"]

        metrics = {}
        all_pass = True
        failures = []
        component_scores = []

        if gt_total_cells is not None and agent_total_cells is not None:
            total_cells_tol_value = total_cells_tolerance.get("value", 0)
            total_cells_diff = abs(agent_total_cells - gt_total_cells)
            total_cells_pass = total_cells_diff <= total_cells_tol_value

            metrics["total_cells_actual"] = agent_total_cells
            metrics["total_cells_expected"] = gt_total_cells
            metrics["total_cells_diff"] = total_cells_diff
            metrics["total_cells_pass"] = total_cells_pass
            total_cells_score = self.score_with_tolerance(total_cells_diff, total_cells_tol_value)
            metrics["total_cells_score"] = self.clamp_score(total_cells_score)
            component_scores.append(self.clamp_score(total_cells_score))

            if not total_cells_pass:
                all_pass = False
                failures.append(f"total_cells: {agent_total_cells} vs {gt_total_cells} (diff: {total_cells_diff})")

        distribution_failures = []
        for cell_type, expected_pct in gt_distribution.items():
            if cell_type not in agent_distribution:
                all_pass = False
                failures.append(f"Missing cell type: {cell_type}")
                distribution_failures.append(cell_type)
                metrics[f"{cell_type}_actual"] = None
                metrics[f"{cell_type}_expected"] = expected_pct
                metrics[f"{cell_type}_diff"] = None
                metrics[f"{cell_type}_pass"] = False
                metrics[f"{cell_type}_score"] = 0.0
                component_scores.append(0.0)
                continue

            actual_pct = agent_distribution[cell_type]
            diff = abs(actual_pct - expected_pct)
            within_tolerance = diff <= pct_tolerance

            metrics[f"{cell_type}_actual"] = actual_pct
            metrics[f"{cell_type}_expected"] = expected_pct
            metrics[f"{cell_type}_diff"] = diff
            metrics[f"{cell_type}_pass"] = within_tolerance
            component_score = self.score_with_tolerance(diff, pct_tolerance)
            metrics[f"{cell_type}_score"] = self.clamp_score(component_score)
            component_scores.append(self.clamp_score(component_score))

            if not within_tolerance:
                all_pass = False
                failures.append(f"{cell_type}: {actual_pct:.2f}% vs {expected_pct:.2f}% (diff: {diff:.2f}%)")
                distribution_failures.append(cell_type)

        extra_types = set(agent_distribution.keys()) - set(gt_distribution.keys())
        if extra_types:
            metrics["extra_cell_types"] = sorted(list(extra_types))
            expected_types = len(gt_distribution)
            extra_penalty = expected_types / (expected_types + len(extra_types)) if expected_types > 0 else 0.0
            component_scores = [s * extra_penalty for s in component_scores]
            metrics["extra_type_penalty_factor"] = extra_penalty

        overall_score = sum(component_scores) / len(component_scores) if component_scores else 0.0
        metrics["score"] = self.clamp_score(overall_score)

        lines = [
            f"Distribution Comparison: {'PASS' if all_pass else 'FAIL'}",
            f"Overall score: {metrics['score']:.3f}",
            "",
            f"Cell type percentages (tolerance: +/-{pct_tolerance}%):"
        ]

        for cell_type in sorted(gt_distribution.keys()):
            expected = gt_distribution[cell_type]
            if cell_type not in agent_distribution:
                lines.append(f"  x {cell_type}: MISSING vs {expected:.2f}%")
            else:
                actual = agent_distribution[cell_type]
                diff = abs(actual - expected)
                within_tol = diff <= pct_tolerance
                check = "+" if within_tol else "x"
                score = metrics.get(f"{cell_type}_score", 0.0)
                lines.append(f"  {check} {cell_type}: {actual:.2f}% vs {expected:.2f}% (diff: {diff:.2f}%, score: {score:.3f})")

        if failures:
            lines.extend(["", "Failures:"])
            for failure in failures:
                lines.append(f"  - {failure}")

        return GraderResult(
            passed=all_pass,
            metrics=metrics,
            reasoning="\n".join(lines),
            agent_answer=agent_answer,
            score=self.clamp_score(overall_score),
        )
