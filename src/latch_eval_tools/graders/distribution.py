from .base import BinaryGrader, GraderResult, configuration_error_result
from .number_contract import is_finite_number


def _validate_configuration(config: object) -> tuple[dict, dict, str | None]:
    if not isinstance(config, dict):
        return {}, {}, "config must be an object"

    ground_truth = config.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return {}, {}, "ground_truth must be an object"
    distribution = ground_truth.get("cell_type_distribution")
    if not isinstance(distribution, dict) or len(distribution) == 0:
        return (
            {},
            {},
            "ground_truth.cell_type_distribution must be a non-empty object",
        )
    for cell_type, expected_pct in distribution.items():
        if not isinstance(cell_type, str) or cell_type.strip() == "":
            return {}, {}, "cell-type names must be non-empty strings"
        if not is_finite_number(expected_pct):
            return (
                {},
                {},
                (
                    "ground_truth.cell_type_distribution"
                    f"[{cell_type!r}] must be a finite number"
                ),
            )

    gt_total_cells = ground_truth.get("total_cells")
    if gt_total_cells is not None and not is_finite_number(gt_total_cells):
        return {}, {}, "ground_truth.total_cells must be a finite number"

    tolerances = config.get("tolerances", {})
    if not isinstance(tolerances, dict):
        return {}, {}, "tolerances must be an object"
    for key in ("total_cells", "cell_type_percentages"):
        if key not in tolerances:
            continue
        tolerance = tolerances[key]
        if not isinstance(tolerance, dict):
            return {}, {}, f"tolerances.{key} must be an object"
        value = tolerance.get("value", 0 if key == "total_cells" else 3.0)
        if not is_finite_number(value) or float(value) < 0.0:
            return (
                {},
                {},
                f"tolerances.{key}.value must be a finite non-negative number",
            )

    return ground_truth, tolerances, None


def _answer_failure(agent_answer: object, reason: str) -> GraderResult:
    return GraderResult(
        passed=False,
        metrics={},
        reasoning=f"Distribution Comparison: FAIL\n\n  x {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
    )


class DistributionComparisonGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        ground_truth, tolerances, error = _validate_configuration(config)
        if error is not None:
            return configuration_error_result(
                agent_answer, "Distribution Comparison", error
            )
        if not isinstance(agent_answer, dict):
            return _answer_failure(agent_answer, "agent answer must be an object")

        gt_total_cells = ground_truth.get("total_cells")
        gt_distribution = ground_truth["cell_type_distribution"]

        total_cells_tolerance = tolerances.get("total_cells", {})
        pct_tolerance_config = tolerances.get("cell_type_percentages", {})
        pct_tolerance = pct_tolerance_config.get("value", 3.0)

        if "cell_type_distribution" not in agent_answer:
            return _answer_failure(
                agent_answer,
                "Agent answer missing required field: cell_type_distribution",
            )

        agent_total_cells = agent_answer.get("total_cells")
        agent_distribution = agent_answer["cell_type_distribution"]
        if not isinstance(agent_distribution, dict):
            return _answer_failure(
                agent_answer, "cell_type_distribution must be an object"
            )
        if not all(
            isinstance(cell_type, str) and cell_type.strip() != ""
            for cell_type in agent_distribution
        ):
            return _answer_failure(
                agent_answer, "cell_type_distribution keys must be non-empty strings"
            )

        metrics = {}
        all_pass = True
        failures = []

        if gt_total_cells is not None:
            if "total_cells" not in agent_answer or agent_total_cells is None:
                all_pass = False
                failures.append("Missing field: total_cells")
                metrics["total_cells_actual"] = None
                metrics["total_cells_expected"] = gt_total_cells
                metrics["total_cells_diff"] = None
                metrics["total_cells_pass"] = False
            elif not is_finite_number(agent_total_cells):
                return _answer_failure(
                    agent_answer, "total_cells must be a finite number"
                )
            else:
                total_cells_tol_value = total_cells_tolerance.get("value", 0)
                total_cells_diff = abs(agent_total_cells - gt_total_cells)
                total_cells_pass = total_cells_diff <= total_cells_tol_value

                metrics["total_cells_actual"] = agent_total_cells
                metrics["total_cells_expected"] = gt_total_cells
                metrics["total_cells_diff"] = total_cells_diff
                metrics["total_cells_pass"] = total_cells_pass

                if not total_cells_pass:
                    all_pass = False
                    failures.append(
                        f"total_cells: {agent_total_cells} vs {gt_total_cells} "
                        f"(diff: {total_cells_diff})"
                    )

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
                continue

            actual_pct = agent_distribution[cell_type]
            if not is_finite_number(actual_pct):
                all_pass = False
                failures.append(f"{cell_type}: expected a finite numeric percentage")
                distribution_failures.append(cell_type)
                metrics[f"{cell_type}_actual"] = actual_pct
                metrics[f"{cell_type}_expected"] = expected_pct
                metrics[f"{cell_type}_diff"] = None
                metrics[f"{cell_type}_pass"] = False
                continue
            diff = abs(actual_pct - expected_pct)
            within_tolerance = diff <= pct_tolerance

            metrics[f"{cell_type}_actual"] = actual_pct
            metrics[f"{cell_type}_expected"] = expected_pct
            metrics[f"{cell_type}_diff"] = diff
            metrics[f"{cell_type}_pass"] = within_tolerance

            if not within_tolerance:
                all_pass = False
                failures.append(
                    f"{cell_type}: {actual_pct:.2f}% vs {expected_pct:.2f}% (diff: {diff:.2f}%)"
                )
                distribution_failures.append(cell_type)

        extra_types = set(agent_distribution.keys()) - set(gt_distribution.keys())
        if extra_types:
            metrics["extra_cell_types"] = sorted(extra_types)

        lines = [
            f"Distribution Comparison: {'PASS' if all_pass else 'FAIL'}",
            "",
            f"Cell type percentages (tolerance: +/-{pct_tolerance}%):",
        ]

        for cell_type in sorted(gt_distribution.keys()):
            expected = gt_distribution[cell_type]
            if cell_type not in agent_distribution:
                lines.append(f"  x {cell_type}: MISSING vs {expected:.2f}%")
            else:
                actual = agent_distribution[cell_type]
                if not is_finite_number(actual):
                    lines.append(f"  x {cell_type}: NON-NUMERIC vs {expected:.2f}%")
                    continue
                diff = abs(actual - expected)
                within_tol = diff <= pct_tolerance
                check = "+" if within_tol else "x"
                lines.append(
                    f"  {check} {cell_type}: {actual:.2f}% vs {expected:.2f}% (diff: {diff:.2f}%)"
                )

        if failures:
            lines.extend(["", "Failures:"])
            for failure in failures:
                lines.append(f"  - {failure}")

        return GraderResult(
            passed=all_pass,
            metrics=metrics,
            reasoning="\n".join(lines),
            agent_answer=agent_answer,
            score=1.0 if all_pass else 0.0,
        )
