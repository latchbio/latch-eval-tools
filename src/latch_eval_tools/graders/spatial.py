import math

from .base import BinaryGrader, GraderResult, configuration_error_result


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        # int magnitudes beyond ~1.8e308 overflow float() but are not finite.
        return False


def _answer_failure(agent_answer: object, reason: str) -> GraderResult:
    return GraderResult(
        passed=False,
        metrics={},
        reasoning=f"Spatial Adjacency Analysis: FAIL\n\n  x {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
    )


class SpatialAdjacencyGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return configuration_error_result(
                agent_answer,
                "Spatial Adjacency Analysis",
                "config must be an object",
            )
        scoring = config.get("scoring", {})
        if not isinstance(scoring, dict):
            return configuration_error_result(
                agent_answer,
                "Spatial Adjacency Analysis",
                "scoring must be an object",
            )
        thresholds = scoring.get("pass_thresholds", {})
        if not isinstance(thresholds, dict):
            return configuration_error_result(
                agent_answer,
                "Spatial Adjacency Analysis",
                "scoring.pass_thresholds must be an object",
            )

        for key in ("max_median_ic_to_pc_um", "max_p90_ic_to_pc_um"):
            if key in thresholds and (
                not _is_finite_number(thresholds[key]) or float(thresholds[key]) < 0.0
            ):
                return configuration_error_result(
                    agent_answer,
                    "Spatial Adjacency Analysis",
                    f"scoring.pass_thresholds.{key} must be a finite "
                    "non-negative number",
                )
        for key in ("min_pct_ic_within_15um", "min_pct_ic_mixed_within_55um"):
            if key in thresholds and (
                not _is_finite_number(thresholds[key])
                or not 0.0 <= float(thresholds[key]) <= 100.0
            ):
                return configuration_error_result(
                    agent_answer,
                    "Spatial Adjacency Analysis",
                    f"scoring.pass_thresholds.{key} must be a finite number "
                    "in [0, 100]",
                )

        max_median_ic_to_pc = thresholds.get("max_median_ic_to_pc_um", 25.0)
        max_p90_ic_to_pc = thresholds.get("max_p90_ic_to_pc_um", 80.0)
        min_pct_within_15um = thresholds.get("min_pct_ic_within_15um", 60.0)
        min_pct_mixed_within_55um = thresholds.get("min_pct_ic_mixed_within_55um", 60.0)

        required_fields = [
            "median_ic_to_pc_um",
            "p90_ic_to_pc_um",
            "pct_ic_within_15um",
            "pct_ic_mixed_within_55um",
            "adjacency_pass",
        ]

        if not isinstance(agent_answer, dict):
            return _answer_failure(agent_answer, "agent answer must be an object")

        missing = [f for f in required_fields if f not in agent_answer]
        if missing:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning=f"Agent answer missing required fields: {missing}",
                agent_answer=agent_answer,
                score=0.0,
            )

        median_ic_to_pc = agent_answer["median_ic_to_pc_um"]
        p90_ic_to_pc = agent_answer["p90_ic_to_pc_um"]
        pct_within_15um = agent_answer["pct_ic_within_15um"]
        pct_mixed_within_55um = agent_answer["pct_ic_mixed_within_55um"]
        adjacency_pass = agent_answer["adjacency_pass"]

        numeric_values = {
            "median_ic_to_pc_um": median_ic_to_pc,
            "p90_ic_to_pc_um": p90_ic_to_pc,
            "pct_ic_within_15um": pct_within_15um,
            "pct_ic_mixed_within_55um": pct_mixed_within_55um,
        }
        for field, value in numeric_values.items():
            if not _is_finite_number(value):
                return _answer_failure(agent_answer, f"{field} must be a finite number")
        if float(median_ic_to_pc) < 0.0 or float(p90_ic_to_pc) < 0.0:
            return _answer_failure(
                agent_answer, "distance metrics must be non-negative"
            )
        if (
            not 0.0 <= float(pct_within_15um) <= 100.0
            or not 0.0 <= float(pct_mixed_within_55um) <= 100.0
        ):
            return _answer_failure(
                agent_answer, "percentage metrics must be in [0, 100]"
            )
        if not isinstance(adjacency_pass, bool):
            return _answer_failure(agent_answer, "adjacency_pass must be a boolean")

        median_pass = median_ic_to_pc <= max_median_ic_to_pc
        p90_pass = p90_ic_to_pc <= max_p90_ic_to_pc
        within_15um_pass = pct_within_15um >= min_pct_within_15um
        mixed_55um_pass = pct_mixed_within_55um >= min_pct_mixed_within_55um

        passed = (
            median_pass
            and p90_pass
            and within_15um_pass
            and mixed_55um_pass
            and adjacency_pass
        )

        metrics = {
            "median_ic_to_pc_um": median_ic_to_pc,
            "p90_ic_to_pc_um": p90_ic_to_pc,
            "pct_ic_within_15um": pct_within_15um,
            "pct_ic_mixed_within_55um": pct_mixed_within_55um,
            "adjacency_pass": adjacency_pass,
            "max_median_threshold": max_median_ic_to_pc,
            "max_p90_threshold": max_p90_ic_to_pc,
            "min_pct_15um_threshold": min_pct_within_15um,
            "min_pct_55um_threshold": min_pct_mixed_within_55um,
            "median_pass": median_pass,
            "p90_pass": p90_pass,
            "within_15um_pass": within_15um_pass,
            "mixed_55um_pass": mixed_55um_pass,
        }

        lines = [
            f"Spatial Adjacency Analysis: {'PASS' if passed else 'FAIL'}",
            "",
            "IC->PC Distance Metrics:",
            f"  {'+' if median_pass else 'x'} Median distance: {median_ic_to_pc:.2f} um (threshold: <={max_median_ic_to_pc:.2f} um)",
            f"  {'+' if p90_pass else 'x'} 90th percentile: {p90_ic_to_pc:.2f} um (threshold: <={max_p90_ic_to_pc:.2f} um)",
            "",
            "IC Proximity to PC:",
            f"  {'+' if within_15um_pass else 'x'} IC within 15 um: {pct_within_15um:.1f}% (threshold: >={min_pct_within_15um:.1f}%)",
            f"  {'+' if mixed_55um_pass else 'x'} IC with PC within 55 um: {pct_mixed_within_55um:.1f}% (threshold: >={min_pct_mixed_within_55um:.1f}%)",
            "",
            f"Agent adjacency assessment: {'+' if adjacency_pass else 'x'} {adjacency_pass}",
        ]

        if not passed:
            failures = []
            if not median_pass:
                failures.append(
                    f"Median {median_ic_to_pc:.2f} > {max_median_ic_to_pc:.2f} um"
                )
            if not p90_pass:
                failures.append(f"P90 {p90_ic_to_pc:.2f} > {max_p90_ic_to_pc:.2f} um")
            if not within_15um_pass:
                failures.append(
                    f"Within 15 um {pct_within_15um:.1f}% < {min_pct_within_15um:.1f}%"
                )
            if not mixed_55um_pass:
                failures.append(
                    f"Within 55 um {pct_mixed_within_55um:.1f}% < {min_pct_mixed_within_55um:.1f}%"
                )
            if not adjacency_pass:
                failures.append("Agent marked adjacency_pass as false")
            lines.append(f"\nFailure: {'; '.join(failures)}")

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning="\n".join(lines),
            agent_answer=agent_answer,
            score=1.0 if passed else 0.0,
        )
