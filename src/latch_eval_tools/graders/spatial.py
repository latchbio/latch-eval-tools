from .base import BinaryGrader, GraderResult


class SpatialAdjacencyGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        scoring = config.get("scoring", {})
        thresholds = scoring.get("pass_thresholds", {})

        max_median_ic_to_pc = thresholds.get("max_median_ic_to_pc_um", 25.0)
        max_p90_ic_to_pc = thresholds.get("max_p90_ic_to_pc_um", 80.0)
        min_pct_within_15um = thresholds.get("min_pct_ic_within_15um", 60.0)
        min_pct_mixed_within_55um = thresholds.get("min_pct_ic_mixed_within_55um", 60.0)

        required_fields = [
            "median_ic_to_pc_um",
            "p90_ic_to_pc_um",
            "pct_ic_within_15um",
            "pct_ic_mixed_within_55um",
            "adjacency_pass"
        ]

        missing = [f for f in required_fields if f not in agent_answer]
        if missing:
            return GraderResult(
                passed=False,
                metrics={"score": 0.0},
                reasoning=f"Agent answer missing required fields: {missing}",
                agent_answer=agent_answer,
                score=0.0,
            )

        median_ic_to_pc = agent_answer["median_ic_to_pc_um"]
        p90_ic_to_pc = agent_answer["p90_ic_to_pc_um"]
        pct_within_15um = agent_answer["pct_ic_within_15um"]
        pct_mixed_within_55um = agent_answer["pct_ic_mixed_within_55um"]
        adjacency_pass = agent_answer["adjacency_pass"]

        median_pass = median_ic_to_pc <= max_median_ic_to_pc
        p90_pass = p90_ic_to_pc <= max_p90_ic_to_pc
        within_15um_pass = pct_within_15um >= min_pct_within_15um
        mixed_55um_pass = pct_mixed_within_55um >= min_pct_mixed_within_55um

        passed = median_pass and p90_pass and within_15um_pass and mixed_55um_pass and adjacency_pass

        median_score = 1.0 if median_ic_to_pc <= max_median_ic_to_pc else self.clamp_score(max_median_ic_to_pc / median_ic_to_pc)
        p90_score = 1.0 if p90_ic_to_pc <= max_p90_ic_to_pc else self.clamp_score(max_p90_ic_to_pc / p90_ic_to_pc)
        within_15um_score = self.clamp_score(pct_within_15um / min_pct_within_15um) if min_pct_within_15um > 0 else (1.0 if within_15um_pass else 0.0)
        mixed_55um_score = self.clamp_score(pct_mixed_within_55um / min_pct_mixed_within_55um) if min_pct_mixed_within_55um > 0 else (1.0 if mixed_55um_pass else 0.0)
        adjacency_score = 1.0 if adjacency_pass else 0.0
        overall_score = self.clamp_score((median_score + p90_score + within_15um_score + mixed_55um_score + adjacency_score) / 5)

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
            "median_score": self.clamp_score(median_score),
            "p90_score": self.clamp_score(p90_score),
            "within_15um_score": self.clamp_score(within_15um_score),
            "mixed_55um_score": self.clamp_score(mixed_55um_score),
            "adjacency_score": adjacency_score,
            "score": overall_score,
        }

        lines = [
            f"Spatial Adjacency Analysis: {'PASS' if passed else 'FAIL'}",
            f"Overall score: {overall_score:.3f}",
            "",
            "IC->PC Distance Metrics:",
            f"  {'+'if median_pass else 'x'} Median distance: {median_ic_to_pc:.2f} um (threshold: <={max_median_ic_to_pc:.2f} um, score: {median_score:.3f})",
            f"  {'+'if p90_pass else 'x'} 90th percentile: {p90_ic_to_pc:.2f} um (threshold: <={max_p90_ic_to_pc:.2f} um, score: {p90_score:.3f})",
            "",
            "IC Proximity to PC:",
            f"  {'+'if within_15um_pass else 'x'} IC within 15 um: {pct_within_15um:.1f}% (threshold: >={min_pct_within_15um:.1f}%, score: {within_15um_score:.3f})",
            f"  {'+'if mixed_55um_pass else 'x'} IC with PC within 55 um: {pct_mixed_within_55um:.1f}% (threshold: >={min_pct_mixed_within_55um:.1f}%, score: {mixed_55um_score:.3f})",
            "",
            f"Agent adjacency assessment: {'+'if adjacency_pass else 'x'} {adjacency_pass} (score: {adjacency_score:.3f})",
        ]

        if not passed:
            failures = []
            if not median_pass:
                failures.append(f"Median {median_ic_to_pc:.2f} > {max_median_ic_to_pc:.2f} um")
            if not p90_pass:
                failures.append(f"P90 {p90_ic_to_pc:.2f} > {max_p90_ic_to_pc:.2f} um")
            if not within_15um_pass:
                failures.append(f"Within 15 um {pct_within_15um:.1f}% < {min_pct_within_15um:.1f}%")
            if not mixed_55um_pass:
                failures.append(f"Within 55 um {pct_mixed_within_55um:.1f}% < {min_pct_mixed_within_55um:.1f}%")
            if not adjacency_pass:
                failures.append("Agent marked adjacency_pass as false")
            lines.append(f"\nFailure: {'; '.join(failures)}")

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning="\n".join(lines),
            agent_answer=agent_answer,
            score=overall_score,
        )
