from .base import BinaryGrader, GraderResult, get_nested_value


class NumericToleranceGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        ground_truth = config.get("ground_truth", {})
        tolerances = config.get("tolerances", config.get("tolerance", {}))

        metrics = {}
        all_pass = True
        failures = []
        field_scores = []

        for field, expected_value in ground_truth.items():
            actual_value, found = get_nested_value(agent_answer, field)
            if not found:
                all_pass = False
                failures.append(f"Missing field: {field}")
                metrics[f"{field}_score"] = 0.0
                field_scores.append(0.0)
                continue

            if isinstance(actual_value, str):
                try:
                    actual_value = float(actual_value)
                except ValueError:
                    all_pass = False
                    failures.append(f"{field}: cannot parse '{actual_value}' as number")
                    metrics[f"{field}_score"] = 0.0
                    field_scores.append(0.0)
                    continue

            if isinstance(actual_value, bool):
                actual_value = int(actual_value)

            if actual_value is None:
                all_pass = False
                failures.append(f"{field}: got null/None value")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_expected"] = expected_value
                metrics[f"{field}_error"] = float('inf')
                metrics[f"{field}_pass"] = False
                metrics[f"{field}_score"] = 0.0
                field_scores.append(0.0)
                continue

            tolerance_config = tolerances.get(field, {"type": "absolute", "value": 0})
            if isinstance(tolerances, dict) and "type" in tolerances and "value" not in tolerances.get(field, {}):
                tolerance_config = tolerances
            tolerance_type = tolerance_config.get("type", "absolute")
            has_asymmetric = "lower" in tolerance_config and "upper" in tolerance_config
            tolerance_value = tolerance_config.get("value", 0)
            tolerance_lower = tolerance_config.get("lower", tolerance_value)
            tolerance_upper = tolerance_config.get("upper", tolerance_value)

            try:
                if tolerance_type == "absolute":
                    if has_asymmetric:
                        within_tolerance = (expected_value - tolerance_lower) <= actual_value <= (expected_value + tolerance_upper)
                        error = abs(actual_value - expected_value)
                        directional_tolerance = tolerance_upper if actual_value >= expected_value else tolerance_lower
                        field_score = self.score_with_tolerance(error, directional_tolerance)
                    else:
                        within_tolerance = abs(actual_value - expected_value) <= tolerance_value
                        error = abs(actual_value - expected_value)
                        field_score = self.score_with_tolerance(error, tolerance_value)
                elif tolerance_type == "relative":
                    relative_error = abs(actual_value - expected_value) / abs(expected_value) if expected_value != 0 else float('inf')
                    within_tolerance = relative_error <= tolerance_value
                    error = relative_error
                    field_score = self.score_with_tolerance(relative_error, tolerance_value)
                elif tolerance_type == "min":
                    threshold = tolerance_value
                    within_tolerance = actual_value >= threshold
                    error = threshold - actual_value if actual_value < threshold else 0
                    if threshold > 0:
                        field_score = self.clamp_score(actual_value / threshold)
                    else:
                        field_score = 1.0 if within_tolerance else 0.0
                elif tolerance_type == "max":
                    threshold = tolerance_value
                    within_tolerance = actual_value <= threshold
                    error = actual_value - threshold if actual_value > threshold else 0
                    if threshold > 0:
                        field_score = self.clamp_score(threshold / actual_value) if actual_value > 0 else 1.0
                    else:
                        field_score = 1.0 if within_tolerance else 0.0
                else:
                    within_tolerance = False
                    error = float('inf')
                    field_score = 0.0
            except TypeError:
                all_pass = False
                failures.append(f"{field}: invalid type {type(actual_value).__name__}, expected numeric")
                metrics[f"{field}_actual"] = actual_value
                metrics[f"{field}_expected"] = expected_value
                metrics[f"{field}_error"] = float('inf')
                metrics[f"{field}_pass"] = False
                metrics[f"{field}_score"] = 0.0
                field_scores.append(0.0)
                continue

            metrics[f"{field}_actual"] = actual_value
            metrics[f"{field}_expected"] = expected_value
            metrics[f"{field}_error"] = error
            metrics[f"{field}_pass"] = within_tolerance
            metrics[f"{field}_score"] = self.clamp_score(field_score)
            field_scores.append(self.clamp_score(field_score))

            if not within_tolerance:
                all_pass = False
                if tolerance_type == "min":
                    failures.append(f"{field}: {actual_value} (minimum required: {tolerance_value})")
                elif tolerance_type == "max":
                    failures.append(f"{field}: {actual_value} (maximum allowed: {tolerance_value})")
                elif has_asymmetric:
                    failures.append(f"{field}: {actual_value} vs {expected_value} (allowed: -{tolerance_lower}/+{tolerance_upper})")
                else:
                    failures.append(f"{field}: {actual_value} vs {expected_value} (error: {error:.2f}, tolerance: {tolerance_value})")

        overall_score = sum(field_scores) / len(field_scores) if field_scores else 0.0
        metrics["score"] = self.clamp_score(overall_score)
        reasoning = self._format_reasoning(ground_truth, tolerances, metrics, failures, all_pass)

        return GraderResult(
            passed=all_pass,
            metrics=metrics,
            reasoning=reasoning,
            agent_answer=agent_answer,
            score=self.clamp_score(overall_score),
        )

    def _format_reasoning(self, ground_truth, tolerances, metrics, failures, passed):
        lines = [f"Numeric Tolerance Check: {'PASS' if passed else 'FAIL'}", f"Overall score: {metrics.get('score', 0.0):.3f}", ""]

        for field in ground_truth.keys():
            if f"{field}_actual" in metrics:
                actual = metrics[f"{field}_actual"]
                expected = metrics[f"{field}_expected"]
                error = metrics[f"{field}_error"]
                field_pass = metrics[f"{field}_pass"]
                field_score = metrics.get(f"{field}_score", 0.0)
                check = "+" if field_pass else "x"
                tolerance_config = tolerances.get(field, {}) if isinstance(tolerances, dict) else {}
                tolerance_type = tolerance_config.get("type", "absolute")
                has_asymmetric = "lower" in tolerance_config and "upper" in tolerance_config
                if tolerance_type == "min":
                    tol_val = tolerance_config.get("value", expected)
                    lines.append(f"  {check} {field}: {actual} (minimum: {tol_val}, score: {field_score:.3f})")
                elif tolerance_type == "max":
                    tol_val = tolerance_config.get("value", expected)
                    lines.append(f"  {check} {field}: {actual} (maximum: {tol_val}, score: {field_score:.3f})")
                elif has_asymmetric:
                    lower = tolerance_config["lower"]
                    upper = tolerance_config["upper"]
                    lines.append(f"  {check} {field}: {actual} vs {expected} (allowed: -{lower}/+{upper}, score: {field_score:.3f})")
                else:
                    lines.append(f"  {check} {field}: {actual} vs {expected} (error: {error:.4f}, score: {field_score:.3f})")

        if not passed and failures:
            lines.extend(["", "Failures:"])
            for failure in failures:
                lines.append(f"  - {failure}")

        return "\n".join(lines)
