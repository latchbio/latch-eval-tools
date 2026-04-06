from .base import BinaryGrader, GraderResult, get_nested_value


class NumericToleranceGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        ground_truth = config.get("ground_truth", {})
        tolerances = config.get("tolerances", config.get("tolerance", {}))

        metrics = {}
        all_pass = True
        failures = []

        for field, expected_value in ground_truth.items():
            actual_value, found = get_nested_value(agent_answer, field)
            if not found:
                all_pass = False
                failures.append(f"Missing field: {field}")
                continue

            if isinstance(actual_value, str):
                try:
                    actual_value = float(actual_value)
                except ValueError:
                    all_pass = False
                    failures.append(f"{field}: cannot parse '{actual_value}' as number")
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
                        error = actual_value - expected_value
                    else:
                        within_tolerance = abs(actual_value - expected_value) <= tolerance_value
                        error = abs(actual_value - expected_value)
                elif tolerance_type == "relative":
                    relative_error = abs(actual_value - expected_value) / abs(expected_value) if expected_value != 0 else float('inf')
                    within_tolerance = relative_error <= tolerance_value
                    error = relative_error
                elif tolerance_type == "min":
                    threshold = tolerance_value
                    within_tolerance = actual_value >= threshold
                    error = threshold - actual_value if actual_value < threshold else 0
                elif tolerance_type == "max":
                    threshold = tolerance_value
                    within_tolerance = actual_value <= threshold
                    error = actual_value - threshold if actual_value > threshold else 0
                else:
                    within_tolerance = False
                    error = float('inf')
            except TypeError:
                all_pass = False
                failures.append(f"{field}: invalid type {type(actual_value).__name__}, expected numeric")
                metrics[f"{field}_actual"] = actual_value
                metrics[f"{field}_expected"] = expected_value
                metrics[f"{field}_error"] = float('inf')
                metrics[f"{field}_pass"] = False
                continue

            metrics[f"{field}_actual"] = actual_value
            metrics[f"{field}_expected"] = expected_value
            metrics[f"{field}_error"] = error
            metrics[f"{field}_pass"] = within_tolerance

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

        reasoning = self._format_reasoning(ground_truth, tolerances, metrics, failures, all_pass)

        total_fields = len(ground_truth)
        fields_passed = sum(1 for field in ground_truth if metrics.get(f"{field}_pass", False))
        score = fields_passed / total_fields if total_fields > 0 else 0.0
        field_scores = {field: float(metrics.get(f"{field}_pass", False)) for field in ground_truth}

        return GraderResult(
            passed=all_pass,
            metrics=metrics,
            reasoning=reasoning,
            agent_answer=agent_answer,
            score=score,
            field_scores=field_scores,
        )

    def _format_reasoning(self, ground_truth, tolerances, metrics, failures, passed):
        lines = [f"Numeric Tolerance Check: {'PASS' if passed else 'FAIL'}", ""]

        for field in ground_truth.keys():
            if f"{field}_actual" in metrics:
                actual = metrics[f"{field}_actual"]
                expected = metrics[f"{field}_expected"]
                error = metrics[f"{field}_error"]
                field_pass = metrics[f"{field}_pass"]
                check = "+" if field_pass else "x"
                tolerance_config = tolerances.get(field, {}) if isinstance(tolerances, dict) else {}
                tolerance_type = tolerance_config.get("type", "absolute")
                has_asymmetric = "lower" in tolerance_config and "upper" in tolerance_config
                if tolerance_type == "min":
                    tol_val = tolerance_config.get("value", expected)
                    lines.append(f"  {check} {field}: {actual} (minimum: {tol_val})")
                elif tolerance_type == "max":
                    tol_val = tolerance_config.get("value", expected)
                    lines.append(f"  {check} {field}: {actual} (maximum: {tol_val})")
                elif has_asymmetric:
                    lower = tolerance_config["lower"]
                    upper = tolerance_config["upper"]
                    lines.append(f"  {check} {field}: {actual} vs {expected} (allowed: -{lower}/+{upper})")
                else:
                    lines.append(f"  {check} {field}: {actual} vs {expected} (error: {error:.4f})")

        if not passed and failures:
            lines.extend(["", "Failures:"])
            for failure in failures:
                lines.append(f"  - {failure}")

        return "\n".join(lines)


class NumericRangeGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        ground_truth = config.get("ground_truth", {})
        ranges = config.get("ranges", {})

        if not isinstance(ground_truth, dict) or not ground_truth:
            reasoning = "\n".join([
                "Numeric Range Check: FAIL",
                "",
                "Failures:",
                "  - No valid ground truth configured",
            ])
            return GraderResult(
                passed=False,
                metrics={},
                reasoning=reasoning,
                agent_answer=agent_answer,
                score=0.0,
                field_scores={},
            )

        if not isinstance(ranges, dict) or not ranges:
            reasoning = "\n".join([
                "Numeric Range Check: FAIL",
                "",
                "Failures:",
                "  - No valid ranges configured",
            ])
            return GraderResult(
                passed=False,
                metrics={},
                reasoning=reasoning,
                agent_answer=agent_answer,
                score=0.0,
                field_scores={},
            )

        metrics = {}
        all_pass = True
        failures = []

        for field, expected_value in ground_truth.items():
            range_config = ranges.get(field)
            minimum = range_config.get("min") if isinstance(range_config, dict) else None
            maximum = range_config.get("max") if isinstance(range_config, dict) else None

            metrics[f"{field}_expected"] = expected_value
            metrics[f"{field}_min"] = minimum
            metrics[f"{field}_max"] = maximum

            if field not in ranges:
                all_pass = False
                failures.append(f"{field}: missing range config")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            if not isinstance(range_config, dict):
                all_pass = False
                failures.append(f"{field}: invalid range config, expected object with 'min' and 'max'")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
                all_pass = False
                failures.append(f"{field}: invalid minimum bound {minimum!r}")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            if not isinstance(maximum, (int, float)) or isinstance(maximum, bool):
                all_pass = False
                failures.append(f"{field}: invalid maximum bound {maximum!r}")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            if not isinstance(expected_value, (int, float)) or isinstance(expected_value, bool):
                all_pass = False
                failures.append(f"{field}: invalid ground truth value {expected_value!r}")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            if minimum >= maximum:
                all_pass = False
                failures.append(f"{field}: invalid open interval ({minimum}, {maximum})")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            if not minimum < expected_value < maximum:
                all_pass = False
                failures.append(
                    f"{field}: ground truth {expected_value} not in open interval ({minimum}, {maximum})"
                )
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            actual_value, found = get_nested_value(agent_answer, field)
            if not found:
                all_pass = False
                failures.append(f"Missing field: {field}")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            if isinstance(actual_value, str):
                try:
                    actual_value = float(actual_value)
                except ValueError:
                    all_pass = False
                    failures.append(f"{field}: cannot parse '{actual_value}' as number")
                    metrics[f"{field}_actual"] = actual_value
                    metrics[f"{field}_pass"] = False
                    continue

            if isinstance(actual_value, bool):
                actual_value = int(actual_value)

            if actual_value is None:
                all_pass = False
                failures.append(f"{field}: got null/None value")
                metrics[f"{field}_actual"] = None
                metrics[f"{field}_pass"] = False
                continue

            try:
                within_range = minimum < actual_value < maximum
            except TypeError:
                all_pass = False
                failures.append(f"{field}: invalid type {type(actual_value).__name__}, expected numeric")
                metrics[f"{field}_actual"] = actual_value
                metrics[f"{field}_pass"] = False
                continue

            metrics[f"{field}_actual"] = actual_value
            metrics[f"{field}_pass"] = within_range

            if not within_range:
                all_pass = False
                failures.append(f"{field}: {actual_value} not in open interval ({minimum}, {maximum})")

        reasoning = self._format_reasoning(ground_truth, ranges, metrics, failures, all_pass)

        total_fields = len(ground_truth)
        fields_passed = sum(1 for field in ground_truth if metrics.get(f"{field}_pass", False))
        score = fields_passed / total_fields if total_fields > 0 else 0.0
        field_scores = {field: float(metrics.get(f"{field}_pass", False)) for field in ground_truth}

        return GraderResult(
            passed=all_pass,
            metrics=metrics,
            reasoning=reasoning,
            agent_answer=agent_answer,
            score=score,
            field_scores=field_scores,
        )

    def _format_reasoning(self, ground_truth, ranges, metrics, failures, passed):
        lines = [f"Numeric Range Check: {'PASS' if passed else 'FAIL'}", ""]

        for field in ground_truth:
            actual = metrics.get(f"{field}_actual")
            expected = metrics.get(f"{field}_expected")
            minimum = metrics.get(f"{field}_min")
            maximum = metrics.get(f"{field}_max")
            field_pass = metrics.get(f"{field}_pass", False)
            check = "+" if field_pass else "x"
            range_config = ranges.get(field)

            if not isinstance(range_config, dict):
                lines.append(f"  {check} {field}: invalid range config")
                continue

            lines.append(
                f"  {check} {field}: {actual} vs {expected} (open interval: ({minimum}, {maximum}))"
            )

        if not passed and failures:
            lines.extend(["", "Failures:"])
            for failure in failures:
                lines.append(f"  - {failure}")

        return "\n".join(lines)
