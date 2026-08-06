import math
from typing import Any

from .base import BinaryGrader, GraderResult


def _normalize_element(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _normalize_tuple(value: Any) -> tuple:
    return tuple(_normalize_element(v) for v in value)


def _lcs_length(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[n][m]


def _longest_subsequence_fail(
    agent_answer: Any,
    answer_field: Any,
    reason: str,
    *,
    configuration_error: bool = False,
) -> GraderResult:
    metrics = (
        {"configuration_error": reason} if configuration_error else {"error": reason}
    )
    verdict = "CONFIGURATION ERROR" if configuration_error else "FAIL"
    return GraderResult(
        passed=False,
        metrics=metrics,
        reasoning=(
            f"longest_subsequence (answer_field={answer_field!r}): {verdict}\n"
            f"  - {reason}"
        ),
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={},
    )


def _parse_pass_threshold(
    agent_answer: Any, answer_field: Any, config: dict
) -> GraderResult | float:
    scoring = config.get("scoring", {})
    if not isinstance(scoring, dict):
        return _longest_subsequence_fail(
            agent_answer,
            answer_field,
            "'scoring' must be an object",
            configuration_error=True,
        )

    raw_threshold = scoring.get("pass_threshold", 1.0)
    if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
        return _longest_subsequence_fail(
            agent_answer,
            answer_field,
            "'scoring.pass_threshold' must be a number in [0, 1]",
            configuration_error=True,
        )

    pass_threshold = float(raw_threshold)
    if (
        not math.isfinite(pass_threshold)
        or pass_threshold < 0.0
        or pass_threshold > 1.0
    ):
        return _longest_subsequence_fail(
            agent_answer,
            answer_field,
            "'scoring.pass_threshold' must be a number in [0, 1]",
            configuration_error=True,
        )

    return pass_threshold


class LongestSubsequenceGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return _longest_subsequence_fail(
                agent_answer,
                None,
                "config must be an object",
                configuration_error=True,
            )
        answer_field = config.get("answer_field")
        ground_truth = config.get("ground_truth", [])

        if not isinstance(answer_field, str) or answer_field.strip() == "":
            return _longest_subsequence_fail(
                agent_answer,
                answer_field,
                "'answer_field' must be a non-empty string",
                configuration_error=True,
            )

        if not isinstance(ground_truth, list) or len(ground_truth) == 0:
            return _longest_subsequence_fail(
                agent_answer,
                answer_field,
                "'ground_truth' must be a non-empty list of tuples/lists",
                configuration_error=True,
            )
        if any(not isinstance(item, (list, tuple)) for item in ground_truth):
            return _longest_subsequence_fail(
                agent_answer,
                answer_field,
                "'ground_truth' elements must be tuples/lists",
                configuration_error=True,
            )
        if any(len(item) == 0 for item in ground_truth):
            return _longest_subsequence_fail(
                agent_answer,
                answer_field,
                "'ground_truth' elements must not be empty",
                configuration_error=True,
            )

        pass_threshold = _parse_pass_threshold(agent_answer, answer_field, config)
        if isinstance(pass_threshold, GraderResult):
            return pass_threshold

        agent_list = (
            agent_answer.get(answer_field) if isinstance(agent_answer, dict) else None
        )
        if not isinstance(agent_list, list):
            return _longest_subsequence_fail(
                agent_answer,
                answer_field,
                f"agent_answer.{answer_field!r} must be a list",
            )
        if any(not isinstance(item, (list, tuple)) for item in agent_list):
            return _longest_subsequence_fail(
                agent_answer,
                answer_field,
                f"agent_answer.{answer_field!r} elements must be tuples/lists",
            )

        try:
            normalized_gt = [_normalize_tuple(t) for t in ground_truth]
            normalized_agent = [_normalize_tuple(t) for t in agent_list]
        except TypeError:
            return _longest_subsequence_fail(
                agent_answer,
                answer_field,
                f"agent_answer.{answer_field!r} elements must be tuples/lists",
            )

        lcs_length = _lcs_length(normalized_gt, normalized_agent)
        gt_length = len(normalized_gt)
        agent_length = len(normalized_agent)
        denominator = max(gt_length, agent_length, 1)
        score = lcs_length / denominator
        passed = score >= pass_threshold

        return GraderResult(
            passed=passed,
            score=score,
            metrics={
                "lcs_length": lcs_length,
                "gt_length": gt_length,
                "agent_length": agent_length,
                "denominator": denominator,
                "pass_threshold": pass_threshold,
            },
            reasoning=_format_longest_subsequence(
                answer_field,
                lcs_length,
                gt_length,
                agent_length,
                score,
                pass_threshold,
                passed,
            ),
            agent_answer=agent_answer,
            field_scores={answer_field: score},
        )


def _format_longest_subsequence(
    answer_field: Any,
    lcs_length: int,
    gt_length: int,
    agent_length: int,
    score: float,
    pass_threshold: float,
    passed: bool,
) -> str:
    verdict = "PASS" if passed else "FAIL"
    marker = "+" if passed else "x"
    lines = [
        f"longest_subsequence (answer_field={answer_field!r}): {verdict}",
        (
            f"  {marker} LCS={lcs_length} "
            f"(gt_len={gt_length}, agent_len={agent_length}, "
            f"score={score:.4f}, threshold={pass_threshold:.4f})"
        ),
    ]
    return "\n".join(lines)
