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
    agent_answer: Any, answer_field: Any, reason: str
) -> GraderResult:
    return GraderResult(
        passed=False,
        metrics={"error": reason},
        reasoning=f"longest_subsequence (answer_field={answer_field!r}): FAIL\n  - {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={},
    )


class LongestSubsequenceGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        answer_field = config.get("answer_field")
        ground_truth = config.get("ground_truth", [])

        if not isinstance(answer_field, str) or not answer_field:
            return _longest_subsequence_fail(
                agent_answer, answer_field, "'answer_field' must be a non-empty string"
            )

        if not isinstance(ground_truth, list):
            return _longest_subsequence_fail(
                agent_answer, answer_field, "'ground_truth' must be a list of tuples"
            )

        agent_list = (
            agent_answer.get(answer_field) if isinstance(agent_answer, dict) else None
        )
        if not isinstance(agent_list, list):
            return _longest_subsequence_fail(
                agent_answer,
                answer_field,
                f"agent_answer.{answer_field!r} must be a list",
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
        passed = score == 1.0

        return GraderResult(
            passed=passed,
            score=score,
            metrics={
                "lcs_length": lcs_length,
                "gt_length": gt_length,
                "agent_length": agent_length,
                "denominator": denominator,
            },
            reasoning=_format_longest_subsequence(
                answer_field, lcs_length, gt_length, agent_length, score, passed
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
    passed: bool,
) -> str:
    verdict = "PASS" if passed else "FAIL"
    marker = "+" if passed else "x"
    lines = [
        f"longest_subsequence (answer_field={answer_field!r}): {verdict}",
        f"  {marker} LCS={lcs_length} (gt_len={gt_length}, agent_len={agent_length}, score={score:.4f})",
    ]
    return "\n".join(lines)
