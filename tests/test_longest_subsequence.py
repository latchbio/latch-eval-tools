"""Reward-surface tests for the ``longest_subsequence`` grader.

``GraderResult.score`` is the reward. The raw LCS ratio is not a safe reward on
a short ranking: two orderings of the same ``n`` elements share an LCS of about
``2*sqrt(n)`` by chance, so an agent that submits the elements in the order it
received them collects a large fraction of the reward for no work - on a
six-element ranking, more than a genuine attempt that gets half the order
right. These tests pin that the reward is gated on ``scoring.pass_threshold``
while the ratio stays available as a diagnostic metric.
"""

from __future__ import annotations

import pytest

from latch_eval_tools.graders.longest_subsequence import LongestSubsequenceGrader

# Ordering by reads surviving human-contamination removal.
GROUND_TRUTH = [
    ["SRR38138952"],
    ["SRR38138951"],
    ["SRR38138954"],
    ["SRR38138955"],
    ["SRR38138953"],
    ["SRR38138956"],
]
# What an agent reports when it does no contaminant removal and just hands back
# the accessions in the order they were delivered. Shares an LCS of 4/6 with the
# ground truth purely by chance.
ZERO_EFFORT = [[accession] for accession in sorted(a[0] for a in GROUND_TRUTH)]
# A real attempt that gets one adjacent pair the wrong way round: 5/6.
ONE_TRANSPOSITION = [
    ["SRR38138952"],
    ["SRR38138951"],
    ["SRR38138955"],
    ["SRR38138954"],
    ["SRR38138953"],
    ["SRR38138956"],
]


def _grade(agent_list: list, pass_threshold: float | None = None) -> object:
    config: dict = {"answer_field": "ranking", "ground_truth": GROUND_TRUTH}
    if pass_threshold is not None:
        config["scoring"] = {"pass_threshold": pass_threshold}
    return LongestSubsequenceGrader().evaluate_answer({"ranking": agent_list}, config)


def test_zero_effort_ordering_pays_no_reward() -> None:
    result = _grade(ZERO_EFFORT, 0.75)

    assert result.metrics["lcs_ratio"] == pytest.approx(4 / 6)
    assert not result.passed
    assert result.score == 0.0
    assert result.field_scores == {"ranking": 0.0}


def test_ordering_that_clears_the_threshold_pays_full_reward() -> None:
    result = _grade(ONE_TRANSPOSITION, 0.75)

    assert result.metrics["lcs_ratio"] == pytest.approx(5 / 6)
    assert result.passed
    assert result.score == 1.0
    assert result.field_scores == {"ranking": 1.0}


def test_exact_ordering_passes() -> None:
    result = _grade(GROUND_TRUTH, 0.75)

    assert result.metrics["lcs_ratio"] == pytest.approx(1.0)
    assert result.passed
    assert result.score == 1.0


def test_default_threshold_requires_the_exact_ordering() -> None:
    partial = _grade(ONE_TRANSPOSITION)

    assert not partial.passed
    assert partial.score == 0.0
    assert _grade(GROUND_TRUTH).score == 1.0


def test_zero_effort_never_outscores_a_closer_ordering() -> None:
    """The floor is what makes the raw ratio unsafe; the gate must remove it."""

    for pass_threshold in (0.0, 0.25, 0.5, 0.6, 0.667, 0.75, 0.9, 1.0):
        zero_effort = _grade(ZERO_EFFORT, pass_threshold)
        genuine = _grade(ONE_TRANSPOSITION, pass_threshold)
        assert zero_effort.score <= genuine.score, pass_threshold


def test_reward_is_binary_in_the_pass_flag() -> None:
    for agent_list in (ZERO_EFFORT, ONE_TRANSPOSITION, GROUND_TRUTH, []):
        result = _grade(agent_list, 0.75)
        assert result.score == (1.0 if result.passed else 0.0)


def test_lcs_ratio_metric_still_reports_partial_progress() -> None:
    """Gating the reward must not cost reviewers the diagnostic signal."""

    ratios = [
        _grade(agent_list, 0.75).metrics["lcs_ratio"]
        for agent_list in (
            list(reversed(GROUND_TRUTH)),
            ZERO_EFFORT,
            ONE_TRANSPOSITION,
            GROUND_TRUTH,
        )
    ]
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(1 / 6)
    assert ratios[-1] == pytest.approx(1.0)
