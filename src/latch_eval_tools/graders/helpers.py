from pydantic import ValidationError

from latch_eval_tools_types.graders import all as models

from ..types import GraderSpec
from .base import GraderResult


def grade_multiple_graders_single_answer(
    agent_answer: models.AgentAnswer, grader_specs: models.Config
) -> list[GraderResult | None]:
    """Run every grader in ``grader_specs`` against ``agent_answer``.

    ``grader_specs`` is a list of ``{"type": <str>, "config": <dict>}`` entries
    (the same shape used by the top-level ``graders`` field in an eval JSON).
    Each sub-grader receives the full ``agent_answer`` and its own sub-config;
    sub-graders are expected to own disjoint answer fields.

    Returns a list aligned 1:1 with ``grader_specs``. A valid spec yields a
    :class:`GraderResult`; any malformed spec (non-dict, missing ``type``,
    unknown type, non-dict ``config``) yields ``None`` at that index so
    callers can distinguish tooling misconfiguration from a real agent
    pass/fail.
    """
    from . import get_grader  # noqa: PLC0415 -- avoid circular import at module load

    results: list[GraderResult | None] = []

    for spec in grader_specs:
        try:
            parsed = GraderSpec.model_validate(spec)
        except ValidationError:
            results.append(None)
            continue

        try:
            sub_grader = get_grader(parsed.type)
        except ValueError:
            results.append(None)
            continue

        results.append(sub_grader.evaluate_answer(agent_answer, parsed.config))

    return results
