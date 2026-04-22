from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Eval(BaseModel):
    id: str
    task: str
    data_node: str | list[str] | None = None
    grader: dict | None = None
    metadata: dict | None = None
    timeout: int | None = None
    download_timeout: int | None = None
    agent_timeout: int | None = None
    notes: str | None = None


# Backward compatibility alias for scbench/spatialbench
TestCase = Eval


class EvalResult(BaseModel):
    eval_id: str
    conversation_history: list[dict] = Field(default_factory=list)
    trajectory: list[dict] = Field(default_factory=list)
    notebook_state: dict = Field(default_factory=dict)
    duration_ms: float = 0.0
    grader_result: dict | None = None
    agent_answer: dict | None = None


# Backward compatibility alias for scbench/spatialbench
TestResult = EvalResult


class GraderSpec(BaseModel):
    """A single grader entry: ``{"type": <str>, "config": <dict>}``.

    Unknown top-level keys are tolerated to match the linter's ``W022``
    "unknown grader field" warning (non-fatal); runtime grading should
    not be stricter than static analysis.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    type: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class EvalGraderSelection(BaseModel):
    """Top-level grader vs graders selection on an eval JSON.

    Enforces mutual exclusivity between ``grader`` and ``graders`` and
    requires ``graders`` to be a non-empty list when present. Per-entry
    deep validation (grader type allowlist, per-type required config
    fields) lives elsewhere — this model only owns the selection-level
    invariants added by the top-level ``graders`` field.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    grader: dict[str, Any] | None = None
    graders: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def _enforce_selection(self) -> Self:
        if self.grader is not None and self.graders is not None:
            raise ValueError(
                "Fields 'grader' and 'graders' are mutually exclusive; specify exactly one"
            )
        if self.graders is not None and len(self.graders) == 0:
            raise ValueError("Field 'graders' must be a non-empty list")
        return self
