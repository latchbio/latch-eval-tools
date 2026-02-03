from pydantic import BaseModel, Field


class Eval(BaseModel):
    id: str
    task: str
    data_node: str | list[str] | None = None
    grader: dict | None = None
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
