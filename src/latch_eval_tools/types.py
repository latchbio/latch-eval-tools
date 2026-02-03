from pydantic import BaseModel, Field


class Eval(BaseModel):
    """Evaluation test case definition.
    
    Alias: TestCase (for backward compatibility with scbench/spatialbench)
    """
    id: str
    task: str
    data_node: str | list[str] | None = None
    grader: dict | None = None
    timeout: int | None = None
    download_timeout: int | None = None
    agent_timeout: int | None = None
    notes: str | None = None


# Backward compatibility alias
TestCase = Eval


class EvalResult(BaseModel):
    """Result from running an evaluation.
    
    Combines fields from both latch-plots-eval-harness and scbench/spatialbench.
    """
    eval_id: str
    conversation_history: list[dict] = Field(default_factory=list)
    trajectory: list[dict] = Field(default_factory=list)
    notebook_state: dict = Field(default_factory=dict)
    duration_ms: float = 0.0
    grader_result: dict | None = None
    agent_answer: dict | None = None


class GraderResult(BaseModel):
    """Structured result from grading an evaluation."""
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    passed: bool = Field(description="Whether the eval passed")
    reasoning: str = Field(description="Detailed reasoning for the score")
    successes: list[str] = Field(default_factory=list, description="List of things the agent did correctly")
    failures: list[str] = Field(default_factory=list, description="List of things the agent failed to do or did incorrectly")


# Backward compatibility alias
TestResult = EvalResult
