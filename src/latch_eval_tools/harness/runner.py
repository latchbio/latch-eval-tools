import json
from pathlib import Path

from latch_eval_tools.types import TestCase
from latch_eval_tools.graders import (
    GRADER_REGISTRY,
    GraderResult,
    grade_multiple_graders_single_answer,
)
from latch_eval_tools.harness.utils import (
    download_data,
    get_agent_workspace_dir,
    setup_workspace,
    cleanup_workspace,
)


class EvalRunner:
    """Main evaluation runner for executing benchmarks with various agents."""
    
    def __init__(
        self,
        eval_path: str | Path,
        keep_workspace: bool = False,
        run_id: str | None = None,
        cache_name: str = ".eval_cache",
        workspace_name: str = ".eval_workspace",
        benchmark_name: str = "Eval"
    ):
        """Initialize evaluation runner.
        
        Args:
            eval_path: Path to eval JSON file
            keep_workspace: Whether to preserve workspace after completion
            run_id: Optional run ID for organizing multiple runs
            cache_name: Name of cache directory (e.g., .scbench, .spatialbench)
            workspace_name: Name of workspace directory
            benchmark_name: Display name for benchmark (e.g., "SCBench", "SpatialBench")
        """
        self.eval_path = Path(eval_path)
        self.keep_workspace = keep_workspace
        self.run_id = run_id
        self.cache_name = cache_name
        self.workspace_name = workspace_name
        self.benchmark_name = benchmark_name

        if not self.eval_path.exists():
            raise FileNotFoundError(f"Eval file not found: {self.eval_path}")

        eval_data = json.loads(self.eval_path.read_text())
        self.test_case = TestCase(**eval_data)

    def run(self, agent_function=None):
        """Run evaluation with specified agent function.
        
        Args:
            agent_function: Callable that takes (task_prompt: str, work_dir: Path)
                          and returns dict with keys "answer" and optionally "metadata".
        
        Returns:
            dict with test results including test_id, agent_answer, grader_result, passed
        """
        print("=" * 80)
        print(f"Running {self.benchmark_name} evaluation: {self.test_case.id}")
        print("=" * 80)

        print("\nTask:")
        print("-" * 80)
        print(self.test_case.task)
        print("-" * 80)

        # `TestCase` accepts mutually-exclusive `grader` or `graders`.
        # Validate before workspace setup so malformed cases fail fast.
        if self.test_case.grader is not None:
            grader_specs: list[dict] = [self.test_case.grader]
        elif self.test_case.graders is not None and len(self.test_case.graders) != 0:
            grader_specs = list(self.test_case.graders)
        else:
            raise ValueError(
                f"TestCase {self.test_case.id!r} has no grader or graders defined"
            )

        work_dir = setup_workspace(self.test_case.id, self.run_id, self.workspace_name)
        print(f"\nWorking directory: {work_dir}")

        print("\n" + "=" * 80)
        print("Staging data files...")
        print("=" * 80)

        download_data(self.test_case.data_node or [], work_dir, self.cache_name)

        task_prompt = self.test_case.task

        print("\n" + "=" * 80)
        print("Running agent on task...")
        print("=" * 80)

        agent_answer = None
        agent_metadata = {}

        if agent_function is None:
            print("\nNo agent function provided. To run this eval, pass an agent_function that:")
            print("  Takes (task_prompt: str, work_dir: Path) as arguments")
            print("  Returns dict with 'answer' key containing the parsed JSON answer")
            print("\nExample:")
            print("  def my_agent(task, work_dir):")
            print("      # Run your agent which writes eval_answer.json to /workspace")
            print("      answer_file = work_dir / 'agent_workspace' / 'eval_answer.json'")
            print("      return json.loads(answer_file.read_text())")
            print("\n  runner = EvalRunner(eval_path)")
            print("  runner.run(agent_function=my_agent)")
        else:
            try:
                result = agent_function(task_prompt, work_dir)

                if isinstance(result, dict) and "answer" in result:
                    agent_answer = result["answer"]
                    agent_metadata = result.get("metadata", {})
                else:
                    agent_answer = result

                print("\nAgent completed successfully")
            except Exception as e:
                print(f"\nAgent error: {e}")
                import traceback
                traceback.print_exc()

        eval_answer_path = get_agent_workspace_dir(work_dir) / "eval_answer.json"
        if agent_answer is None and eval_answer_path.exists():
            try:
                agent_answer = json.loads(eval_answer_path.read_text())
                print(f"Loaded agent answer from {eval_answer_path}")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {eval_answer_path}: {e}")

        grader_result: GraderResult | None = None
        if agent_answer is not None:
            print("\n" + "=" * 80)
            print(f"Running grader{'s' if len(grader_specs) > 1 else ''}...")
            print("=" * 80)

            grader_result = (
                _run_single_grader(grader_specs[0], agent_answer)
                if len(grader_specs) == 1
                else _run_multi_grader(grader_specs, agent_answer)
            )

            print(f"\n{'✓ EVAL PASSED' if grader_result.passed else '✗ EVAL FAILED'}")
            print("\nGrader reasoning:")
            print("-" * 80)
            print(grader_result.reasoning)
            print("-" * 80)

            if grader_result.metrics:
                print("\nMetrics:")
                for key, value in grader_result.metrics.items():
                    if isinstance(value, (list, dict)):
                        continue
                    print(f"   {key}: {value}")

        print("\n" + "=" * 80)
        print("Cleanup...")
        print("=" * 80)

        cleanup_workspace(work_dir, keep=self.keep_workspace)

        if self.keep_workspace:
            print("\nTo inspect results:")
            print(f"  cd {work_dir}")

        result_dict = {
            "test_id": self.test_case.id,
            "agent_answer": agent_answer,
            "grader_result": grader_result,
            "passed": grader_result.passed if grader_result else None,
        }

        if agent_metadata:
            result_dict["metadata"] = agent_metadata

        return result_dict


def _run_single_grader(spec: dict, agent_answer: dict) -> GraderResult:
    grader_type = spec.get("type")
    if grader_type not in GRADER_REGISTRY:
        return GraderResult(
            passed=False,
            metrics={"grader_error": f"unknown_grader_type:{grader_type}"},
            reasoning=(
                f"Unknown grader type {grader_type!r}. Available: "
                f"{sorted(GRADER_REGISTRY.keys())}."
            ),
            agent_answer=agent_answer,
        )
    grader = GRADER_REGISTRY[grader_type]()
    try:
        return grader.evaluate_answer(agent_answer, spec.get("config", {}))
    except Exception as e:
        import traceback
        return GraderResult(
            passed=False,
            metrics={"grader_error": str(e)},
            reasoning=f"Grader failed due to malformed agent output: {e}\n\n{traceback.format_exc()}",
            agent_answer=agent_answer,
        )


def _run_multi_grader(specs: list[dict], agent_answer: dict) -> GraderResult:
    try:
        sub_results = grade_multiple_graders_single_answer(agent_answer, specs)
        metrics: dict = {}
        reasoning: list[str] = []
        all_passed = True
        for idx, (spec, sub) in enumerate(zip(specs, sub_results)):
            key = f"{idx}:{spec.get('type') if isinstance(spec, dict) else 'unknown'}"
            if sub is None:
                metrics[f"{key}:error"] = "malformed_or_unknown"
                reasoning.append(f"[{key}] malformed spec or unknown grader type")
                all_passed = False
                continue
            metrics[f"{key}:passed"] = sub.passed
            for mk, mv in (sub.metrics or {}).items():
                metrics[f"{key}:{mk}"] = mv
            reasoning.append(f"[{key}] {sub.reasoning}")
            if not sub.passed:
                all_passed = False
        return GraderResult(
            passed=all_passed,
            metrics=metrics,
            reasoning="\n\n".join(reasoning),
            agent_answer=agent_answer,
        )
    except Exception as e:
        import traceback
        return GraderResult(
            passed=False,
            metrics={"grader_error": str(e)},
            reasoning=f"Multi-grader failed: {e}\n\n{traceback.format_exc()}",
            agent_answer=agent_answer,
        )
