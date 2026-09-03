import json
import os
import subprocess
import uuid

from . import grade_entry as _grade_entry
from .base import BinaryGrader, GraderResult, configuration_error_result
from .numeric import NumericToleranceGrader


_SANDBOX_AGENT_WORKSPACE_DIR = "/agent_workspace"
_SANDBOX_REFERENCE_ARTIFACT_DIR = "/reference_artifact_dir"
_SANDBOX_ENTRYPOINT_PATH = "/grade_entry.py"
_SANDBOX_DOCKER_TIMEOUT_S = 600
_SANDBOX_IMAGE_ENV = "ARTIFACT_GRADER_SANDBOX_IMAGE"
_SANDBOX_WORKSPACE_ENV = "ARTIFACT_GRADER_WORKSPACE_DIR"


class _SandboxError(RuntimeError):
    """The grading sandbox failed to produce a result (infra/timeout)."""


class ArtifactGrader(BinaryGrader):
    """Grade output *files* with a per-field Python function, then tolerance-check."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
       
        err = self._validate_config(agent_answer, config)

        if err is not None:
            return err
        
        sandbox_image = os.environ.get(_SANDBOX_IMAGE_ENV)

        if not sandbox_image:
            raise RuntimeError(f"Set {_SANDBOX_IMAGE_ENV} to a sandbox image (needs Docker)")

        workspace_dir = os.environ.get(_SANDBOX_WORKSPACE_ENV)

        if not workspace_dir:
            raise RuntimeError(f"Set {_SANDBOX_WORKSPACE_ENV} to the agent workspace dir")

        try:
            computed, errors = self._compute_in_container(
                agent_answer, config, sandbox_image, workspace_dir
            )

        except _SandboxError as exc:
            return configuration_error_result(agent_answer, "Artifact Check", str(exc))
        
        return self._band(agent_answer, config, computed, errors)

    def _validate_config(self, agent_answer: dict, config: dict):

        if not isinstance(config, dict):
            return configuration_error_result(agent_answer, "Artifact Check", "config must be an object")
        
        ground_truth = config.get("ground_truth", {})
        artifact_fns = config.get("artifact_fns", {})

        if not isinstance(ground_truth, dict) or len(ground_truth) == 0:
            return configuration_error_result(agent_answer, "Artifact Check", "ground_truth must be a non-empty object")
        
        if not isinstance(artifact_fns, dict):
            return configuration_error_result(agent_answer, "Artifact Check", "artifact_fns must be an object")
        
        for field in ground_truth:

            spec = artifact_fns.get(field)

            if not isinstance(spec, dict) or not _grade_entry.fn_source(spec.get("fn")).strip():
                return configuration_error_result(
                    agent_answer,
                    "Artifact Check",
                    f"artifact_fns[{field!r}].fn must be a non-empty code string or list of lines",
                )
            
        if not isinstance(agent_answer, dict):
            return configuration_error_result(
                agent_answer, "Artifact Check", "agent answer must be an object"
            )
        
        return None

    def _band(self, agent_answer: dict, config: dict, computed: dict, errors: list) -> GraderResult:
        """Trusted (host-side): tolerance-check the computed numbers via
        numeric_tolerance and enrich with paths + any fn errors."""

        ground_truth = config.get("ground_truth", {})
        tolerances = config.get("tolerances", config.get("tolerance", {}))

        result = NumericToleranceGrader().evaluate_answer(
            computed, {"ground_truth": ground_truth, "tolerances": tolerances}
        )

        for field in ground_truth:
            result.metrics[f"{field}_path"] = agent_answer.get(field)
            result.metrics[f"{field}_value"] = computed.get(field)

        if errors:
            result.reasoning = result.reasoning + "\n\nartifact errors:\n  - " + "\n  - ".join(errors)

        result.agent_answer = agent_answer

        return result

    def _reference_artifact_mounts(self, config: dict) -> list[str]:
        """Download reference_artifact_node on the HOST and bind-mount it read-only.

        The sandbox stays --network none; only the already-cached paths go in.
        """
        from latch_eval_tools.harness.utils import download_single_dataset

        node = config.get("reference_artifact_node")
        nodes = node if isinstance(node, list) else ([node] if node else [])

        mounts: list[str] = []

        for uri in nodes:
            cached = download_single_dataset(uri)
            mounts += [
                "-v",
                f"{cached}:{_SANDBOX_REFERENCE_ARTIFACT_DIR}/{cached.name}:ro",
            ]

        return mounts

    def _compute_in_container(self, agent_answer: dict, config: dict, sandbox_image: str, workspace_dir: str) -> tuple[dict, list]:

        container_name = f"artifact-grade-{uuid.uuid4().hex[:12]}"

        reference_mounts = self._reference_artifact_mounts(config)

        sandbox_config = dict(config)
        if reference_mounts:
            sandbox_config["reference_artifact_dir"] = _SANDBOX_REFERENCE_ARTIFACT_DIR

        request = json.dumps(
            {"agent_answer": agent_answer, "config": sandbox_config}
        ).encode()

        sentinel = _grade_entry.RESULT_SENTINEL

        try:
            proc = subprocess.run(
                [
                    "docker", "run", "--rm", "-i", "--network", "none",
                    "--name", container_name,
                    "-v", f"{workspace_dir}:{_SANDBOX_AGENT_WORKSPACE_DIR}:ro",
                    "-v", f"{_grade_entry.__file__}:{_SANDBOX_ENTRYPOINT_PATH}:ro",
                    *reference_mounts,
                    "-w", _SANDBOX_AGENT_WORKSPACE_DIR,
                    sandbox_image, "python", _SANDBOX_ENTRYPOINT_PATH,
                ],
                input=request,
                capture_output=True,
                timeout=_SANDBOX_DOCKER_TIMEOUT_S,
            )

        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            raise _SandboxError("sandbox grading timed out")

        for line in proc.stdout.decode(errors="replace").splitlines():
            if line.startswith(sentinel):
                payload = json.loads(line[len(sentinel):])
                return payload["computed"], payload["errors"]

        stderr = proc.stderr.decode(errors="replace").strip()

        raise _SandboxError(
            f"sandbox grading failed (exit {proc.returncode}): {stderr[:2000]}"
        )
