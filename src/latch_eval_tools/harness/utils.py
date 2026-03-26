import hashlib
import json
import os
import shutil
from importlib.resources import files
from pathlib import Path

from jinja2 import DebugUndefined, Environment
from latch.ldata.path import LPath
import subprocess

DEFAULT_DOCKER_IMAGE = "public.ecr.aws/p5z7v3z8/benchmark_agent:latest"
GIBIBYTE = 1024**3
MEMORY_HEADROOM_BYTES = 2 * GIBIBYTE
MIN_MEMORY_LIMIT_BYTES = 128 * 1024**2
CGROUP_MEMORY_LIMIT_PATHS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)
PROMPT_TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    undefined=DebugUndefined,
)



def ensure_docker_image(image: str) -> None:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
    )
    if result.returncode == 0:
        return
    print(f"Pulling Docker image {image} ...")
    subprocess.run(
        ["docker", "pull", image],
        check=True,
    )
    print(f"Docker image {image} pulled successfully")


def load_trajectory_identifier(trajectory_path: Path, key: str) -> str | None:
    if not trajectory_path.exists():
        print("No trajectory.json file found")
        return None

    try:
        trajectory = json.loads(trajectory_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"Failed to parse trajectory.json: {exc}")
        return None

    if not trajectory:
        print("No trajectory events found")
        return None

    if not isinstance(trajectory, list):
        print("Unexpected trajectory format in trajectory.json")
        return None

    for event in trajectory:
        if not isinstance(event, dict):
            continue
        identifier = event.get(key)
        if identifier:
            return str(identifier)

    print(f"No {key} found in trajectory.json")
    return None


def get_memory_limit_bytes(headroom_bytes: int = MEMORY_HEADROOM_BYTES) -> int:
    host_total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    cgroup_limit_bytes: int | None = None
    for path in CGROUP_MEMORY_LIMIT_PATHS:
        try:
            raw_value = path.read_text().strip()
        except OSError:
            continue

        if not raw_value or raw_value == "max":
            continue

        try:
            limit_bytes = int(raw_value)
        except ValueError:
            continue

        if limit_bytes > 0:
            cgroup_limit_bytes = limit_bytes
            break

    memory_ceiling_bytes = host_total_bytes
    if cgroup_limit_bytes is not None:
        memory_ceiling_bytes = min(memory_ceiling_bytes, cgroup_limit_bytes)

    limit_bytes = memory_ceiling_bytes - headroom_bytes
    if limit_bytes > 0:
        return limit_bytes

    if memory_ceiling_bytes <= MIN_MEMORY_LIMIT_BYTES:
        return memory_ceiling_bytes

    return max(memory_ceiling_bytes - MIN_MEMORY_LIMIT_BYTES, MIN_MEMORY_LIMIT_BYTES)


def render_packaged_prompt(filename: str, **template_values: object) -> str:
    template = PROMPT_TEMPLATE_ENVIRONMENT.from_string(
        read_packaged_prompt(filename)
    )
    return template.render(
        **{key: str(value) for key, value in template_values.items()}
    )


def _inspect_docker_container_state(
    container_name: str,
    state_field: str,
    docker_executable: str = "docker",
) -> str | None:
    result = subprocess.run(
        [
            docker_executable,
            "inspect",
            "--format",
            f"{{{{.State.{state_field}}}}}",
            container_name,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def is_docker_container_running(
    container_name: str,
    docker_executable: str = "docker",
) -> bool:
    state = _inspect_docker_container_state(
        container_name,
        "Running",
        docker_executable=docker_executable,
    )
    return state == "true"


def is_docker_container_oom_killed(
    container_name: str,
    docker_executable: str = "docker",
) -> bool:
    state = _inspect_docker_container_state(
        container_name,
        "OOMKilled",
        docker_executable=docker_executable,
    )
    return state == "true"


def get_project_root():
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_cache_dir(cache_name: str = ".eval_cache"):
    """Get cache directory for datasets.
    
    Args:
        cache_name: Name of cache directory (default: .eval_cache)
                   Can be customized per benchmark (e.g., .scbench, .spatialbench)
    """
    project_root = get_project_root()
    cache_dir = project_root / cache_name / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_manifest(cache_name: str = ".eval_cache"):
    """Load cache manifest tracking downloaded datasets."""
    cache_dir = get_cache_dir(cache_name)
    manifest_file = cache_dir / "manifest.json"

    if manifest_file.exists():
        return json.loads(manifest_file.read_text())
    return {}


def save_cache_manifest(manifest: dict, cache_name: str = ".eval_cache"):
    """Save cache manifest."""
    cache_dir = get_cache_dir(cache_name)
    manifest_file = cache_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))


def get_cache_key(uri: str) -> str:
    """Generate cache key from URI."""
    uri_hash = hashlib.sha256(uri.encode()).hexdigest()[:16]
    filename = LPath(uri).name() or Path(uri).name or "data"
    return f"{uri_hash}__{filename}"


def download_single_dataset(uri: str, show_progress: bool = True, cache_name: str = ".eval_cache") -> Path:
    """Download a single dataset with caching.
    
    Args:
        uri: URI of dataset to download (e.g., latch://...)
        show_progress: Whether to print progress messages
        cache_name: Name of cache directory
    
    Returns:
        Path to cached file
    """
    cache_dir = get_cache_dir(cache_name)
    manifest = get_cache_manifest(cache_name)

    if uri in manifest:
        cached_file = cache_dir / manifest[uri]
        if cached_file.exists():
            if show_progress:
                print(f"Using cached: {Path(uri).name}")
            return cached_file

    remote_name = LPath(uri).name() or Path(uri).name or "data"
    cache_key = hashlib.sha256(uri.encode()).hexdigest()[:16]
    cache_rel_path = str(Path(cache_key) / remote_name)
    cached_file = cache_dir / cache_rel_path

    if show_progress:
        print(f"Downloading: {uri}")
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    LPath(uri).download(dst=cached_file, cache=True)
    if show_progress:
        print(f"Cached as: {cache_rel_path}")

    manifest[uri] = cache_rel_path
    save_cache_manifest(manifest, cache_name)

    return cached_file


def download_data(data_node: str | list[str], work_dir: Path, cache_name: str = ".eval_cache") -> list[dict]:
    """Download data files into the workspace data directory.
    
    Args:
        data_node: Single URI or list of URIs to download
        work_dir: Working directory to stage data files in
        cache_name: Name of cache directory
    
    Returns:
        List of contextual data dicts with file info
    """
    data_nodes = data_node if isinstance(data_node, list) else ([data_node] if data_node else [])
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    contextual_data = []

    for node in data_nodes:
        cached_file = download_single_dataset(node, cache_name=cache_name)
        data_filename = LPath(node).name() or Path(node).name or "data"

        target_file = data_dir / data_filename
        if target_file.is_symlink() or target_file.is_file():
            target_file.unlink()
        elif target_file.is_dir():
            shutil.rmtree(target_file)
        os.symlink(cached_file, target_file)
        print(f"Linked: {data_filename} -> workspace/data")

        contextual_data.append({
            "type": "File",
            "local_path": f"data/{data_filename}",
        })

    return contextual_data


def resolve_data_mounts(work_dir: Path, container_data_dir: str = "/workspace/data") -> list[str]:
    data_dir = work_dir / "data"
    if not data_dir.is_dir():
        return []

    seen_cache_dirs: set[Path] = set()
    mounts: list[str] = []

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_symlink():
            continue
        real_path = entry.resolve(strict=True)
        cache_dir = real_path.parent
        container_cache_dir = f"/workspace/.data_cache/{cache_dir.name}"

        entry.unlink()
        entry.symlink_to(f"{container_cache_dir}/{real_path.name}")

        if cache_dir not in seen_cache_dirs:
            seen_cache_dirs.add(cache_dir)
            mounts.extend(["-v", f"{cache_dir}:{container_cache_dir}:ro"])

    return mounts


def batch_download_datasets(uris: list[str], show_progress: bool = True, cache_name: str = ".eval_cache"):
    """Download multiple datasets in batch.
    
    Args:
        uris: List of URIs to download
        show_progress: Whether to print progress messages
        cache_name: Name of cache directory
    """
    if show_progress and uris:
        print(f"Preparing to download {len(uris)} unique dataset(s)...")
        print("=" * 80)

    for i, uri in enumerate(uris, 1):
        if show_progress:
            print(f"[{i}/{len(uris)}] ", end="")
        download_single_dataset(uri, show_progress=show_progress, cache_name=cache_name)

    if show_progress and uris:
        print("=" * 80)
        print(f"Downloaded/verified {len(uris)} dataset(s)")
        print()


def setup_workspace(eval_id: str, run_id: str | None = None, workspace_name: str = ".eval_workspace") -> Path:
    """Setup workspace directory for evaluation.
    
    Args:
        eval_id: ID of evaluation
        run_id: Optional run ID for organizing multiple runs
        workspace_name: Name of workspace directory (default: .eval_workspace)
                       Can be customized per benchmark (e.g., .scbench, .spatialbench)
    
    Returns:
        Path to workspace directory
    """
    project_root = get_project_root()
    if run_id:
        work_dir = project_root / workspace_name / "workspace" / run_id / eval_id
    else:
        work_dir = project_root / workspace_name / "workspace" / eval_id

    if work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    return work_dir


def cleanup_workspace(work_dir: Path, keep: bool = False):
    """Clean up workspace directory.
    
    Args:
        work_dir: Workspace directory to clean up
        keep: Whether to keep the workspace
    """
    if keep:
        print(f"Workspace preserved at: {work_dir}")
    else:
        import shutil
        shutil.rmtree(work_dir)
        print(f"Workspace deleted: {work_dir}")



def load_data_instructions() -> str:
    return read_packaged_prompt("data_instructions.md")


def read_packaged_prompt(filename: str) -> str:
    return files("latch_eval_tools").joinpath("prompts", filename).read_text(encoding="utf-8")
