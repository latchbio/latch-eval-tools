import hashlib
import json
import os
import subprocess
from pathlib import Path


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
    filename = Path(uri).name
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

    cache_key = get_cache_key(uri)
    cached_file = cache_dir / cache_key

    if show_progress:
        print(f"Downloading: {uri}")
    subprocess.run(
        ["latch", "cp", uri, str(cached_file)],
        check=True,
        capture_output=True
    )
    if show_progress:
        print(f"Cached as: {cache_key}")

    manifest[uri] = cache_key
    save_cache_manifest(manifest, cache_name)

    return cached_file


def download_data(data_node: str | list[str], work_dir: Path, cache_name: str = ".eval_cache") -> list[dict]:
    """Download and symlink data files to workspace.
    
    Args:
        data_node: Single URI or list of URIs to download
        work_dir: Working directory to create symlinks in
        cache_name: Name of cache directory
    
    Returns:
        List of contextual data dicts with file info
    """
    data_nodes = data_node if isinstance(data_node, list) else ([data_node] if data_node else [])

    contextual_data = []

    for node in data_nodes:
        cached_file = download_single_dataset(node, cache_name=cache_name)
        data_filename = Path(node).name

        target_file = work_dir / data_filename
        if target_file.exists():
            target_file.unlink()
        os.symlink(cached_file, target_file)
        print(f"Linked: {data_filename} -> workspace")

        contextual_data.append({
            "type": "File",
            "path": node,
            "local_path": data_filename,
            "id": node.replace("latch:///", "").replace(".csv", "").replace(".h5ad", ""),
        })

    return contextual_data


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
