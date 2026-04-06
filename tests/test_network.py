#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "latch_eval_tools"
HARNESS_ROOT = PACKAGE_ROOT / "harness"


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_network_modules():
    if "latch_eval_tools" not in sys.modules:
        package = types.ModuleType("latch_eval_tools")
        package.__path__ = [str(PACKAGE_ROOT)]
        sys.modules["latch_eval_tools"] = package

    if "latch_eval_tools.harness" not in sys.modules:
        harness_package = types.ModuleType("latch_eval_tools.harness")
        harness_package.__path__ = [str(HARNESS_ROOT)]
        sys.modules["latch_eval_tools.harness"] = harness_package

    config_module = sys.modules.get("latch_eval_tools.harness.config")
    if config_module is None:
        config_module = load_module(
            "latch_eval_tools.harness.config",
            HARNESS_ROOT / "config.py",
        )

    network_module = sys.modules.get("latch_eval_tools.harness.network")
    if network_module is None:
        network_module = load_module(
            "latch_eval_tools.harness.network",
            HARNESS_ROOT / "network.py",
        )

    return config_module, network_module


def create_sandbox_container(image: str, command: list[str]) -> str:
    config_module, network_module = load_network_modules()

    docker_command = [
        "docker",
        "run",
        "-d",
        "--network",
        config_module.NETWORK_SANDBOX_CONFIG.network_name,
        image,
        *command,
    ]
    try:
        with network_module.sandbox_network():
            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to create sandboxed container.\n"
            f"Command: {' '.join(docker_command)}\n"
            f"stdout:\n{(exc.stdout or '').strip() or '<empty>'}\n"
            f"stderr:\n{(exc.stderr or '').strip() or '<empty>'}"
        ) from exc

    container_id = result.stdout.strip()
    print(container_id)
    return container_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a container on the eval sandbox network and print its ID.",
    )
    parser.add_argument(
        "--image",
        default="busybox:1.36.1",
        help="Container image to launch.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run in the container. Use '-- <cmd> ...'. Defaults to 'sleep infinity'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command or ["sleep", "infinity"]
    create_sandbox_container(args.image, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
