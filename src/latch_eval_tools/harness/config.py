from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import platform

def enable_sandbox() -> bool:
    sys_os = platform.system()
    if sys_os == "Linux":
        return True
    print(f"Cannot use network sandbox on {sys_os} as it depends on host nftables. RUNNING IN UNSANDBOXED MODE.")
    return False


@dataclass(frozen=True)
class NetworkSandboxConfig:
    network_name: str
    nft_command: tuple[str, ...]
    nft_table_family: str
    nft_table_name: str
    nft_allowed_ipv4_set: str
    nft_allowed_ipv6_set: str
    nft_forward_chain: str
    nft_log_prefix: str
    allowed_tcp_ports: tuple[int, ...]
    refresh_interval_seconds: int
    enabled: bool
    log_drops: bool
    state_dir: Path
    state_file: Path
    state_lock_file: Path

STATE_DIR = Path(tempfile.gettempdir()) / "latch_eval_tools"


NETWORK_SANDBOX_CONFIG = NetworkSandboxConfig(
    network_name="eval-sandbox",
    nft_command=("nft",),
    nft_table_family="inet",
    nft_table_name="eval_sandbox",
    nft_allowed_ipv4_set="allowed_ipv4",
    nft_allowed_ipv6_set="allowed_ipv6",
    nft_forward_chain="forward",
    nft_log_prefix="eval-sandbox-drop: ",
    allowed_tcp_ports=(80, 443),
    refresh_interval_seconds=300,
    enabled=enable_sandbox(),
    log_drops=True,
    state_dir=STATE_DIR,
    state_file=STATE_DIR / "network_sandbox_state.json",
    state_lock_file=STATE_DIR / "network_sandbox_state.lock",
)
