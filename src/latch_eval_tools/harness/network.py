from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from importlib.resources import files
import ipaddress
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid

import yaml

from .config import NETWORK_SANDBOX_CONFIG


@dataclass(frozen=True)
class NetworkAllowlist:
    provider_apis: tuple[str, ...]
    annotation_services: tuple[str, ...]
    package_managers: tuple[str, ...]

    @property
    def domains(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for domain in (
            *self.provider_apis,
            *self.annotation_services,
            *self.package_managers,
        ):
            if domain not in seen:
                seen.add(domain)
                ordered.append(domain)
        return tuple(ordered)


def load_network_allowlist() -> NetworkAllowlist:
    resource = files("latch_eval_tools").joinpath("prompts").joinpath(
        "network_allowlist.yaml"
    )
    raw_config = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    return NetworkAllowlist(
        provider_apis=tuple(raw_config.get("provider_apis", [])),
        annotation_services=tuple(raw_config.get("annotation_services", [])),
        package_managers=tuple(raw_config.get("package_managers", [])),
    )


def resolve_all_ips(domain: str) -> tuple[set[str], set[str]]:
    try:
        records = socket.getaddrinfo(
            domain,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise RuntimeError(f"Failed to resolve {domain}: {exc}") from exc

    ipv4_addresses: set[str] = set()
    ipv6_addresses: set[str] = set()
    for family, _, _, _, sockaddr in records:
        address = str(sockaddr[0])
        if family == socket.AF_INET:
            ipv4_addresses.add(address)
        elif family == socket.AF_INET6:
            ipv6_addresses.add(address)

    if not ipv4_addresses and not ipv6_addresses:
        raise RuntimeError(f"DNS returned no A/AAAA records for {domain}")

    return ipv4_addresses, ipv6_addresses


def resolve_allowlist(domains: tuple[str, ...]) -> tuple[set[str], set[str]]:
    all_ipv4: set[str] = set()
    all_ipv6: set[str] = set()
    for domain in domains:
        ipv4_addresses, ipv6_addresses = resolve_all_ips(domain)
        resolved = sorted(
            (*ipv4_addresses, *ipv6_addresses),
            key=address_sort_key,
        )
        print(f"[sandbox] {domain} -> {', '.join(resolved)}")
        all_ipv4.update(ipv4_addresses)
        all_ipv6.update(ipv6_addresses)

    return all_ipv4, all_ipv6


def run_nft_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(NETWORK_SANDBOX_CONFIG.nft_command) + command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nftables command not found. Install nftables or update "
            "`NETWORK_SANDBOX_CONFIG.nft_command` in `harness/config.py`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise RuntimeError(
            "nftables command failed.\n"
            f"Command: {shlex.join(exc.cmd)}\n"
            f"stdout:\n{stdout or '<empty>'}\n"
            f"stderr:\n{stderr or '<empty>'}"
        ) from exc


def nft_table_exists() -> bool:
    try:
        result = subprocess.run(
            list(NETWORK_SANDBOX_CONFIG.nft_command)
            + [
                "list",
                "table",
                NETWORK_SANDBOX_CONFIG.nft_table_family,
                NETWORK_SANDBOX_CONFIG.nft_table_name,
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    if result.returncode == 0:
        return True
    stderr = (result.stderr or "").strip().lower()
    if "no such file or directory" in stderr or "not found" in stderr:
        return False
    raise RuntimeError(
        "Failed to inspect nftables table.\n"
        f"Command: {shlex.join(list(NETWORK_SANDBOX_CONFIG.nft_command) + ['list', 'table', NETWORK_SANDBOX_CONFIG.nft_table_family, NETWORK_SANDBOX_CONFIG.nft_table_name])}\n"
        f"stdout:\n{(result.stdout or '').strip() or '<empty>'}\n"
        f"stderr:\n{(result.stderr or '').strip() or '<empty>'}"
    )


def format_nft_elements(addresses: set[str]) -> str:
    return ", ".join(
        str(ipaddress.ip_address(address))
        for address in sorted(addresses, key=lambda value: ipaddress.ip_address(value))
    )


def address_sort_key(address: str) -> tuple[int, int]:
    parsed = ipaddress.ip_address(address)
    return parsed.version, int(parsed)


def build_nftables_ruleset(
    bridge_iface: str,
    ipv4_addresses: set[str],
    ipv6_addresses: set[str],
    *,
    log_drops: bool,
) -> str:
    bridge_literal = json.dumps(bridge_iface)
    tcp_ports = ", ".join(str(port) for port in NETWORK_SANDBOX_CONFIG.allowed_tcp_ports)
    lines = [
        f"add table {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name}",
        (
            f"add set {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_allowed_ipv4_set} "
            "{ type ipv4_addr; }"
        ),
        (
            f"add set {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_allowed_ipv6_set} "
            "{ type ipv6_addr; }"
        ),
        (
            f"add chain {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
            "{ type filter hook forward priority 0; policy accept; }"
        ),
    ]

    if ipv4_addresses:
        lines.append(
            f"add element {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_allowed_ipv4_set} "
            f"{{ {format_nft_elements(ipv4_addresses)} }}"
        )
    if ipv6_addresses:
        lines.append(
            f"add element {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_allowed_ipv6_set} "
            f"{{ {format_nft_elements(ipv6_addresses)} }}"
        )

    lines.extend(
        [
            (
                f"add rule {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
                f"iifname != {bridge_literal} accept"
            ),
            (
                f"add rule {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
                "ct state established,related accept"
            ),
            (
                f"add rule {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
                f"iifname {bridge_literal} udp dport 53 ip daddr 127.0.0.11 accept"
            ),
            (
                f"add rule {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
                f"iifname {bridge_literal} tcp dport 53 ip daddr 127.0.0.11 accept"
            ),
            (
                f"add rule {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
                f"iifname {bridge_literal} tcp dport {{ {tcp_ports} }} "
                f"ip daddr @{NETWORK_SANDBOX_CONFIG.nft_allowed_ipv4_set} accept"
            ),
            (
                f"add rule {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
                f"iifname {bridge_literal} tcp dport {{ {tcp_ports} }} "
                f"ip6 daddr @{NETWORK_SANDBOX_CONFIG.nft_allowed_ipv6_set} accept"
            ),
        ]
    )

    if log_drops:
        lines.append(
            f"add rule {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
            f"iifname {bridge_literal} counter log prefix {json.dumps(NETWORK_SANDBOX_CONFIG.nft_log_prefix)} drop"
        )
    else:
        lines.append(
            f"add rule {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_forward_chain} "
            f"iifname {bridge_literal} counter drop"
        )

    return "\n".join(lines) + "\n"


def build_refresh_script(
    ipv4_addresses: set[str],
    ipv6_addresses: set[str],
) -> str:
    lines = [
        f"flush set {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_allowed_ipv4_set}",
        f"flush set {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_allowed_ipv6_set}",
    ]
    if ipv4_addresses:
        lines.append(
            f"add element {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_allowed_ipv4_set} "
            f"{{ {format_nft_elements(ipv4_addresses)} }}"
        )
    if ipv6_addresses:
        lines.append(
            f"add element {NETWORK_SANDBOX_CONFIG.nft_table_family} {NETWORK_SANDBOX_CONFIG.nft_table_name} {NETWORK_SANDBOX_CONFIG.nft_allowed_ipv6_set} "
            f"{{ {format_nft_elements(ipv6_addresses)} }}"
        )
    return "\n".join(lines) + "\n"


def run_nft_script(script: str) -> None:
    NETWORK_SANDBOX_CONFIG.state_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="latch-eval-nft-",
        suffix=".nft",
        dir=NETWORK_SANDBOX_CONFIG.state_dir,
        delete=False,
    ) as handle:
        handle.write(script)
        script_path = Path(handle.name)

    try:
        run_nft_command(["-f", str(script_path)])
    finally:
        script_path.unlink(missing_ok=True)


def inspect_docker_network(
    network_name: str = NETWORK_SANDBOX_CONFIG.network_name,
) -> dict | None:
    result = subprocess.run(
        ["docker", "network", "inspect", network_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "no such network" in stderr or "not found" in stderr:
            return None
        raise RuntimeError(
            "Failed to inspect Docker network.\n"
            f"Command: docker network inspect {network_name}\n"
            f"stdout:\n{(result.stdout or '').strip() or '<empty>'}\n"
            f"stderr:\n{(result.stderr or '').strip() or '<empty>'}"
        )

    network_data = json.loads(result.stdout)
    if not network_data:
        return None
    return network_data[0]


def ensure_sandbox_network(
    network_name: str = NETWORK_SANDBOX_CONFIG.network_name,
) -> dict:
    network = inspect_docker_network(network_name)
    if network is not None:
        return network

    create_result = subprocess.run(
        ["docker", "network", "create", "--driver", "bridge", network_name],
        capture_output=True,
        text=True,
    )
    if create_result.returncode != 0:
        stderr = (create_result.stderr or "").strip().lower()
        if "already exists" not in stderr:
            raise RuntimeError(
                "Failed to create Docker sandbox network.\n"
                f"Command: docker network create --driver bridge {network_name}\n"
                f"stdout:\n{(create_result.stdout or '').strip() or '<empty>'}\n"
                f"stderr:\n{(create_result.stderr or '').strip() or '<empty>'}"
            )

    network = inspect_docker_network(network_name)
    if network is None:
        raise RuntimeError(f"Docker network {network_name} was not found after creation")
    return network


def get_bridge_interface(
    network_name: str = NETWORK_SANDBOX_CONFIG.network_name,
) -> str:
    network = ensure_sandbox_network(network_name)
    options = network.get("Options") or {}
    bridge_name = options.get("com.docker.network.bridge.name")
    if bridge_name:
        return bridge_name

    network_id = network.get("Id")
    if not network_id:
        raise RuntimeError(
            f"Docker network {network_name} did not include an Id for bridge discovery"
        )
    return f"br-{network_id[:12]}"


def network_has_attached_containers(
    network_name: str = NETWORK_SANDBOX_CONFIG.network_name,
) -> bool:
    network = inspect_docker_network(network_name)
    if network is None:
        return False
    return bool(network.get("Containers"))


def apply_nftables_rules(
    bridge_iface: str,
    ipv4_addresses: set[str],
    ipv6_addresses: set[str],
    *,
    log_drops: bool,
) -> None:
    if nft_table_exists():
        run_nft_command(
            [
                "delete",
                "table",
                NETWORK_SANDBOX_CONFIG.nft_table_family,
                NETWORK_SANDBOX_CONFIG.nft_table_name,
            ],
        )
    ruleset = build_nftables_ruleset(
        bridge_iface,
        ipv4_addresses,
        ipv6_addresses,
        log_drops=log_drops,
    )
    run_nft_script(ruleset)


def teardown_nftables_rules() -> None:
    if not nft_table_exists():
        return
    run_nft_command(
        [
            "delete",
            "table",
            NETWORK_SANDBOX_CONFIG.nft_table_family,
            NETWORK_SANDBOX_CONFIG.nft_table_name,
        ],
    )


def refresh_ips(allowlist: NetworkAllowlist) -> None:
    if not nft_table_exists():
        return
    ipv4_addresses, ipv6_addresses = resolve_allowlist(allowlist.domains)
    run_nft_script(
        build_refresh_script(
            ipv4_addresses,
            ipv6_addresses,
        )
    )

def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_process_start_ticks(pid: int) -> int | None:
    if pid <= 0:
        return None
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        _, suffix = stat_text.rsplit(") ", maxsplit=1)
        fields = suffix.split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def pid_matches_start_ticks(pid: int, expected_start_ticks: int | None) -> bool:
    if not pid_is_running(pid):
        return False
    if expected_start_ticks is None:
        return True
    return read_process_start_ticks(pid) == expected_start_ticks


def read_state() -> dict:
    if not NETWORK_SANDBOX_CONFIG.state_file.exists():
        return {"holders": {}}
    try:
        state = json.loads(NETWORK_SANDBOX_CONFIG.state_file.read_text())
        assert isinstance(state, dict), "Expected dict state"
        holders = state["holders"]
        assert isinstance(holders, dict), "Expected dict holders"
        return state
    except Exception:
        return {"holders": {}}


def write_state(state: dict) -> None:
    NETWORK_SANDBOX_CONFIG.state_dir.mkdir(parents=True, exist_ok=True)
    NETWORK_SANDBOX_CONFIG.state_file.write_text(json.dumps(state, indent=2))


def prune_stale_holders(state: dict) -> dict:
    holders = state["holders"]
    active_holders: dict[str, dict] = {}
    for token, info in holders.items():
        if not isinstance(info, dict):
            continue
        pid = info.get("pid")
        start_ticks = info.get("process_start_ticks")
        if isinstance(pid, int) and (
            start_ticks is None or isinstance(start_ticks, int)
        ) and pid_matches_start_ticks(pid, start_ticks):
            active_holders[token] = info
    return {"holders": active_holders}


@contextmanager
def state_lock():
    NETWORK_SANDBOX_CONFIG.state_dir.mkdir(parents=True, exist_ok=True)
    with NETWORK_SANDBOX_CONFIG.state_lock_file.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = prune_stale_holders(read_state())
        try:
            yield state
        finally:
            write_state(state)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def refresh_loop(
    stop_event: threading.Event,
    token: str,
    allowlist: NetworkAllowlist,
    refresh_interval_seconds: int,
) -> None:
    while not stop_event.wait(refresh_interval_seconds):
        try:
            with state_lock() as state:
                holders = state.get("holders") or {}
                if token not in holders:
                    return
                refresh_ips(allowlist)
        except Exception as exc:
            print(f"[sandbox] Failed to refresh allowlist IPs: {exc}")


@contextmanager
def sandbox_network(
    refresh_interval_seconds: int = NETWORK_SANDBOX_CONFIG.refresh_interval_seconds,
):
    if not NETWORK_SANDBOX_CONFIG.enabled:
        yield
        return

    assert platform.system() == "Linux", "Expected Linux host for network sandboxing"
    nft_binary = NETWORK_SANDBOX_CONFIG.nft_command[0]
    if shutil.which(nft_binary) is None:
        raise RuntimeError(
            f"Container network sandboxing requires {nft_binary!r} to be installed."
        )

    token = uuid.uuid4().hex
    stop_event = threading.Event()
    refresh_thread: threading.Thread | None = None
    allowlist = load_network_allowlist()

    with state_lock() as state:
        ensure_sandbox_network(NETWORK_SANDBOX_CONFIG.network_name)
        bridge_iface = get_bridge_interface(NETWORK_SANDBOX_CONFIG.network_name)
        ipv4_addresses, ipv6_addresses = resolve_allowlist(allowlist.domains)
        apply_nftables_rules(
            bridge_iface,
            ipv4_addresses,
            ipv6_addresses,
            log_drops=NETWORK_SANDBOX_CONFIG.log_drops,
        )
        holders = state.setdefault("holders", {})
        holders[token] = {
            "pid": os.getpid(),
            "process_start_ticks": read_process_start_ticks(os.getpid()),
            "created_at": time.time(),
        }

    if refresh_interval_seconds > 0:
        refresh_thread = threading.Thread(
            target=refresh_loop,
            args=(stop_event, token, allowlist, refresh_interval_seconds),
            daemon=True,
            name="sandbox-network-refresh",
        )
        refresh_thread.start()

    try:
        yield
    finally:
        stop_event.set()
        if refresh_thread is not None:
            refresh_thread.join(timeout=1)

        should_teardown = False
        with state_lock() as state:
            holders = state.setdefault("holders", {})
            holders.pop(token, None)
            should_teardown = not holders and not network_has_attached_containers(
                NETWORK_SANDBOX_CONFIG.network_name
            )

        if should_teardown:
            teardown_nftables_rules()
