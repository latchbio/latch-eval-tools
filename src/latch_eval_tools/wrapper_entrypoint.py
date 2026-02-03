#!/usr/bin/env python3

import asyncio
import os
import signal
import sys
from pathlib import Path


def setup_environment(sandbox_dir: Path, notebook_id: str):
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    root_dir = sandbox_dir / "root"
    root_dir.mkdir(exist_ok=True)

    latch_dir = root_dir / ".latch"
    latch_dir.mkdir(exist_ok=True)

    user_latch_token = Path.home() / ".latch" / "token"
    if not user_latch_token.exists():
        raise RuntimeError("Latch token required at ~/.latch/token")
    token = user_latch_token.read_text().strip()

    (latch_dir / "token").write_text(token)
    (latch_dir / "id").write_text("99999")
    (latch_dir / "notebook-id").write_text(notebook_id)
    (latch_dir / "session-id").write_text("local-eval-session")
    (latch_dir / "nucleus-url").write_text("https://nucleus.latch.bio")

    os.environ.update({
        "DD_VERSION": "local-dev",
        "DD_SERVICE": "latch-plots-eval",
        "DD_ENV": "local",
        "DD_AGENT_HOST": "localhost",
        "DD_TRACE_ENABLED": "false",
        "DD_PROFILING_ENABLED": "false",
        "DD_RUNTIME_METRICS_ENABLED": "false",
        "OTEL_SDK_DISABLED": "true",
        "auth_jwks_url": "https://example.com/jwks",
        "auth_issuer": "local-dev",
        "auth_audience": "local-dev",
        "auth_self_signed_jwk": "{}",
        "auto_reload": "false",
        "logging_mode": "console",
        "domain": "latch.bio",
        "AGENT_DEBUG": "1",
        "LATCH_SANDBOX_ROOT": str(latch_dir),
        "LD_LIBRARY_PATH": f"/root/miniconda3/envs/plots-faas/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
    })

    import pathlib
    original_path_new = pathlib.Path.__new__

    def patched_path_new(cls, *args, **kwargs):
        if args and str(args[0]) == "/root/.latch":
            return original_path_new(cls, str(latch_dir), *args[1:], **kwargs)
        return original_path_new(cls, *args, **kwargs)

    pathlib.Path.__new__ = patched_path_new

    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

    print(f"[wrapper] Using sandbox: {sandbox_dir}", flush=True)
    print(f"[wrapper] Latch dir: {latch_dir}", flush=True)
    print(f"[wrapper] Added {venv_bin} to PATH", flush=True)
    return latch_dir


async def mock_add_pod_event(*, auth, event_type):
    print(f"[wrapper] Pod event (mocked): {event_type}", flush=True)


async def run_server(latch_dir: Path, port: int, notebook_id: str):
    from hypercorn.asyncio import serve
    from hypercorn.config import Config as HypercornConfig
    from latch_asgi.server import LatchASGIServer

    from runtime.mount.endpoints import http_routes, websocket_routes
    from runtime.mount.entrypoint import shutdown
    from runtime.mount import entrypoint, headless_browser

    print(f"[wrapper] Using real notebook {notebook_id}", flush=True)

    entrypoint.add_pod_event = mock_add_pod_event

    entrypoint.plots_ctx_manager.session_owner = "eval-harness"
    original_screenshot = headless_browser.HeadlessBrowser.screenshot

    async def patched_screenshot(self, path: str):
        if path.startswith("/var/log/"):
            path = str(latch_dir / path.split("/")[-1])
        return await original_screenshot(self, path)

    async def patched_start(self, notebook_url, local_storage, *, timeout_ms=30000):
        from playwright.async_api import async_playwright
        import json as json_mod
        from collections.abc import Mapping

        notebook_url = f"https://console.latch.bio/plots/{notebook_id}"
        print(f"[wrapper] Patched notebook URL to: {notebook_url}", flush=True)

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page(viewport={"width": 1280, "height": 800})

        storage = dict(local_storage) if isinstance(local_storage, Mapping) else local_storage
        serialized = json_mod.dumps(storage)
        await self.page.add_init_script(
            f"""
            const entries = JSON.parse({json_mod.dumps(serialized)});
            for (const [k, v] of Object.entries(entries)) {{
                localStorage.setItem(k, v);
            }}
            """
        )

        def should_log_browser_msg(msg):
            text = msg.text
            if "ERR_FAILED" in text or "[Network error]" in text:
                return False
            if "Failed to load resource" in text:
                return False
            return True

        self.page.on("console", lambda msg: print(f"[browser] {msg.type}: {msg.text}", flush=True) if should_log_browser_msg(msg) else None)
        self.page.on("pageerror", lambda err: print(f"[browser-error] Page error: {err}", flush=True))

        async def handle_api_calls(route):
            url = route.request.url

            if "graphql" in url or "vacuole" in url:
                try:
                    post_data = route.request.post_data
                    if post_data:
                        import re
                        query_match = re.search(r'(query|mutation)\s+(\w+)', post_data)
                        query_name = query_match.group(2) if query_match else "unknown"
                        print(f"[browser-gql] GraphQL request: {query_name}", flush=True)

                        fake_pod = {
                            "__typename": "PodInfo",
                            "id": "99999",
                            "status": "RUNNING",
                            "jupyterToken": "eval-token",
                            "cpuMillicores": "4000",
                            "memoryBytes": "8589934592",
                            "gpus": "0",
                            "gpuType": None,
                            "storageGigs": "50",
                            "usedStorageGigs": "0",
                            "internalIpAddress": "127.0.0.1",
                            "autoShutoffDelay": {
                                "__typename": "Interval",
                                "days": 0, "months": 0, "hours": 1,
                                "minutes": 0, "seconds": 0, "years": 0
                            },
                            "deployment": {
                                "__typename": "PodDeployment",
                                "id": "99999",
                                "targetRegion": "us-west-2",
                                "targetDomain": "us-west-2"
                            }
                        }

                        pod_list_queries = ["GetPodStatus", "GetPodInfoFromNotebookId"]
                        if query_name in pod_list_queries:
                            print(f"[browser-gql] Intercepting {query_name} (podInfos), returning fake RUNNING pod", flush=True)
                            fake_pod_response = json_mod.dumps({
                                "data": {
                                    "podInfos": {
                                        "__typename": "PodInfosConnection",
                                        "nodes": [fake_pod]
                                    }
                                }
                            })
                            await route.fulfill(status=200, content_type="application/json", body=fake_pod_response)
                            return

                        if query_name == "PodInfoByPodId":
                            print(f"[browser-gql] Intercepting {query_name} (podInfo), returning fake RUNNING pod", flush=True)
                            fake_pod_response = json_mod.dumps({
                                "data": {
                                    "podInfo": fake_pod
                                }
                            })
                            await route.fulfill(status=200, content_type="application/json", body=fake_pod_response)
                            return
                except Exception as e:
                    print(f"[browser-gql] Error checking GQL query: {e}", flush=True)
                await route.continue_()
            elif "nucleus" in url:
                print(f"[browser-api] Blocking nucleus API call: {url}", flush=True)
                await route.fulfill(status=200, content_type="application/json", body='{"data":{}}')
            else:
                await route.continue_()

        await self.page.route("**/*", handle_api_calls)

        print(f"[wrapper] Headless browser navigating to: {notebook_url}", flush=True)
        await self.page.goto(notebook_url, wait_until="load")

        try:
            await self.page.wait_for_selector("[data-plot-ready='true']", timeout=timeout_ms)
        except Exception:
            await self.screenshot(str(latch_dir / "headless_browser_no_selector.png"))
            raise

        await self.screenshot(str(latch_dir / "headless_browser_ready.png"))
        print("[wrapper] Headless browser page ready", flush=True)

    headless_browser.HeadlessBrowser.screenshot = patched_screenshot
    headless_browser.HeadlessBrowser.start = patched_start

    async def patched_start_agent_proc():
        conn_a = entrypoint.a_proc.conn_a = await entrypoint.SocketIo.from_socket(entrypoint.sock_a)

        entrypoint.async_tasks.append(
            asyncio.create_task(entrypoint.handle_agent_messages(conn_a))
        )

        print("[wrapper] Starting agent subprocess", flush=True)
        entrypoint.a_proc.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            (entrypoint.dir_p / "agent.py"),
            str(entrypoint.sock_agent_fd),
            pass_fds=[entrypoint.sock_agent_fd],
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "LATCH_SANDBOX_ROOT": str(latch_dir),
                "PYTHONUNBUFFERED": "1",
            },
            preexec_fn=lambda: os.nice(5),
        )

        async def stream_output(stream, prefix=""):
            while True:
                line = await stream.readline()
                if not line:
                    break
                print(f"[agent] {prefix}{line.decode().rstrip()}", flush=True)

        asyncio.create_task(stream_output(entrypoint.a_proc.proc.stdout, ""))
        asyncio.create_task(stream_output(entrypoint.a_proc.proc.stderr, "[stderr] "))
        print(f"[wrapper] Agent subprocess started, PID: {entrypoint.a_proc.proc.pid}", flush=True)

    entrypoint.start_agent_proc = patched_start_agent_proc

    latch_server = LatchASGIServer(
        http_routes=http_routes,
        websocket_routes=websocket_routes,
        startup_tasks=[entrypoint.start_kernel_proc(), patched_start_agent_proc()],
        shutdown_tasks=[shutdown()],
    )

    cfg = HypercornConfig()
    cfg.bind = [f"127.0.0.1:{port}"]
    cfg.graceful_timeout = 0.1

    print(f"\n[wrapper] Server starting on port {port}", flush=True)
    print(f"[wrapper] WebSocket: ws://127.0.0.1:{port}/agent", flush=True)
    print(f"[wrapper] HTTP: http://127.0.0.1:{port}/readyz", flush=True)

    shutdown_event = asyncio.Event()

    async def await_shutdown():
        await shutdown_event.wait()

    def shutdown_signal(*args):
        print("\n[wrapper] Shutting down...", flush=True)
        shutdown_event.set()

    signal.signal(signal.SIGINT, shutdown_signal)
    signal.signal(signal.SIGTERM, shutdown_signal)

    try:
        await serve(latch_server.raw_app, cfg, shutdown_trigger=await_shutdown)
    finally:
        await shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox-dir", required=True, help="Sandbox directory path")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--notebook-id", required=True, help="Notebook ID to use")
    args = parser.parse_args()

    sandbox_dir = Path(args.sandbox_dir)
    
    # Get the faas directory from environment or default location
    faas_dir = Path(os.environ.get("LATCH_PLOTS_FAAS_PATH", "/root/latch-plots-faas"))
    mount_dir = faas_dir / "runtime" / "mount"

    sys.path.insert(0, str(faas_dir))
    sys.path.insert(0, str(mount_dir))
    sys.path.insert(0, str(mount_dir / "python_lib"))

    latch_dir = setup_environment(sandbox_dir, args.notebook_id)

    from latch_o11y.o11y import setup as setup_o11y
    setup_o11y()

    try:
        asyncio.run(run_server(latch_dir, args.port, args.notebook_id))
    except KeyboardInterrupt:
        print("\n[wrapper] Stopped", flush=True)


if __name__ == "__main__":
    main()
