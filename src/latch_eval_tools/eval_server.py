from latch_eval_tools import faas_utils

import argparse
import asyncio
import json
import os
import shutil
import socket
import sys
import textwrap
import time
import uuid
from pathlib import Path

import websockets
import websockets.server
from latch_eval_tools.types import Eval, EvalResult
from latch_eval_tools.graders import GRADER_REGISTRY
from latch_eval_tools.answer_extraction import extract_answer_from_conversation
from latch_eval_tools.headless_eval_server import run_eval_batch_headless

faas_runtime_dir = Path(os.environ.get("LATCH_PLOTS_FAAS_PATH", "/root/latch-plots-faas")) / "runtime" / "mount"
sys.path.insert(0, str(faas_runtime_dir))

from socketio import SocketIo
from utils import gql_query


def get_auth_token() -> str:
    return f"Latch-SDK-Token {(Path.home() / '.latch' / 'token').read_text()}"

class EvalServer:
    sandbox_dir: Path
    current_eval_case: Eval | None
    agent_proc: asyncio.subprocess.Process | None
    agent_sock: socket.socket | None
    agent_conn: SocketIo | None
    websocket: websockets.server.WebSocketServerProtocol | None
    session_id: int | None
    eval_complete: bool

    def __init__(self, sandbox_dir: Path):
        self.sandbox_dir = sandbox_dir
        self.current_eval_case = None
        self.agent_proc = None
        self.agent_sock = None
        self.agent_conn = None
        self.websocket = None
        self.session_id = None
        self.eval_complete = False

    async def start_agent(self):
        print("[eval] Starting agent")

        sock_a, sock_agent = socket.socketpair(family=socket.AF_UNIX)
        sock_a.setblocking(False)
        sock_agent_fd = sock_agent.detach()

        self.agent_sock = sock_a
        self.agent_conn = await SocketIo.from_socket(sock_a)

        agent_path = faas_runtime_dir / "agent.py"

        self.agent_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            str(agent_path),
            str(sock_agent_fd),
            pass_fds=[sock_agent_fd],
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "LATCH_SANDBOX_ROOT": str(self.sandbox_dir),
                "PYTHONUNBUFFERED": "1",
                "AGENT_DEBUG": "1",
            },
            preexec_fn=lambda: os.nice(5),
            limit=1024 * 1024,
        )

        async def stream_output(stream, prefix=""):
            while True:
                try:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode().rstrip()
                    if len(decoded) > 1000:
                        decoded = decoded[:1000] + "... [TRUNCATED]"
                    print(f"[agent stream] {prefix}{decoded}", flush=True)
                except (ValueError, asyncio.LimitOverrunError) as e:
                    if "limit" in str(e).lower():
                        chunk = await stream.read(8192)
                        if not chunk:
                            break
                        print(f"[agent] {prefix}[Large output truncated: {len(chunk)} bytes]", flush=True)
                    else:
                        raise
                except Exception as e:
                    print(f"[agent] {prefix}[Error reading output: {e}]", flush=True)
                    break

        asyncio.create_task(stream_output(self.agent_proc.stdout, ""))
        asyncio.create_task(stream_output(self.agent_proc.stderr, "[stderr] "))

        msg = await self.agent_conn.recv()
        if msg.get("type") == "ready":
            print("[eval] Agent subprocess started and ready")

    async def stop_agent(self):
        if self.agent_proc:
            print("[eval] Stopping agent")
            try:
                self.agent_proc.terminate()
                await asyncio.wait_for(self.agent_proc.wait(), timeout=2)
            except TimeoutError:
                self.agent_proc.kill()
                await self.agent_proc.wait()

        if self.agent_sock:
            try:
                self.agent_sock.close()
            except Exception:
                pass

        self.agent_proc = None
        self.agent_sock = None
        self.agent_conn = None

    def clear_notebook_context(self):
        context_dir = faas_runtime_dir / "agent_config" / "context" / "notebook_context"
        if context_dir.exists():
            for file in context_dir.iterdir():
                if file.is_file() and file.name != ".gitkeep":
                    file.unlink()
            print("[eval] Cleared notebook context files")

    async def initialize_agent_session(self, websocket):
        print("[eval] Waiting for console init to get session_id...")
        init_msg = await websocket.recv()
        console_init = json.loads(init_msg)
        if console_init.get("type") == "init":
            self.session_id = int(console_init.get("session_id"))

        await self.agent_conn.send({"type": "init", "session_id": self.session_id, "eval_mode": True})

        while True:
            msg = await self.agent_conn.recv()
            if msg.get("type") == "agent_status" and msg.get("status") == "ready":
                print("[eval] Agent initialized and ready")
                break
            print(f"[eval] Skipping init message: {msg.get('type')}")

        self.clear_notebook_context()

    async def handle_agent_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "agent_history_updated":
            await self.check_for_completion()

    async def check_for_completion(self):
        history = await self.fetch_full_conversation_history()
        for payload in history:
            if payload.get("type") == "anthropic_message" and payload.get("role") == "assistant":
                content = payload.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "submit_response":
                        tool_input = block.get("input", {})
                        if tool_input.get("next_status") == "done":
                            self.eval_complete = True
                            return

    async def fetch_full_conversation_history(self) -> list[dict]:
        try:
            resp = await gql_query(
                auth=get_auth_token(),
                query="""
                    query AgentHistory($sessionId: BigInt!) {
                        agentHistories(condition: {sessionId: $sessionId, removed: false}, orderBy: ID_ASC) {
                            nodes { id payload }
                        }
                    }
                """,
                variables={"sessionId": str(self.session_id)},
            )
            nodes = resp.get("data", {}).get("agentHistories", {}).get("nodes", [])
            return [node.get("payload", {}) for node in nodes]
        except Exception as e:
            print(f"[eval] Error fetching conversation history: {e}")
            return []

    async def reset_for_next_test(self):
        print("[eval] Clearing agent history for next test...")
        await self.agent_conn.send({"type": "agent_clear_history"})
        self.clear_notebook_context()
        self.eval_complete = False
        self.current_eval_case = None
        print("[eval] Reset complete")

    async def keep_forwarding(self):
        async def forward_agent_to_console():
            while True:
                msg = await self.agent_conn.recv()
                await self.websocket.send(json.dumps(msg))

        async def forward_console_to_agent():
            async for message in self.websocket:
                msg = json.loads(message)
                await self.agent_conn.send(msg)

        forward_task = asyncio.create_task(forward_agent_to_console())
        receive_task = asyncio.create_task(forward_console_to_agent())

        try:
            await asyncio.gather(forward_task, receive_task)
        except asyncio.CancelledError:
            forward_task.cancel()
            receive_task.cancel()

    async def run_eval(self, eval_case: Eval) -> EvalResult:
        print(f"\n{'=' * 70}")
        print(f"Running eval: {eval_case.id}")
        print("=" * 70)

        if not self.websocket:
            raise RuntimeError("websocket must be set before calling run_eval()")
        if not self.agent_proc:
            raise RuntimeError("agent must be started before calling run_eval()")

        start_time = time.time()

        self.current_eval_case = eval_case
        self.eval_complete = False

        data_context = ""
        if eval_case.data_node:
            data_nodes = eval_case.data_node if isinstance(eval_case.data_node, list) else [eval_case.data_node]
            contextual_data = []
            for node in data_nodes:
                contextual_data.append({
                    "type": "File",
                    "path": node,
                    "id": node.replace("latch:///", "").replace(".csv", "").replace(".h5ad", ""),
                })
            data_context = f"\n\nHere is the context of the selected nodes the user would like to use: <ContextualNodeData>{json.dumps(contextual_data)}</ContextualNodeData>"

        initial_query = textwrap.dedent(f"""
            {eval_case.task}

            IMPORTANT: When you have completed this task:
            1. Include your answer in your submit_response summary wrapped in <EVAL_ANSWER></EVAL_ANSWER> tags
            2. The content should be ONLY the JSON object with the required fields

            Example format for your submit_response summary:
            <EVAL_ANSWER>
            {{"field1": value1, "field2": value2}}
            </EVAL_ANSWER>

            CRITICAL: 
            - Do NOT use markdown code fences (```json) inside the EVAL_ANSWER tags - use raw JSON only
            - Put the answer directly in your submit_response tool call summary
            - The answer extraction relies on finding <EVAL_ANSWER> tags in your submit_response summary
            {data_context}
        """).strip()

        async def forward_agent_to_console():
            try:
                while True:
                    msg = await self.agent_conn.recv()
                    msg_type = msg.get("type", "unknown")
                    if msg_type != "agent_stream_delta":
                        print(f"[eval] agent→console: {msg_type}")
                    await self.handle_agent_message(msg)
                    await self.websocket.send(json.dumps(msg))
            except Exception as e:
                print(f"[eval] Agent forwarding ended: {e}")

        async def forward_console_to_agent():
            try:
                async for message in self.websocket:
                    msg = json.loads(message)
                    msg_type = msg.get("type")
                    print(f"[eval] console→agent: {msg_type}")
                    await self.agent_conn.send(msg)
            except Exception as e:
                print(f"[eval] Console forwarding ended: {e}")

        forward_task = asyncio.create_task(forward_agent_to_console())
        receive_task = asyncio.create_task(forward_console_to_agent())

        print("[eval] Resetting kernel state...")
        await self.websocket.send(json.dumps({
            "type": "agent_action",
            "action": "reset_kernel_globals",
            "params": {},
            "tx_id": str(uuid.uuid4()),
        }))

        await self.agent_conn.send({
            "type": "agent_query",
            "query": initial_query,
            "request_id": f"eval-init-{self.session_id}"
        })

        while not self.eval_complete:
            if forward_task.done() or receive_task.done():
                print("[eval] One of the forwarding tasks completed unexpectedly")
                if forward_task.done():
                    try:
                        forward_task.result()
                    except Exception as e:
                        print(f"[eval] Forward task error: {e}")
                    if self.websocket:
                        forward_task = asyncio.create_task(forward_agent_to_console())
                if receive_task.done():
                    try:
                        receive_task.result()
                    except Exception as e:
                        print(f"[eval] Receive task error: {e}")
                    if self.websocket:
                        receive_task = asyncio.create_task(forward_console_to_agent())
                break
            await asyncio.sleep(1)

        print("[eval] Eval complete, stopping forwarding tasks...")
        receive_task.cancel()
        try:
            await asyncio.wait_for(receive_task, timeout=0.1)
        except (TimeoutError, asyncio.CancelledError):
            pass

        forward_task.cancel()
        try:
            await asyncio.wait_for(forward_task, timeout=0.1)
        except (TimeoutError, asyncio.CancelledError):
            pass

        duration_ms = (time.time() - start_time) * 1000

        print(f"[eval] Fetching full conversation history from database...")
        conversation_history = await self.fetch_full_conversation_history()
        print(f"[eval] Retrieved {len(conversation_history)} messages from database")

        agent_answer = extract_answer_from_conversation(conversation_history)
        if agent_answer is not None:
            print(f"[eval] Extracted answer: {json.dumps(agent_answer)[:200]}...")
        else:
            print("[eval] No answer extracted from conversation")

        eval_result = EvalResult(
            eval_id=eval_case.id,
            conversation_history=conversation_history,
            duration_ms=duration_ms,
            agent_answer=agent_answer,
        )

        if eval_case.grader:
            print("[eval] Running binary grader...")
            grader_type = eval_case.grader.get("type")
            grader_config = eval_case.grader.get("config", {})

            if agent_answer is None:
                eval_result.grader_result = {
                    "passed": False,
                    "metrics": {},
                    "reasoning": "Failed to extract answer from conversation history",
                    "agent_answer": None
                }
                print("[eval] Grader result: FAIL (no answer extracted)")
            elif grader_type in GRADER_REGISTRY:
                grader_cls = GRADER_REGISTRY[grader_type]
                grader = grader_cls()
                grader_result = grader.evaluate(agent_answer, grader_config)

                eval_result.grader_result = {
                    "passed": grader_result.passed,
                    "metrics": grader_result.metrics,
                    "reasoning": grader_result.reasoning,
                    "agent_answer": grader_result.agent_answer
                }

                print(f"[eval] Grader result: {'PASS' if grader_result.passed else 'FAIL'}")
                print(f"[eval] Grader reasoning:\n{grader_result.reasoning}")
            else:
                print(f"[eval] Warning: Unknown grader type '{grader_type}'")

        print(f"\n[eval] Eval completed in {duration_ms / 1000:.2f}s")
        print(f"[eval] Total conversation turns: {len(conversation_history)}")

        return eval_result


async def run_with_websocket_server(port: int, connection_handler, done_event: asyncio.Event):
    async with websockets.serve(
        connection_handler,
        "localhost",
        port,
        max_size=10 * 1024 * 1024
    ):
        print(f"[eval] WebSocket server listening on ws://localhost:{port}/agent")
        print("[eval] Waiting for a running plot notebook to connect to the local agent.")

        await done_event.wait()


async def run_eval_batch(eval_cases: list[Eval], port: int, sandbox_dir: Path, interactive: bool = False) -> list[EvalResult]:
    server = EvalServer(sandbox_dir)
    done_event = asyncio.Event()
    results: list[EvalResult] = []

    async def connection_handler(websocket):
        if websocket.path == "/agent":
            if server.agent_proc is not None:
                print(f"[eval] Console reconnected, updating websocket")
                server.websocket = websocket
                try:
                    await done_event.wait()
                except Exception:
                    pass
                return

            num_evals = len(eval_cases)
            print(f"[eval] Console connected ({'single eval' if num_evals == 1 else f'batch of {num_evals} evals'})")

            server.websocket = websocket
            await server.start_agent()

            await server.initialize_agent_session(websocket)

            for eval_case in eval_cases:
                await server.reset_for_next_test()
                result = await server.run_eval(eval_case)
                results.append(result)

            if interactive:
                print("\n[eval] Interactive mode - agent still running. Press Ctrl+C to exit.")
                await server.keep_forwarding()

            await server.stop_agent()
            done_event.set()
        else:
            print(f"[eval] Unknown path: {websocket.request.path}")
            await websocket.close()

    await run_with_websocket_server(port, connection_handler, done_event)
    return results


def create_sandbox(sandbox_dir: Path, eval_case: Eval):
    if sandbox_dir.exists():
        print(f"[eval] Removing existing sandbox at {sandbox_dir}")
        shutil.rmtree(sandbox_dir)

    sandbox_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] Created fresh sandbox at {sandbox_dir}")

    user_token = Path.home() / ".latch" / "token"
    if user_token.exists():
        (sandbox_dir / "token").write_text(user_token.read_text())
    else:
        (sandbox_dir / "token").write_text("local-dev-token")

    (sandbox_dir / "session-id").write_text(f"eval-{eval_case.id}")
    (sandbox_dir / "nucleus-url").write_text("https://nucleus.latch.bio")
    (sandbox_dir / "id").write_text("0")


def write_results(results: list[EvalResult], output_path: Path):
    evals = []
    for r in results:
        entry = {
            "eval_id": r.eval_id,
            "duration_ms": r.duration_ms,
            "passed": r.grader_result.get("passed") if r.grader_result else None,
            "reasoning": r.grader_result.get("reasoning") if r.grader_result else None,
            "agent_answer": r.agent_answer,
        }
        evals.append(entry)

    passed = sum(1 for e in evals if e["passed"] is True)
    total = len(evals)
    accuracy = passed / total if total > 0 else 0

    output = {
        "accuracy": accuracy,
        "passed": passed,
        "total": total,
        "evals": evals,
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"[eval] Results written to {output_path}")
    print(f"[eval] Accuracy: {passed}/{total} ({accuracy:.1%})")

    workspaces_dir = output_path.parent / "workspaces"
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        eval_dir = workspaces_dir / r.eval_id
        eval_dir.mkdir(parents=True, exist_ok=True)

        (eval_dir / "trajectory.json").write_text(json.dumps(r.trajectory, indent=2))

        agent_log_lines = []
        for event in r.trajectory:
            agent_log_lines.append(json.dumps(event))
        (eval_dir / "agent_output.log").write_text("\n".join(agent_log_lines))

        if r.agent_answer is not None:
            (eval_dir / "eval_answer.json").write_text(json.dumps(r.agent_answer, indent=2))

        result_data = {
            "eval": r.eval_id,
            "model": "anthropic/claude-sonnet-4",
            "agent": "plots-agent",
            "passed": r.grader_result.get("passed") if r.grader_result else None,
            "duration_s": r.duration_ms / 1000,
            "agent_answer": r.agent_answer,
            "grader_result": r.grader_result,
        }
        (eval_dir / "_result.json").write_text(json.dumps(result_data, indent=2))

        print(f"[eval] Wrote trajectory for {r.eval_id} to {eval_dir}")


async def main():
    parser = argparse.ArgumentParser(description="Run agent eval server")
    parser.add_argument("--eval", help="Eval file or directory to run")
    parser.add_argument("--output", "-o", help="Output file for results (default: results.json)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Keep agent running after eval for interaction")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode with temporary notebook")
    args = parser.parse_args()

    sandbox_dir = Path.cwd() / "sandboxes" / "batch"
    eval_path = Path(args.eval)
    output_path = Path(args.output) if args.output else Path("results.json")

    eval_cases = []
    if eval_path.is_dir():
        print(f"[eval] Loading test cases from directory: {eval_path}")
        for json_file in sorted(eval_path.rglob("*.json")):
            with open(json_file) as f:
                test_data = json.load(f)
                eval_cases.append(Eval(**test_data))
        print(f"[eval] Found {len(eval_cases)} test cases")
    else:
        with open(eval_path) as f:
            test_data = json.load(f)
            eval_cases.append(Eval(**test_data))

    if not eval_cases:
        print("[eval] No test cases found")
        return

    create_sandbox(sandbox_dir, eval_cases[0])

    try:
        if args.headless:
            results = await run_eval_batch_headless(eval_cases, sandbox_dir)
        else:
            results = await run_eval_batch(eval_cases, 8765, sandbox_dir, interactive=args.interactive)
        print(f"\n[eval] Batch complete: {len(results)}/{len(eval_cases)} evals completed")
        write_results(results, output_path)
    except KeyboardInterrupt:
        print("\n[eval] Interrupted by user")
    except Exception:
        print("[eval] Error during eval execution")
        raise


if __name__ == "__main__":
    asyncio.run(main())
