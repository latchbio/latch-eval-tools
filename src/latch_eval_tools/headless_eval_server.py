import asyncio
import json
import os
import sys
import textwrap
import time
import uuid
from pathlib import Path

import aiohttp
import websockets
from eval_graders import GRADER_REGISTRY

from latch_eval_tools.answer_extraction import extract_answer_from_conversation
from latch_eval_tools.types import Eval, EvalResult

faas_runtime_dir = Path(os.environ.get("LATCH_PLOTS_FAAS_PATH", "/root/latch-plots-faas")) / "runtime" / "mount"
sys.path.insert(0, str(faas_runtime_dir))

from utils import gql_query


def get_auth_token() -> str:
    return f"Latch-SDK-Token {(Path.home() / '.latch' / 'token').read_text().strip()}"


async def get_workspace_id_from_token() -> str:
    auth = get_auth_token()

    resp = await gql_query(
        auth=auth,
        query="""
            query GetAccount {
                accountInfoCurrentOrRegister {
                    id
                }
            }
        """,
        variables={},
    )

    return resp["data"]["accountInfoCurrentOrRegister"]["id"]


async def create_eval_notebook(workspace_id: str, eval_id: str) -> str:
    auth = get_auth_token()

    display_name = f"eval-{eval_id}-{uuid.uuid4().hex[:8]}"
    resp = await gql_query(
        auth=auth,
        query="""
            mutation PlotsCreateNotebook($wsId: BigInt!, $displayName: String!) {
                createPlotNotebookInfo(
                    input: {
                        plotNotebookInfo: {
                            ownerId: $wsId
                            displayName: $displayName
                        }
                    }
                ) {
                    plotNotebookInfo {
                        id
                    }
                }
            }
        """,
        variables={
            "wsId": workspace_id,
            "displayName": display_name,
        },
    )

    notebook_id = resp["data"]["createPlotNotebookInfo"]["plotNotebookInfo"]["id"]
    print(f"[headless] Created eval notebook: {notebook_id}")

    await gql_query(
        auth=auth,
        query="""
            mutation DeletePlotNotebook($id: BigInt!) {
                deletePlotNotebook(input: { argNotebookId: $id }) {
                    clientMutationId
                }
            }
        """,
        variables={"id": notebook_id},
    )
    print(f"[headless] Deleted eval notebook (hidden from frontend list): {notebook_id}")

    return notebook_id


async def get_or_create_session(notebook_id: str) -> int:
    auth = get_auth_token()

    resp = await gql_query(
        auth=auth,
        query="""
            query AgentSessionsByNotebook($notebookId: BigInt!) {
                agentSessions(
                    filter: {plotNotebookId: {equalTo: $notebookId}}
                    orderBy: [CREATED_AT_DESC]
                ) {
                    nodes { id removedAt }
                }
            }
        """,
        variables={"notebookId": notebook_id},
    )

    nodes = resp.get("data", {}).get("agentSessions", {}).get("nodes", [])
    active_sessions = [n for n in nodes if n.get("removedAt") is None]

    if active_sessions:
        session_id = int(active_sessions[0]["id"])
        print(f"[headless] Using existing session: {session_id}")
        return session_id

    print(f"[headless] Creating new session for notebook {notebook_id}...")
    await gql_query(
        auth=auth,
        query="""
            mutation CreateAgentSession($notebookId: BigInt!, $metadata: JSON) {
                createAgentSession(
                    input: {agentSession: {plotNotebookId: $notebookId, metadata: $metadata}}
                ) {
                    clientMutationId
                }
            }
        """,
        variables={"notebookId": notebook_id, "metadata": None},
    )

    resp = await gql_query(
        auth=auth,
        query="""
            query AgentSessionsByNotebook($notebookId: BigInt!) {
                agentSessions(
                    filter: {plotNotebookId: {equalTo: $notebookId}}
                    orderBy: [CREATED_AT_DESC]
                ) {
                    nodes { id }
                }
            }
        """,
        variables={"notebookId": notebook_id},
    )

    nodes = resp.get("data", {}).get("agentSessions", {}).get("nodes", [])
    if not nodes:
        raise RuntimeError("Failed to create session")

    session_id = int(nodes[0]["id"])
    print(f"[headless] Created session: {session_id}")
    return session_id


class HeadlessEvalServer:
    def __init__(self, sandbox_dir: Path, port: int = 5000):
        self.sandbox_dir = sandbox_dir
        self.port = port
        self.workspace_id: str | None = None
        self.notebook_id: str | None = None
        self.server_proc = None
        self.websocket = None
        self.session_id = None
        self.eval_complete = False
        self.conversation_history: list[dict] = []
        self.current_streaming_message: dict | None = None
        self.current_streaming_blocks: list[dict] = []
        self.trajectory: list[dict] = []
        self.trajectory_session_id: str = ""
        self.current_usage: dict | None = None
        self.turn_number: int = 0
        self.eval_start_time: float = 0

    async def start_server(self):
        print("[headless] Starting runtime server via wrapper...")

        if self.notebook_id is None:
            raise RuntimeError("notebook_id must be set before starting server")

        faas_venv_python_override = os.environ.get("PLOTS_FAAS_PYTHON")
        if faas_venv_python_override:
            faas_venv_python = Path(faas_venv_python_override)
        else:
            faas_dir = Path(os.environ.get("LATCH_PLOTS_FAAS_PATH", "/root/latch-plots-faas"))
            faas_venv_python = faas_dir / ".venv" / "bin" / "python"
        
        # Use wrapper_entrypoint from this package
        wrapper_script = Path(__file__).parent / "wrapper_entrypoint.py"

        if not faas_venv_python.exists():
            raise RuntimeError(f"latch-plots-faas venv not found at {faas_venv_python}")

        if not wrapper_script.exists():
            raise RuntimeError(f"Wrapper script not found at {wrapper_script}")

        cmd = [
            str(faas_venv_python),
            "-u",
            str(wrapper_script),
            "--sandbox-dir", str(self.sandbox_dir),
            "--port", str(self.port),
            "--notebook-id", str(self.notebook_id),
        ]

        self.server_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
                    print(f"[server] {prefix}{decoded}", flush=True)
                except Exception as e:
                    print(f"[server] {prefix}[Error reading output: {e}]", flush=True)
                    break

        asyncio.create_task(stream_output(self.server_proc.stdout, ""))
        asyncio.create_task(stream_output(self.server_proc.stderr, "[stderr] "))

        await self.wait_for_ready()

    async def wait_for_ready(self, timeout: float = 60.0, poll_interval: float = 1.0):
        print("[headless] Waiting for server to be ready...")
        start = time.time()
        server_responded = False

        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.port}/readyz", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            print("[headless] Server is ready!")
                            return
                        if resp.status == 500:
                            if not server_responded:
                                print("[headless] Server responding, waiting for agent...")
                                server_responded = True
            except aiohttp.ClientConnectorError:
                pass
            except Exception:
                pass

            if self.server_proc.returncode is not None:
                raise RuntimeError(f"Server process exited unexpectedly with code {self.server_proc.returncode}")

            await asyncio.sleep(poll_interval)

        if server_responded:
            print("[headless] Server responding but agent not ready, proceeding anyway")
            return

        raise TimeoutError("Server did not become ready in time")

    async def connect(self):
        print("[headless] Waiting for agent to be ready...")
        await asyncio.sleep(3)

        print("[headless] Connecting to /agent WebSocket...")

        for attempt in range(5):
            try:
                self.websocket = await websockets.connect(
                    f"ws://localhost:{self.port}/agent",
                    max_size=10 * 1024 * 1024,
                )
                break
            except websockets.exceptions.InvalidStatus as e:
                print(f"[headless] WebSocket connection attempt {attempt + 1} failed: {e}")
                if attempt < 4:
                    await asyncio.sleep(2)
                else:
                    raise

        self.session_id = await get_or_create_session(self.notebook_id)

        sdk_token = (Path.home() / ".latch" / "token").read_text().strip()
        local_storage = {
            "plots.is_agent_controlled": "yes",
            "plots.is_eval_harness": "yes",
            "viewAccountId": self.workspace_id,
            "latch.authData": json.dumps({
                "status": "done",
                "auth0Data": {
                    "idToken": sdk_token,
                    "idTokenPayload": {
                        "sub": "agent-session",
                        "latch.bio/tos_ok": "true",
                    },
                },
            }),
        }

        init_msg = {
            "type": "init",
            "notebook_id": self.notebook_id,
            "session_id": self.session_id,
            "local_storage": local_storage,
        }

        await self.websocket.send(json.dumps(init_msg))
        print(f"[headless] Sent init message for notebook {self.notebook_id} with session_id {self.session_id}")

        while True:
            msg_str = await self.websocket.recv()
            msg = json.loads(msg_str)
            msg_type = msg.get("type")

            if msg_type == "agent_status" and msg.get("status") == "ready":
                print(f"[headless] Agent is ready! session_id={self.session_id}")
                break
            elif msg_type == "agent_error":
                raise RuntimeError(f"Agent error: {msg.get('error')}")

            print(f"[headless] Waiting for agent ready, got: {msg_type}")

    def get_conversation_history(self) -> list[dict]:
        return list(self.conversation_history)

    def get_trajectory(self) -> list[dict]:
        return list(self.trajectory)

    def init_trajectory(self, eval_id: str):
        self.trajectory_session_id = str(uuid.uuid4())
        self.trajectory = []
        self.turn_number = 0
        self.eval_start_time = time.time()
        self.trajectory.append({
            "type": "system",
            "subtype": "init",
            "timestamp": self.eval_start_time,
            "session_id": self.trajectory_session_id,
            "eval_id": eval_id,
            "notebook_id": self.notebook_id,
            "tools": [
                "create_cell",
                "delete_cell",
                "update_cell",
                "run_cell",
                "execute_code",
                "get_context",
                "request_reactivity_summary",
                "submit_response",
            ],
            "model": "claude-sonnet-4-20250514",
            "agent": "plots-agent",
            "uuid": str(uuid.uuid4()),
        })

    def add_assistant_to_trajectory(self, message: dict):
        self.turn_number += 1
        self.trajectory.append({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": message.get("content", []),
            },
            "turn": self.turn_number,
            "timestamp": time.time(),
            "elapsed_s": time.time() - self.eval_start_time,
            "usage": self.current_usage,
            "session_id": self.trajectory_session_id,
            "uuid": str(uuid.uuid4()),
        })
        self.current_usage = None

    def add_tool_result_to_trajectory(self, tool_use_id: str, result: str, is_error: bool = False, cell_id: str | None = None):
        content = [{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result,
            "is_error": is_error,
        }]
        entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": content,
            },
            "timestamp": time.time(),
            "elapsed_s": time.time() - self.eval_start_time,
            "session_id": self.trajectory_session_id,
            "uuid": str(uuid.uuid4()),
            "tool_use_result": f"{'Error: ' if is_error else ''}{result[:500]}{'...' if len(result) > 500 else ''}",
        }
        if cell_id:
            entry["cell_id"] = cell_id
        self.trajectory.append(entry)

    def add_to_history(self, msg: dict):
        msg_type = msg.get("type")
        if msg_type in ("anthropic_message", "user_message"):
            self.conversation_history.append(msg)
        elif msg_type == "agent_stream_start":
            self.current_streaming_message = {
                "type": "anthropic_message",
                "role": "assistant",
                "content": [],
            }
            self.current_streaming_blocks = []
        elif msg_type == "agent_stream_block_start":
            block_type = msg.get("block_type")
            if block_type == "text":
                self.current_streaming_blocks.append({"type": "text", "text": ""})
            elif block_type == "thinking":
                self.current_streaming_blocks.append({"type": "thinking", "thinking": ""})
            elif block_type == "tool_use":
                self.current_streaming_blocks.append({
                    "type": "tool_use",
                    "id": msg.get("block_id"),
                    "name": msg.get("block_name"),
                    "input": {},
                })
        elif msg_type == "agent_stream_delta":
            block_index = msg.get("block_index", 0)
            delta = msg.get("delta", "")
            if block_index < len(self.current_streaming_blocks):
                block = self.current_streaming_blocks[block_index]
                if block.get("type") == "text":
                    block["text"] += delta
                elif block.get("type") == "thinking":
                    block["thinking"] += delta
                elif block.get("type") == "tool_use":
                    block["input_raw"] = block.get("input_raw", "") + delta
        elif msg_type == "agent_usage_update":
            self.current_usage = msg.get("usage")
        elif msg_type == "agent_stream_complete":
            if self.current_streaming_message is not None:
                for block in self.current_streaming_blocks:
                    if block.get("type") == "tool_use" and "input_raw" in block:
                        try:
                            block["input"] = json.loads(block.pop("input_raw"))
                        except json.JSONDecodeError:
                            block["input"] = {}
                self.current_streaming_message["content"] = self.current_streaming_blocks
                self.conversation_history.append(self.current_streaming_message)
                self.add_assistant_to_trajectory(self.current_streaming_message)
                print(f"[headless] Built message with {len(self.current_streaming_blocks)} blocks")
                self.current_streaming_message = None
                self.current_streaming_blocks = []
        elif msg_type == "kernel_message":
            inner_msg = msg.get("message", {})
            if inner_msg.get("type") == "cell_result":
                cell_id = inner_msg.get("cell_id", "")
                has_exception = inner_msg.get("has_exception", False)
                logs = inner_msg.get("logs", "")
                exception = inner_msg.get("exception")
                result_str = logs if logs else "Cell executed successfully"
                if has_exception and exception:
                    result_str = f"Exception: {exception}\n{logs}"
                tool_use_id = self.find_last_tool_use_id_for_cell(cell_id)
                if tool_use_id:
                    self.add_tool_result_to_trajectory(tool_use_id, result_str, is_error=has_exception, cell_id=cell_id)

    def find_last_tool_use_id_for_cell(self, cell_id: str) -> str | None:
        for entry in reversed(self.trajectory):
            if entry.get("type") == "assistant":
                content = entry.get("message", {}).get("content", [])
                for block in content:
                    if block.get("type") == "tool_use" and block.get("name") == "create_cell":
                        return block.get("id")
        return None

    async def handle_agent_action(self, msg: dict):
        action = msg.get("action")
        tx_id = msg.get("tx_id")
        params = msg.get("params", {})

        print(f"[headless] Handling action: {action} (tx_id={tx_id})")

        response = {
            "type": "agent_action_response",
            "tx_id": tx_id,
            "status": "success",
        }

        if action == "get_context":
            response["context"] = {
                "cells": [],
                "selected_cells": [],
                "data_tree": {},
            }
        elif action == "create_cell":
            cell_id = f"cell_{uuid.uuid4().hex[:8]}"
            tf_id = f"tf_{uuid.uuid4().hex[:8]}"
            response["cell_id"] = cell_id
            response["tf_id"] = tf_id
            response["title"] = params.get("title", "")
            if params.get("auto_run"):
                asyncio.create_task(self.send_mock_cell_result(cell_id))
        elif action == "delete_cell":
            response["deleted"] = True
        elif action == "update_cell":
            response["updated"] = True
        elif action == "run_cell":
            response["started"] = True
            cell_id = params.get("cell_id")
            if cell_id:
                asyncio.create_task(self.send_mock_cell_result(cell_id))
        else:
            response["status"] = "error"
            response["error"] = f"Unknown action: {action}"

        await self.websocket.send(json.dumps(response))

    async def send_mock_cell_result(self, cell_id: str):
        await asyncio.sleep(0.5)
        result_msg = {
            "type": "kernel_message",
            "message": {
                "type": "cell_result",
                "cell_id": cell_id,
                "has_exception": False,
                "outputs": [],
            }
        }
        if self.websocket:
            await self.websocket.send(json.dumps(result_msg))
            print(f"[headless] Sent mock cell result for {cell_id}")

    def clear_history(self):
        self.conversation_history.clear()

    def check_for_completion(self) -> bool:
        for payload in self.conversation_history:
            if payload.get("type") == "anthropic_message" and payload.get("role") == "assistant":
                content = payload.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "submit_response":
                        tool_input = block.get("input", {})
                        if tool_input.get("next_status") == "done":
                            return True
        return False

    async def clear_agent_history(self):
        print("[headless] Clearing agent history...")
        self.clear_history()
        await self.websocket.send(json.dumps({"type": "agent_clear_history"}))
        await asyncio.sleep(1)

    async def run_eval(self, eval_case: Eval) -> EvalResult:
        print(f"\n{'=' * 70}")
        print(f"Running eval: {eval_case.id}")
        print("=" * 70)

        start_time = time.time()
        self.eval_complete = False
        self.init_trajectory(eval_case.id)

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

            IMPORTANT: When you finish this task, include your answer in your submit_response summary as raw JSON (no markdown code fences) wrapped in <EVAL_ANSWER></EVAL_ANSWER> tags.

            Example format for your summary:
            <EVAL_ANSWER>
            {{"field1": value1, "field2": value2}}
            </EVAL_ANSWER>

            Do NOT use markdown code fences (```json) inside the EVAL_ANSWER tags - use raw JSON only.
            {data_context}
        """).strip()

        await self.websocket.send(json.dumps({
            "type": "agent_query",
            "query": initial_query,
            "request_id": f"eval-{eval_case.id}-{uuid.uuid4()}",
        }))

        print("[headless] Query sent, waiting for completion...")

        while not self.eval_complete:
            try:
                msg_str = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                msg = json.loads(msg_str)
                msg_type = msg.get("type", "unknown")

                if msg_type != "agent_stream_delta":
                    print(f"[headless] Received: {msg_type}")

                self.add_to_history(msg)

                if msg_type == "agent_error":
                    error_msg = msg.get("error", "Unknown error")
                    print(f"[headless] Agent error received: {error_msg}")
                    raise RuntimeError(f"Agent error: {error_msg}")

                if msg_type in ("agent_history_updated", "agent_stream_complete"):
                    self.eval_complete = self.check_for_completion()
                    if self.eval_complete:
                        print("[headless] Detected completion via submit_response")

            except asyncio.TimeoutError:
                self.eval_complete = self.check_for_completion()
            except websockets.exceptions.ConnectionClosed:
                print("[headless] WebSocket connection closed")
                break

        duration_ms = (time.time() - start_time) * 1000

        conversation_history = self.get_conversation_history()
        print(f"[headless] Retrieved {len(conversation_history)} messages from local history")

        trajectory = self.get_trajectory()
        print(f"[headless] Captured {len(trajectory)} trajectory events")

        agent_answer = extract_answer_from_conversation(conversation_history)
        if agent_answer is not None:
            print(f"[headless] Extracted answer: {json.dumps(agent_answer)[:200]}...")
        else:
            print("[headless] No answer extracted from conversation")

        eval_result = EvalResult(
            eval_id=eval_case.id,
            conversation_history=conversation_history,
            trajectory=trajectory,
            duration_ms=duration_ms,
            agent_answer=agent_answer,
        )

        if eval_case.grader:
            print("[headless] Running grader...")
            grader_type = eval_case.grader.get("type")
            grader_config = eval_case.grader.get("config", {})

            if agent_answer is None:
                eval_result.grader_result = {
                    "passed": False,
                    "metrics": {},
                    "reasoning": "Failed to extract answer from conversation history",
                    "agent_answer": None,
                }
                print("[headless] Grader result: FAIL (no answer extracted)")
            elif grader_type in GRADER_REGISTRY:
                grader_cls = GRADER_REGISTRY[grader_type]
                grader = grader_cls()
                grader_result = grader.evaluate(agent_answer, grader_config)

                eval_result.grader_result = {
                    "passed": grader_result.passed,
                    "metrics": grader_result.metrics,
                    "reasoning": grader_result.reasoning,
                    "agent_answer": grader_result.agent_answer,
                }

                print(f"[headless] Grader result: {'PASS' if grader_result.passed else 'FAIL'}")
                print(f"[headless] Grader reasoning:\n{grader_result.reasoning}")
            else:
                print(f"[headless] Warning: Unknown grader type '{grader_type}'")

        print(f"\n[headless] Eval completed in {duration_ms / 1000:.2f}s")
        print(f"[headless] Total conversation turns: {len(conversation_history)}")

        return eval_result

    async def stop_server(self):
        print("[headless] Stopping server...")

        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None

        if self.server_proc:
            try:
                if self.server_proc.returncode is None:
                    self.server_proc.terminate()
                    await asyncio.wait_for(self.server_proc.wait(), timeout=5)
            except TimeoutError:
                self.server_proc.kill()
                await self.server_proc.wait()
            except ProcessLookupError:
                pass
            self.server_proc = None

        print("[headless] Server stopped")


async def run_eval_batch_headless(eval_cases: list[Eval], sandbox_dir: Path) -> list[EvalResult]:
    results: list[EvalResult] = []

    server = HeadlessEvalServer(sandbox_dir)

    server.workspace_id = await get_workspace_id_from_token()
    print(f"[headless] Using workspace: {server.workspace_id}")

    first_eval_id = eval_cases[0].id if eval_cases else "batch"
    server.notebook_id = await create_eval_notebook(server.workspace_id, first_eval_id)

    await server.start_server()
    await server.connect()

    for i, eval_case in enumerate(eval_cases):
        print(f"\n[headless] Running eval {i + 1}/{len(eval_cases)}")

        await server.clear_agent_history()

        result = await server.run_eval(eval_case)
        results.append(result)

        await server.stop_server()

    return results
