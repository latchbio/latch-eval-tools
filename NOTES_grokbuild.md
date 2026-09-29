# grok-build harness — status & verification checklist

Adds [xai-org/grok-build](https://github.com/xai-org/grok-build) (SpaceXAI's
open-source Rust coding agent, the `grok` CLI) as a CLI harness alongside
claudecode / openaicodex / pi. Structurally it's a fourth `_run_cli_agent`
agent: a thin adapter (`harness/grokbuild.py`) plus per-agent branches in
`_cli_runner.py` and `run_summary.py`.

## What's wired up

- `harness/grokbuild.py` — `run_grokbuild_task`, `MODEL_MAP` (`xai/…` → grok
  model string), `XAI_API_KEY` check. Exported from `harness/__init__.py`.
- `_cli_runner.py` — `GROK_ENV_KEYS`, `AGENT_STATE_DIRS["grokbuild"]=".grok"`,
  `AGENT_IDENTIFIER_KEYS`, command construction, ENV_KEYS selection,
  `_extract_last_message` (ACP chunk concatenation), `_extract_metadata`
  (session id).
- `run_summary.py` — `CliHarnessAgentType` += `"grokbuild"`, `_grok_metrics`
  (+ usage/step/cost helpers), dispatch in `build_cli_run_summary`.
- `agent_env/Dockerfile` — installs the `grok` CLI.
- README — lists `run_grokbuild_task`.

Command emitted (headless):

```
grok --no-auto-update --no-alt-screen --always-approve --cwd /workspace \
     --output-format streaming-json [--resume <ID>] [--system-prompt <SP>] \
     -p "<PROMPT>" --model <MODEL>
```

## Verified against a real grok build (grok 1.0.3, via examples/Dockerfile.grok-smoke)

- **Install works.** `curl -fsSL https://x.ai/cli/install.sh | bash` installs
  grok 1.0.3 and `grok --version` runs. The whole install lands under
  `/root/.grok` (binary at `~/.grok/downloads/…`, symlinked via `~/.grok/bin`
  and `/usr/local/bin/grok`).
- **State-dir collision found + fixed.** Bind-mounting the state dir over
  `/root/.grok` (the original design) SHADOWS the binary — `grok` then vanishes
  from PATH and every run dies with "executable file not found". Fixed by
  mounting the host `.grok` state dir onto `/root/.grok/sessions` instead (see
  `AGENT_CONTAINER_STATE_MOUNTS`). Verified both the break and the fix.
- **streaming-json is FLAT, not ACP.** `grok --help` calls it "NDJSON of the
  agent native ACP session updates", but a real capture shows plain
  `{"type":...}` events (`text`/`thought`/`tool_call`/`tool_call_update`/
  `usage`/`available_commands`/`end`) — NOT the JSON-RPC `session/update`
  envelope. The parser was rewritten to this real schema
  (`examples/grok_smoke_traj.jsonl` is the reference capture). There is also a
  `--prompt-file <PATH>` option (an alternative to `-p` if arg escaping bites).

## ✅ End-to-end verified (grok 1.0.3, grok-4.6, real API)

Ran the full harness against `examples/grok_smoke_eval.json` via
`examples/run_grokbuild_eval.py` (image `grok-smoke`):

```
answer: {"answer": 42}              # grok wrote /workspace/eval_answer.json; harness parsed it
usage: input 914, output 100, cache_read 21888, reasoning 65
cost: $0.013762  (provider_reported)   n_steps: 2   n_turns: 2
session_id: 019ff78a-996e-73b3-a4e0-da92a85f5898
timed_out: False   error: None
```

Everything resolved, and all against the real event schema:

1. **Prompt delivery** — `-p <arg>` works (the redundant stdin write is ignored).
2. **Token usage** — pulled from the `end` event's cumulative `usage`
   (field names `input_tokens` / `output_tokens` / `cache_read_input_tokens` /
   `cache_creation_input_tokens` / `reasoning_tokens`).
3. **Session id** — top-level `sessionId` on the `end` event → captured, and
   `load_trajectory_identifier()` reaches it, so **resume works** (no descend
   needed).
4. **Assistant text** — last contiguous run of `{"type":"text","data":...}`.
5. **Step count** — one per `{"type":"tool_call"}`.
6. **Cost** — `end.total_cost_usd` is provider-reported; no pricing table needed.
7. **Answer protocol** — grok writes `/workspace/eval_answer.json` under
   `--cwd /workspace`, exactly where the runner looks.

## Resolved follow-ups

- **grok version pinned** to `1.0.3` in both Dockerfiles
  (`... | bash -s 1.0.3`).
- **One-shot answer-resume parity.** grok now shares Claude Code's resume-nudge
  safety net: if it ends its turn (one-shot `-p`) without writing
  `eval_answer.json`, the harness resumes the session and nudges it to finish
  synchronously (up to `MAX_CLAUDECODE_ANSWER_RESUMES`). Was claudecode-only;
  the block in `_run_cli_agent` now covers `("claudecode", "grokbuild")`.
- **`agent_env` "uv sync" failure was an arch artifact, not a real break.**
  `pyscipopt` (transitive via the scipy stack) has x86_64 manylinux wheels
  only. Building locally on arm64 (Apple Silicon) source-compiles it and fails
  for lack of SCIP headers; the eval platform's amd64 build uses the wheel and
  succeeds. Fix = build for the target arch: `examples/build_agent_env.sh` (and
  a Dockerfile note) force `--platform=linux/amd64`. Nothing in the grok layer
  was at fault. Verified on amd64: `pyscipopt==5.7.1` installs from a wheel
  (no compile). Caveat: a full *local emulated* amd64 build on Apple Silicon
  still can't finish — the Claude Code (Bun) installer segfaults under qemu
  before uv sync — so build the full image on a NATIVE amd64 host / CI, which
  is how the published `benchmark_agent` image is already produced.

## Remaining follow-ups

- **Multi-turn resume** wired (`--resume`, sessionId reachable) but not yet
  exercised across an actual OOM/clean-exit resume.
- **Provider-failure classification** (rate-limit retry) not implemented;
  `classify_terminal_provider_failure` returns None for grok, so no auto-retry
  on transient 429/5xx. Add a `_grok_provider_failure` once the error event
  shape is captured (look for an `end.stopReason` / error event on a throttle).
