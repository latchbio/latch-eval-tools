---
name: run-eval
description: Run a drafted eval JSON locally through latch-eval-tools — download its data, run a real agent against the task in a Docker sandbox, and grade the answer. Use when the user says "run this eval", "test this eval", "try the eval", "/run-eval", or wants a fast draft→run→inspect loop on an eval JSON.
---

# Run Eval (local)

Close the loop between drafting an eval and seeing how a real agent does on it.
This is a **local** smoke test built on the `latch-eval` CLI: it downloads the
eval's `data_node`, runs the chosen agent harness against the `task` prompt
inside a Docker sandbox, captures the agent's `eval_answer.json`, and runs the
eval's grader on it. Faster and tighter than the platform judge, but a single
replicate — a sanity loop, not a replacement for the platform's solvability /
judge runs.

> Drop this file into `.claude/skills/run-eval/SKILL.md` (project) or
> `~/.claude/skills/run-eval/SKILL.md` (user-wide) to expose it as a skill.

## Setup

- **`latch-eval` on PATH.** Confirm with `command -v latch-eval`. If missing,
  install it as a standalone tool (on PATH regardless of the active env):
  `uv tool install latch-eval-tools` (or `pipx install latch-eval-tools`); for a
  local checkout use `uv tool install --editable .`. No-PATH fallback:
  `uvx --from latch-eval-tools latch-eval run ...`.
- **Docker** must be running — every harness runs the agent in a container.
- **Provider API key** for the chosen harness:
  - `claudecode` → `ANTHROPIC_API_KEY`
  - `openaicodex` → `OPENAI_API_KEY` or `CODEX_API_KEY`
  - `minisweagent` → key matching `--model` (e.g. `ANTHROPIC_API_KEY`)
  - `pi` → the relevant provider key
- **Latch token** (`~/.latch/token`, via `latch login`) so `data_node` downloads
  work. Evals with no `data_node`, or runs using `--data`, skip this.

## Steps

1. **Locate the eval JSON.** Take the path from the user; if not given, ask (or
   find the eval/task JSON in the current project).

2. **Pick the harness and model — every run.** No default. Ask which harness
   (`claudecode` | `minisweagent` | `openaicodex` | `pi`) and which model.
   **`minisweagent` requires an explicit `--model`** (e.g.
   `anthropic/claude-sonnet-4-6`); the others fall back to the harness default if
   `--model` is omitted.

3. **Run it**, writing a structured result and keeping the workspace (default):

   ```bash
   latch-eval run \
     -e <path/to/eval.json> \
     --harness <harness> \
     [--model <model>] \
     --json-out /tmp/run-eval-<eval_id>.json
   ```

   Add `--eval-timeout <seconds>` for long tasks. The command prints a RUN
   SUMMARY and exits non-zero on FAIL / NO GRADE.

   **Data: cached, not re-downloaded.** Each `latch://` `data_node` is downloaded
   once into the shared cache (`~/.cache/latch-eval`, override with `--cache-dir`),
   then hardlinked (not copied) into each run's workspace. Re-running the same
   eval — from any directory — reuses the cache, so there is no re-download.

   **Point at local data you already have** to skip Latch entirely:

   ```bash
   latch-eval run -e <eval.json> --harness <harness> \
     --data /path/to/data.tsv.gz \
     --json-out /tmp/run-eval-<eval_id>.json
   ```

   `--data` overrides the eval's `data_node` (so no token/download needed) and is
   repeatable. A file lands at `/workspace/data/<filename>`; a directory lands at
   `/workspace/data/<dirname>/…`.

4. **Report the result.** Read `/tmp/run-eval-<eval_id>.json` and the workspace
   artifacts under `./.latch-eval-runs/<eval_id>/` (relative to where you invoked
   the CLI; override with `--output-dir`; the CLI also prints the exact path):
   - `trajectory.json` — the agent's full trajectory
   - `agent_output.log` — stdout/stderr from the run
   - `agent_workspace/eval_answer.json` — what the agent submitted

   Summarize: **PASS / FAIL / NO GRADE**, the grader's reasoning and key metrics,
   and the agent's answer. Then **interpret against the eval's intent**: a
   competent agent should PASS a well-formed eval, and the trap should make a
   naive approach FAIL. Flag surprises — a PASS that should have been trapped, or
   a FAIL on the intended-correct path (often a too-tight tolerance, an ambiguous
   prompt, or a data issue) — as signals to revise the eval, not just an agent
   result.

## Caveats

- **Single grader only.** The runner grades the eval's single `grader`. A
  multi-grader `graders` list runs but produces no grade (the CLI warns, reports
  NO GRADE) — use the platform judge for those.
- **One replicate.** A single local run is a smoke test; pass/fail can vary.
- **Workspace recreated each run.** Copy out anything you want to keep before
  re-running.
