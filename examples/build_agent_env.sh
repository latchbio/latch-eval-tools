#!/usr/bin/env bash
# Build the agent_env image (now including the grok CLI) for the eval platform.
#
# MUST be linux/amd64: pyscipopt (pulled transitively via the scipy stack) has
# x86_64 manylinux wheels only, so an arm64 build source-compiles it and dies at
# `uv sync` needing SCIP headers. On amd64 it's a wheel and just works.
#
# CAVEAT: run this on a NATIVE amd64 host / CI. Building it on Apple Silicon via
# qemu emulation does NOT work either -- the Claude Code installer (Bun-based)
# segfaults under qemu (`uncaught target signal 11`) before uv sync is reached.
# That's an emulation artifact, not a Dockerfile bug; native amd64 builds fine.
#
# Usage:
#   bash examples/build_agent_env.sh [tag]
# Default tag matches DEFAULT_DOCKER_IMAGE so the harness picks it up directly.
set -euo pipefail

TAG="${1:-public.ecr.aws/p5z7v3z8/benchmark_agent:latest}"

docker build --platform=linux/amd64 \
  -f agent_env/Dockerfile \
  -t "$TAG" \
  agent_env/

echo "Built $TAG (linux/amd64). Verify grok is present:"
echo "  docker run --rm --platform=linux/amd64 $TAG grok --version"
