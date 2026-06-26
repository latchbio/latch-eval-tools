import { pathToFileURL } from "node:url";

const FALLBACK_BASH_TIMEOUT_SECONDS = 300;
const envBashTimeout = Number(process.env.PI_BASH_DEFAULT_TIMEOUT_SECONDS);
const DEFAULT_BASH_TIMEOUT_SECONDS =
  Number.isFinite(envBashTimeout) && envBashTimeout > 0
    ? envBashTimeout
    : FALLBACK_BASH_TIMEOUT_SECONDS;
const PI_BASH_TOOL_PATH =
  "/root/.local/lib/node_modules/@earendil-works/pi-coding-agent/dist/core/tools/bash.js";

const { createBashToolDefinition } = await import(
  pathToFileURL(PI_BASH_TOOL_PATH).href
);

export default function(pi) {
  const bash = createBashToolDefinition(process.cwd());
  bash.parameters.properties.timeout.description = `Timeout in seconds (${DEFAULT_BASH_TIMEOUT_SECONDS}s default)`;

  pi.registerTool({
    ...bash,

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const requestedTimeout = params.timeout;
      const timeout =
        typeof requestedTimeout === "number" && requestedTimeout > 0
          ? requestedTimeout
          : DEFAULT_BASH_TIMEOUT_SECONDS;

      return bash.execute(toolCallId, { ...params, timeout }, signal, onUpdate, ctx);
    },
  });
}
