import { createBashToolDefinition } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const DEFAULT_BASH_TIMEOUT_SECONDS = 300;

const bashSchema = Type.Object({
  command: Type.String({ description: "Bash command to execute" }),
  timeout: Type.Optional(
    Type.Number({ description: "Timeout in seconds (300s default)" }),
  ),
});

export default function(pi) {
  const bash = createBashToolDefinition(process.cwd());

  pi.registerTool({
    ...bash,
    parameters: bashSchema,

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const timeout = params.timeout > 0 ? params.timeout : DEFAULT_BASH_TIMEOUT_SECONDS;

      return bash.execute(toolCallId, { ...params, timeout }, signal, onUpdate, ctx);
    },
  });
}
