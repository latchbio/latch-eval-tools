export default function(pi) {
  pi.on("before_provider_request", (event, ctx) => {
    const payload = event.payload;
    if (!payload.tools?.length) {
      return;
    }

    switch (ctx.model.api) {
      case "anthropic-messages":
        return {
          ...payload,
          tool_choice: { type: "auto", ...payload.tool_choice, disable_parallel_tool_use: true },
        };
      case "openai-completions":
      case "openai-responses":
        return { ...payload, parallel_tool_calls: false };
    }
  });
}
