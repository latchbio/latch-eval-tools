export default function(pi) {
  pi.on("tool_call", (event) => {
    if (event.toolName !== "bash") return;

    const timeout = event.input.timeout;
    event.input.timeout =
      typeof timeout === "number" && timeout > 0
        ? timeout
        : 300;
  });
}
