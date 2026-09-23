export default function(pi) {
  pi.registerFlag("max-turns", { description: "Stop after N turns", type: "string", default: "0" });

  let turns = 0;
  let stopped = false;

  pi.on("turn_end", async (event) => {
    if (event.message.stopReason !== "error") {
      turns += 1;
    }
  });

  pi.on("before_provider_request", async (event, ctx) => {
    const maxTurns = Number(pi.getFlag("max-turns"));
    if (stopped || maxTurns <= 0 || turns < maxTurns) {
      return;
    }
    stopped = true;
    ctx.abort();
  });
}
