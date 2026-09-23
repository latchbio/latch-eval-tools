export default function(pi) {
  pi.registerFlag("max-turns", { description: "Stop after N turns", type: "string" });

  let turns = 0;

  pi.on("turn_end", (event) => {
    if (event.message.stopReason !== "error") {
      turns += 1;
    }
  });

  pi.on("before_provider_request", (_event, ctx) => {
    if (turns >= Number(pi.getFlag("max-turns"))) {
      ctx.abort();
    }
  });
}
