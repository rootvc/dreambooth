import { defineFunction } from "../tools";

export default defineFunction(
  "Process a new set of photos",
  "dreambooth/booth.photos",
  async ({
    tools: { run, send, redis },
    event: {
      data: { phone, key },
    },
  }) => {
    await run("store ts", async () => await redis.set(`ts/${key}`, Date.now()));
    await send("dreambooth/train.start", { id: key, phone });
    await send("dreambooth/sms.notify", { phone, key: "STARTED" });
  }
);
