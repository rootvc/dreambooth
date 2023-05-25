import got from "got";
import { PRINT_SERVER } from "../constants";
import { defineFunction } from "../tools";

export default defineFunction(
  "Complete a training run",
  "dreambooth/train.complete",
  async ({
    tools: { run, send, sleep },
    event: {
      data: { id, phone },
    },
  }) => {
    await run(
      "send to printer server",
      async () => await got.post(PRINT_SERVER, { json: { id } }).json()
    );
    await send("dreambooth/sms.notify", { phone, id, key: "FINISHED" });
  }
);
