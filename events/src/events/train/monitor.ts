import got from "got";
import { defineFunction } from "../tools";

const URL = "https://api.dreambooth.app/v1/runs";

export default defineFunction(
  "Monitor a training run",
  "dreambooth/train.monitor",
  async ({
    tools: { run, send },
    event: {
      data: { id },
    },
  }) => {
    const response = await run("request new run", async () =>
     await got.post(URL, { json: { id } }).json();
    );
    await send("dreambooth")
  }
);
