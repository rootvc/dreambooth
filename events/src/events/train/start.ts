import got from "got";
import { defineFunction } from "../tools";

const URL = "https://api.dreambooth.app/v1/runs";

export default defineFunction(
  "Start a training run",
  "dreambooth/train.start",
  async ({
    tools: { run, send },
    event: {
      data: { id },
    },
  }) => {
    const response = (await run(
      "request new run",
      async () => await got.post(URL, { json: { id } }).json()
    )) as { name: string };
    await send("dreambooth/train.monitor", { id, name: response.name });
  }
);
