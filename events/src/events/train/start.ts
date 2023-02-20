import got from "got";
import { defineFunction } from "../tools";

const URL = "https://api.dreambooth.app/v1/runs";

export default defineFunction(
  "Start a training run",
  "dreambooth/train.start",
  async ({ tools: { run }, event }) => {
    await run("request new run", async () => {
      await got.post(URL, { json: { event: event.id } });
    });
  }
);
