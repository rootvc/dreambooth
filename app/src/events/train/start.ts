import got from "got";
import { API_ID, BUCKET, GIT_REMOTE_URL } from "../constants";
import { defineFunction } from "../tools";

const URL = `https://api.runpod.ai/v2/${API_ID}/run`;

export default defineFunction(
  "Start a training run",
  "dreambooth/train.start",
  async ({
    tools: { run, send },
    event: {
      data: { id, phone },
    },
  }) => {
    const payload = {
      input: {
        id,
        env: {
          WANDB_API_KEY: process.env.WANDB_API_KEY,
          WANDB_GIT_REMOTE_URL: GIT_REMOTE_URL,
          WANDB_GIT_COMMIT: "main",
          WANDB_NAME: `dreambooth-${id}`,
          DREAMBOOTH_ID: id,
          DREAMBOOTH_BUCKET: BUCKET,
        },
      },
    };
    const request = (await run(
      "request new run",
      async () =>
        await got
          .post(URL, {
            json: payload,
            headers: {
              Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
            },
          })
          .json()
    )) as { id: string };
    await send("dreambooth/train.monitor", {
      id,
      phone,
      requestId: request.id,
    });
  }
);
