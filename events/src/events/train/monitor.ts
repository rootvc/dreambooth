import got from "got";
import { API_ID } from "../constants";
import { defineFunction } from "../tools";

const URL = `https://api.runpod.ai/v2/${API_ID}/status`;

type StatusResponse = {
  delayTime: number;
  id: string;
  status: "IN_PROGRESS" | "COMPLETED" | "FAILED" | "IN_QUEUE";
};

export default defineFunction(
  "Monitor a training run",
  "dreambooth/train.monitor",
  async ({
    tools: { run, send, sleep },
    event: {
      data: { id, phone, requestId },
    },
  }) => {
    while (true) {
      const response = (await run(
        "get training job status",
        async () =>
          await got
            .get(`${URL}/${requestId}`, {
              headers: {
                Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
              },
            })
            .json()
      )) as StatusResponse;
      if (["COMPLETED", "FAILED"].includes(response.status)) {
        break;
      }
      await sleep("1 second");
    }

    await send("dreambooth/train.complete", { phone, id });
  }
);
