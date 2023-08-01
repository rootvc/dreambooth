import got, { RequestError } from "got";
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
      const response = <StatusResponse>(
        await run("get training job status", async () => {
          try {
            return await got
              .get(`${URL}/${requestId}`, {
                headers: {
                  Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
                },
              })
              .json();
          } catch (error: any) {
            console.error(<RequestError>error.response);
            throw error;
          }
        })
      );
      if (["COMPLETED", "FAILED"].includes(response.status)) {
        break;
      }
      await sleep("3 seconds");
    }

    await send("dreambooth/train.complete", { phone, id });
  }
);
