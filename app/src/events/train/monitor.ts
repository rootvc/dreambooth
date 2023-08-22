import got, { RequestError } from "got";
import { API_ID, STATUS } from "../constants";
import { defineFunction } from "../tools";

const URL = `https://api.runpod.ai/v2/${API_ID}/status`;

type StatusResponse = {
  delayTime: number;
  id: string;
  status: "IN_PROGRESS" | "COMPLETED" | "FAILED" | "IN_QUEUE";
  error?: string;
};

export default defineFunction(
  "Monitor a training run",
  "dreambooth/train.monitor",
  async ({
    tools: { run, send, sleep, redis },
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
      if (response.status === "IN_PROGRESS") {
        await run(
          "update status",
          async () => await redis.hset(`ts/${id}`, { status: STATUS.TRAINING })
        );
      } else if (response.status === "COMPLETED") {
        await run(
          "update status",
          async () => await redis.hset(`ts/${id}`, { status: STATUS.PRINTING })
        );
        await send("dreambooth/train.complete", { phone, id });
        break;
      } else if (response.status === "FAILED") {
        console.error(response.error);
        await run(
          "update status",
          async () => await redis.hset(`ts/${id}`, { status: STATUS.FINISHED })
        );
        await send("dreambooth/sms.notify", {
          phone,
          key: "ERRORED",
          error: response.error,
        });
        break;
      }
      await sleep("3 seconds");
    }

    await send("dreambooth/train.complete", { phone, id });
  }
);
