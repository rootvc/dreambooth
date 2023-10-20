import got, { RequestError } from "got";
import Redis from "ioredis";
import { STATUS } from "../constants";
import { defineFunction } from "../tools";

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
      data: { id, phone, resultUrl },
    },
  }) => {
    await sleep("3 seconds");
    const response = <StatusResponse>await run("get result", async () => {
      try {
        return await got.get(resultUrl).json();
      } catch (error: any) {
        console.error(<RequestError>error.response);
        throw error;
      }
    });
    console.warn(response);
    await run(
      "update status",
      async () => await redis.hset(`ts/${id}`, { status: STATUS.PRINTING })
    );
    await send("dreambooth/train.complete", { phone, id });
  },
  {
    // @ts-ignore
    onFailure: async ({
      error,
      event: {
        data: { id, phone },
      },
      step: { run, sendEvent },
    }) => {
      console.error(error);

      await run("update status", async () => {
        const redis = new Redis(process.env.REDIS_URL || "");
        await redis.hset(`ts/${id}`, { status: STATUS.FINISHED });
      });
      await sendEvent({
        name: "dreambooth/sms.notify",
        data: {
          phone,
          key: "ERRORED",
          error: error,
        },
      });
    },
  }
);
