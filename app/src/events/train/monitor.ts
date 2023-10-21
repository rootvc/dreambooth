import * as Got from "got";
import got, { HTTPError, RequestError, TimeoutError } from "got";
import Redis from "ioredis";
import { STATUS } from "../constants";
import { defineFunction } from "../tools";

type StatusResponse = {
  delayTime: number;
  id: string;
  status: "IN_PROGRESS" | "COMPLETED" | "FAILED" | "IN_QUEUE";
  error?: string;
};

type SerializedError = { name: keyof typeof Got };

const restoreError = (obj: SerializedError): RequestError => {
  const err = Object.create(Got[obj.name].prototype);
  Object.assign(err, obj);
  return err;
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
    while (1) {
      const response = <
        | { type: "response"; value: StatusResponse }
        | { type: "error"; value: SerializedError }
      >await run("get result", async () => {
        try {
          const resp = await got
            .get(resultUrl, {
              timeout: {
                request: 1500,
              },
            })
            .json();
          return { type: "response", value: resp };
        } catch (error: any) {
          console.error(<RequestError>error.response);
          return { type: "error", value: error };
        }
      });

      if (response.type == "response") {
        console.log(response.value);
        await run(
          "update status",
          async () => await redis.hset(`ts/${id}`, { status: STATUS.PRINTING })
        );
        break;
      }

      const err = restoreError(response.value);

      if (err instanceof TimeoutError) {
        await sleep("3 seconds");
        continue;
      } else if (
        err instanceof HTTPError &&
        (err.response.statusCode == 410 || err.response.statusCode == 404)
      ) {
        await run(
          "update status",
          async () => await redis.hset(`ts/${id}`, { status: STATUS.FINISHED })
        );
        break;
      } else {
        await run("update status", async () => {
          const redis = new Redis(process.env.REDIS_URL || "");
          await redis.hset(`ts/${id}`, { status: STATUS.FINISHED });
        });
        await send("dreambooth/sms.notify", {
          phone,
          key: "ERRORED",
          error: response,
        });
        return;
      }
    }
    await send("dreambooth/train.complete", { phone, id });
  }
);
