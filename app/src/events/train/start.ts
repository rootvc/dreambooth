import got from "got";
import { STATUS } from "../constants";
import { defineFunction } from "../tools";
import { modalUrl } from "../utils";

export default defineFunction(
  "Start a training run",
  "dreambooth/train.start",
  async ({
    tools: { run, send, redis },
    event: {
      data: { id, phone },
    },
  }) => {
    await run(
      "update status",
      async () => await redis.hset(`ts/${id}`, { status: STATUS.QUEUED })
    );
    const request = (await run(
      "request new run",
      async () =>
        await got.post(modalUrl("dream"), { searchParams: { id } }).json()
    )) as { result_url: string };
    await run(
      "update status",
      async () => await redis.hset(`ts/${id}`, { status: STATUS.STARTED })
    );
    await send("dreambooth/train.monitor", {
      id,
      phone,
      resultUrl: request.result_url,
    });
  }
);
