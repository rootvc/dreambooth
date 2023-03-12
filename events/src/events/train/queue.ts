import { KEYS } from "../constants";
import { defineFunction } from "../tools";

export default defineFunction(
  "Queue a training run",
  "dreambooth/train.queue",
  async ({
    tools: { run, send, redis },
    event: {
      data: { id },
    },
  }) => {
    await run(
      "enqueue run",
      async () => await redis.rpush(KEYS.QUEUE.TRAINING, id)
    );
    while (true) {
      const peek = await run(
        "wait to start run",
        async () => await redis.lrange(KEYS.QUEUE.TRAINING, 0, 0)
      );
      if (peek[0] === id) {
        break;
      }
    }
    send("dreambooth/train.start", { id });
  }
);
