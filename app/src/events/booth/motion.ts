import { defineFunction } from "../tools";

export default defineFunction(
  "Trigger warmup on booth motion",
  "dreambooth/booth.motion",
  async ({ tools: { send } }) => {
    await send("dreambooth/train.warm", {});
  }
);
