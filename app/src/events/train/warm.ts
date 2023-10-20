import got from "got";
import { defineFunction } from "../tools";
import { modalUrl } from "../utils";

export default defineFunction(
  "Trigger server warmup",
  "dreambooth/train.warm",
  async ({}) => await got.post(modalUrl("warm"))
);
