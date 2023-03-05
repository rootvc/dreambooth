import { SMS_COPY } from "../constants";
import { defineFunction } from "../tools";

export default defineFunction(
  "Notify a user",
  "dreambooth/sms.notify",
  async ({
    tools: { run },
    event: {
      data: { phone, key },
    },
  }) => {
    const copy = await run(
      "get copy",
      () => SMS_COPY[key as keyof typeof SMS_COPY]
    );
    await run("send SMS", () => {
      // import twilio
    });
  }
);
