import { Twilio } from "twilio";
import { SMS_COPY, TWILIO } from "../constants";
import { defineFunction } from "../tools";

const twilio = new Twilio(TWILIO.ACCOUNT_SID, TWILIO.AUTH_TOKEN);

export default defineFunction(
  "Notify a user",
  "dreambooth/sms.notify",
  async ({
    tools: { run },
    event: {
      data: { phone, key, mediaUrl, ...args },
    },
  }) => {
    const copy = await run("get copy", () =>
      SMS_COPY[key as keyof typeof SMS_COPY](args)
    );
    await run("send SMS", () => {
      twilio.messages.create({
        from: TWILIO.PHONE_NUMBER,
        to: phone,
        body: copy,
        mediaUrl,
      });
    });
  }
);
