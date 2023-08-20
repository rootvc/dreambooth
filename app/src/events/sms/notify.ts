import { OpenAIChatMessage, OpenAIChatModel, generateText } from "modelfusion";
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
    if (!phone) return;
    const copy = await run("get copy", () =>
      SMS_COPY[key as keyof typeof SMS_COPY](args)
    );
    const { text: body } = await run(
      "make copy unique",
      async () =>
        await generateText(
          new OpenAIChatModel({ model: "gpt-3.5-turbo", temperature: 0.8 }),
          [
            OpenAIChatMessage.system(
              "Take the following SMS message and modify it to be unique, " +
                "so that it does not mistakenly get filtered out as spam. " +
                "Make sure it is short enough to fit in a single SMS message. " +
                "Strip out any links."
            ),
            OpenAIChatMessage.user(copy),
          ]
        )
    );
    await run("send SMS", () =>
      twilio.messages.create({
        from: TWILIO.PHONE_NUMBER,
        to: phone,
        body,
        mediaUrl: mediaUrl ? [mediaUrl] : undefined,
      })
    );
  }
);
