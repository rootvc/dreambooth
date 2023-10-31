import { OpenAIChatMessage, OpenAIChatModel, generateText } from "modelfusion";
import Telnyx from "telnyx/lib/telnyx";
import { SMS_COPY } from "../constants";
import { defineFunction } from "../tools";

const telnyx = Telnyx(process.env.TELNYX_API_KEY);

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
    const { text } = await run(
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
    await run(
      "send SMS",
      async () =>
        await telnyx.messages.create({
          from: process.env.TELNYX_PHONE_NUMBER,
          messaging_profile_id: process.env.TELNYX_MESSAGING_PROFILE_ID,
          to: phone,
          text,
          media_urls: mediaUrl ? [mediaUrl] : undefined,
          type: mediaUrl ? "MMS" : "SMS",
        })
    );
  }
);
