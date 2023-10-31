import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import got, { RequestError } from "got";
import { NonRetriableError } from "inngest";
import { PRINT_SERVER } from "../constants";
import { defineFunction } from "../tools";

const AWS_REGION = "us-west-2";
const BUCKET = "rootvc-photobooth";

export default defineFunction(
  "Complete a training run",
  "dreambooth/train.complete",
  async ({
    tools: { run, send, redis, sleep },
    event: {
      data: { id, phone },
    },
  }) => {
    const mmsUrl = await run("get pre-signed MMS URL", async () => {
      try {
        return await getSignedUrl(
          <any>new S3Client({ region: AWS_REGION }),
          <any>new GetObjectCommand({
            Bucket: BUCKET,
            Key: `output/${id}/grid.jpg`,
          }),
          { expiresIn: 3600 }
        );
      } catch (error: any) {
        if (error.httpStatusCode === 404) {
          throw new NonRetriableError("Image not found", { cause: error });
        } else {
          throw error;
        }
      }
    });

    await run("store ts", async () => {
      await redis.hmset(`fin/${id}`, { ts: Date.now(), phone: phone });
      await redis.del(`ts/${id}`);
    });

    await send("dreambooth/sms.notify", {
      phone,
      id,
      key: "FINISHED",
      mmsUrl,
    });

    while (1) {
      try {
        let error = await run("send to printer server", async () => {
          try {
            const resp = await got.post(PRINT_SERVER, { json: { id } }).json();
            console.log(resp);
          } catch (error) {
            return error;
          }
        });
        if (error) throw error;
        break;
      } catch (error: any) {
        console.log(<RequestError>error.response);
        await sleep("1 second");
        continue;
      }
    }
  },
  {
    // @ts-ignore
    onFailure: async ({
      error,
      event: { id, phone },
      step: { sleep, sendEvent },
    }) => {
      console.log(error);
      await sleep("5 seconds");
      await sendEvent({ name: "dreambooth/train.start", data: { id, phone } });
    },
  }
);
