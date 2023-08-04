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
    const mediaUrl = await run("get pre-signed URL", async () => {
      try {
        return await getSignedUrl(
          <any>new S3Client({ region: AWS_REGION }),
          <any>new GetObjectCommand({
            Bucket: BUCKET,
            Key: `output/${id}/grid.png`,
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
      mediaUrl,
    });

    while (true) {
      try {
        await run(
          "send to printer server",
          async () => await got.post(PRINT_SERVER, { json: { id } }).json()
        );
      } catch (error: any) {
        console.error(<RequestError>error.response);
        await sleep("30 seconds");
        continue;
      }
      break;
    }
  },
  {
    // @ts-ignore
    onFailure: async ({
      error,
      event: { id, phone },
      step: { sleep, send },
    }) => {
      console.error(error);
      await sleep("10 seconds");
      await send("dreambooth/train.start", { id, phone });
    },
  }
);
