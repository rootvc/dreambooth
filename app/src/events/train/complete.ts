import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import got from "got";
import { PRINT_SERVER } from "../constants";
import { defineFunction } from "../tools";

const AWS_REGION = "us-west-2";
const BUCKET = "rootvc-photobooth";

export default defineFunction(
  "Complete a training run",
  "dreambooth/train.complete",
  async ({
    tools: { run, send, redis },
    event: {
      data: { id, phone },
    },
  }) => {
    await run("store ts", async () => {
      await redis.hmset(`fin/${id}`, { ts: Date.now(), phone: phone });
      await redis.del(`ts/${id}`);
    });
    await run(
      "send to printer server",
      async () => await got.post(PRINT_SERVER, { json: { id } }).json()
    );
    const mediaUrl = await run(
      "get pre-signed URL",
      async () =>
        await getSignedUrl(
          <any>new S3Client({ region: AWS_REGION }),
          <any>new GetObjectCommand({
            Bucket: BUCKET,
            Key: `output/${id}/grid.png`,
          }),
          { expiresIn: 3600 }
        )
    );
    await send("dreambooth/sms.notify", {
      phone,
      id,
      key: "FINISHED",
      mediaUrl,
    });
  }
);
