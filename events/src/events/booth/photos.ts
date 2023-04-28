import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import hash from "object-hash";
import { defineFunction } from "../tools";

const AWS_REGION = "us-west-2";
const BUCKET = "rootvc-photobooth";
const EXPECTED_COUNT = 4;

export default defineFunction(
  "Process a new set of photos",
  "dreambooth/booth.photos",
  async ({
    tools: { run, send, redis },
    event: {
      data: { phone, key },
    },
  }) => {
    const id = await run("calculate ID", () => hash({ phone }));

    const seq = await run(
      "increment sequence",
      async () => await redis.incr(`dataset/${id}`)
    );

    await run("upload to S3", async () => {
      const blob = (await redis.get(key))!;
      const s3 = new S3Client({ region: AWS_REGION });
      await s3.send(
        new PutObjectCommand({
          Bucket: BUCKET,
          Body: Buffer.from(blob, "base64"),
          Key: `dataset/${id}/${seq}.jpg`,
        })
      );
      await redis.del(key);
    });

    if (seq !== EXPECTED_COUNT) {
      return;
    }
    await run("delete sequence", async () => await redis.del(`dataset/${id}`));
    await run("store ts", async () => await redis.set(`ts/${id}`, Date.now()));
    await send("dreambooth/train.start", { id, phone });
    await send("dreambooth/sms.notify", { phone, key: "STARTED" });
  }
);
