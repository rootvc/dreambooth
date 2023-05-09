import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import child_process from "node:child_process";
import util from "node:util";
import hash from "object-hash";
import sharp from "sharp";
import { defineFunction } from "../tools";

const exec = util.promisify(child_process.exec);

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

    await run("split and upload images", async () => {
      const image = sharp(
        Buffer.from((await redis.get(key))!, "base64")
      ).trim();
      const { data, info } = await image.toBuffer({ resolveWithObject: true });
      const height = Math.trunc(info.height / EXPECTED_COUNT);

      const s3 = new S3Client({ region: AWS_REGION });

      for (let i = 0; i < EXPECTED_COUNT; i++) {
        console.log(info, {
          left: 0,
          top: i * height,
          width: info.width,
          height: height,
        });
        await s3.send(
          new PutObjectCommand({
            Bucket: BUCKET,
            Body: await sharp(data)
              .extract({
                left: 0,
                top: i * height,
                width: info.width,
                height: height,
              })
              .toBuffer(),
            Key: `dataset/${id}/${i}.jpg`,
          })
        );
      }
    });

    await run("delete from redis", async () => await redis.del(key));
    await run("store ts", async () => await redis.set(`ts/${id}`, Date.now()));
    await send("dreambooth/train.start", { id, phone });
    await send("dreambooth/sms.notify", { phone, key: "STARTED" });
  }
);
