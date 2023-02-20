import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import hash from "object-hash";
import { v4 as uuid } from "uuid";
import { defineFunction } from "../tools";

const AWS_REGION = "us-west-2";
const BUCKET = "rootvc-photobooth";

export default defineFunction(
  "Process a new set of photos",
  "dreambooth/booth.photos",
  async ({
    tools: { run },
    event: {
      data: { email, blob },
    },
  }) => {
    const key = await run("calculate ID", () => {
      const id = hash({ email });
      const seq = uuid();
      return `dataset/${id}/${seq}.jpg`;
    });

    await run("upload to S3", async () => {
      const s3 = new S3Client({ region: AWS_REGION });
      await s3.send(
        new PutObjectCommand({
          Bucket: BUCKET,
          Body: blob,
          Key: key,
        })
      );
    });
  }
);
