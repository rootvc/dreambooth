import { GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { BUCKET, redis, s3client } from "../helpers";
import Pending from "./pending";

export default async function Session({ id }: { id: string }) {
  let { ts, phone } = await redis.hgetall(id);
  let key = id.split("/")[1];
  let photo = await getSignedUrl(
    s3client as any,
    new GetObjectCommand({
      Bucket: BUCKET,
      Key: `output/${id}/grid.png`,
    }) as any,
    { expiresIn: 3600 }
  );
  return <Pending ts={Number(ts)} phone={phone} photo={photo} />;
}
