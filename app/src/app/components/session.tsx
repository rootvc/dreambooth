import { GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import util from "node:util";
import { BUCKET, redis, s3client } from "../helpers";

export default async function Session({
  id,
  keyTemplate,
  el: El,
}: {
  id: string;
  keyTemplate: string;
  el: React.FunctionComponent<{
    ts: number;
    phone: string;
    photo: string;
    status: string;
  }>;
}) {
  let { ts, phone, status } = await redis.hgetall(id);
  let key = id.split("/")[1];
  let photo = await getSignedUrl(
    s3client as any,
    new GetObjectCommand({
      Bucket: BUCKET,
      Key: util.format(keyTemplate, key),
    }) as any,
    { expiresIn: 3600 }
  );
  return <El ts={Number(ts)} phone={phone} photo={photo} status={status} />;
}
