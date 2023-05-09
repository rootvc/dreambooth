import { randomUUID } from "crypto";
import { Fields, File, Files, IncomingForm } from "formidable";
import { readFile } from "fs";
import { Inngest } from "inngest";
import { Redis } from "ioredis";
import { NextApiRequest, NextApiResponse } from "next";
import pify from "pify";

export const config = {
  api: {
    bodyParser: false,
  },
};

const redisClient = new Redis(process.env.REDIS_URL || "");

const inngest = new Inngest({
  name: "rootvc-fn-media",
  inngestBaseUrl: process.env.INNGEST_BASE_URL,
});

const media = async (request: NextApiRequest, response: NextApiResponse) => {
  const form = new IncomingForm();
  const parse = pify(form.parse.bind(form), { multiArgs: true }) as (
    req: NextApiRequest
  ) => Promise<[Fields, Files]>;
  const [fields, files] = await parse(request);
  const media = files.media as File;

  const key = `media/${randomUUID()}`;
  await redisClient.setex(
    key,
    60 * 5,
    (await pify(readFile)(media.filepath)).toString("base64")
  );

  await inngest.send("dreambooth/booth.photos", {
    data: {
      phone: fields.name as string,
      key: key,
    },
  });

  response.status(200).json({ status: true });
};

export default media;
