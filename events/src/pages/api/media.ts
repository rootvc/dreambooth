import { Fields, File, Files, IncomingForm } from "formidable";
import { readFile } from "fs";
import { Inngest } from "inngest";
import { NextApiRequest, NextApiResponse } from "next";
import pify from "pify";

const inngest = new Inngest({ name: "rootvc-fn-media" });

const media = async (request: NextApiRequest, response: NextApiResponse) => {
  const form = new IncomingForm();
  const parse = pify(form.parse.bind(form), { multiArgs: true }) as (
    req: NextApiRequest
  ) => Promise<[Fields, Files]>;
  const [fields, files] = await parse(request);
  const media = files.media as File;

  inngest.send("dreambooth/booth.photos", {
    data: {
      email: fields.email as string,
      phone: fields.phone as string,
      blob: (await pify(readFile)(media.filepath)).toString("base64"),
    },
  });
};

export default media;
