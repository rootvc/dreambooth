import { EventSchemas, Inngest } from "inngest";

type PythonExecute = {
  name: "dreambooth/python.execute";
  data: { code: string };
};

type Rest = {
  name: string;
  data: any;
};

export type Events = PythonExecute | Rest;

export const inngest = new Inngest({
  name: "Dreambooth Events",
  schemas: new EventSchemas().fromUnion<Events>(),
});
