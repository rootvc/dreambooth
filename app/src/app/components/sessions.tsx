import _ from "lodash";
import { redis } from "../helpers";
import Complete from "./complete";
import Pending from "./pending";
import Session from "./session";

export const revalidate = 0;

export default async function Sessions() {
  let sessions = await redis.keys("ts/*");
  let completed = _.shuffle(_.take(await redis.keys("fin/*"), 5));
  return (
    <div className="flex flex-col space-y-10">
      <div className="grid grid-cols-1 gap-7 justify-items-center">
        {sessions.map((s) => (
          /* @ts-expect-error Async Server Component */
          <Session key={s} id={s} el={Pending} keyTemplate="dataset/%s/1.jpg" />
        ))}
      </div>
      <div className="flex justify-around space-x-1">
        {completed.map((s) => (
          /* @ts-expect-error Async Server Component */
          <Session
            key={s}
            id={s}
            el={Complete}
            keyTemplate="output/%s/grid.png"
          />
        ))}
      </div>
    </div>
  );
}
