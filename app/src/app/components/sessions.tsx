import { redis } from "../helpers";
import Session from "./session";

export default async function Sessions() {
  await redis.hmset(`ts/test`, { ts: Date.now(), phone: "6176316733" });
  await redis.hmset(`fin/xxx`, {
    ts: Date.now() - 1 * 60 * 60 * 1000,
    phone: "5555555555",
  });

  let sessions = await redis.keys("ts/*");
  /* @ts-expect-error Async Server Component */
  return sessions.map((s) => <Session key={s} id={s} />);
}
