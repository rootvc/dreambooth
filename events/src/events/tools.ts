import { FunctionOptions, Inngest } from "inngest";
import { Events as GenEvents } from "inngest-events";
import { createStepTools } from "inngest/components/InngestStepTools";
import _ from "lodash";
import hash from "object-hash";
import { inngest } from "./client";

type ParametersExceptFirst<F> = F extends (arg0: any, ...rest: infer R) => any
  ? R
  : never;
type PromiseOrVal<T> = T | Promise<T>;
type PreserveNonNullable<T> = T extends NonNullable<infer U> ? U : T;

type Events = GenEvents<{
  "dreambooth/python.execute": { name: string; data: any };
}>;
type Event = keyof Events;
type Tools<E extends Event> = ReturnType<typeof createStepTools<Events, E>>[0];

const _run = async <E extends Event, T>(
  tools: Tools<E>,
  name: string,
  fn: () => PromiseOrVal<T>
): Promise<T> =>
  tools.run<() => Promise<PreserveNonNullable<T>>>(name, async () => {
    const result = await fn();
    return result as PreserveNonNullable<T>;
  }) as T;

const _log = async <E extends Event, T>(tools: Tools<E>, obj: T): void => {
  const h = hash(obj as any, { unorderedArrays: true });
  await _run<E, T>(tools, `log:${h}`, () => obj);
};

const _get = async <E extends Event, T, A extends string>(
  tools: Tools<E>,
  attr: A,
  fn: () => PromiseOrVal<T>
): Promise<{ [a in A]: T }> => {
  const result = await _run<E, T>(tools, "get " + attr, fn);
  return { [attr]: result } as { [a in A]: T };
};

const _collect = async <
  E extends Event,
  C extends { [k: string]: any }[],
  O extends object
>(
  tools: Tools<E>,
  computed: C,
  extra: O = {} as O
) =>
  (await tools.run("collect attrs", () =>
    _.assign(
      { ...extra },
      ..._.filter(computed, (o) => !_.isEmpty(_.pickBy(o, Boolean)))
    )
  )) as O & { [K in keyof C[number]]: C[number][K] };

const _send = async <S extends Event, D extends Event>(
  inngest: Inngest<Events>,
  tools: Tools<S>,
  dest: D,
  body: Events[D]["data"],
  user: {} = {}
) => {
  const h = hash(body, { unorderedArrays: true });
  await _run<S, any>(tools, `send ${dest}:${h}`, async () => {
    await inngest.send<any>({ name: dest, data: body, user: user });
  });
};

const partialRun = <A, F extends (...args: any) => R, R>(arg0: A, fn: F) => {
  return (...args: ParametersExceptFirst<F>) =>
    fn(arg0, ...args) as ReturnType<F>;
};

const partial = <E extends Event, T, F extends (...args: any) => T>(
  tools: Tools<E>,
  fn: F
) => {
  const f = partialRun<Tools<E>, F, T>(tools, fn);
  const wrapped = (...args: Parameters<typeof f>) => {
    const result = f(...args);
    return result as ReturnType<typeof f>;
  };
  return wrapped;
};

const getTools = <E extends Event>(
  inngest: Inngest<Events>,
  tools: Tools<E>
) => {
  const run = async <T>(...args: ParametersExceptFirst<typeof _run<E, T>>) =>
    _run<E, T>(tools, ...args);
  const get = async <T, A extends string>(
    ...args: ParametersExceptFirst<typeof _get<E, T, A>>
  ) => _get<E, T, A>(tools, ...args);
  return {
    run,
    get,
    log: partial(tools, _log),
    collect: partial(tools, _collect),
    send: partialRun(tools, partialRun(inngest, _send)),
  };
};

export const defineFunction = <E extends Event>(
  name: string,
  event: E,
  fn: (arg: {
    event: Events[E];
    tools: Omit<Tools<E>, "run"> & ReturnType<typeof getTools<E>>;
  }) => any,
  opts: Omit<FunctionOptions, "name"> = {}
) => {
  return inngest.createFunction<
    { event: E },
    { name: typeof name } & typeof opts
  >(
    { name, ...opts },
    { event },
    ({ event, step }: { event: Events[E]; step: Tools<E> }) => {
      const { run, ...rest } = step;
      return fn({ event, tools: { ...getTools<E>(inngest, step), ...rest } });
    }
  );
};