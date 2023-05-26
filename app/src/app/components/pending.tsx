"use client";
import { DateTime } from "luxon";
import Image from "next/image";
import { LinearProgressBar } from "react-percentage-bar/dist";

export default function Pending({
  ts,
  phone,
  photo,
}: {
  ts: number;
  phone: string;
  photo: string;
}) {
  const start = DateTime.fromMillis(ts);
  const end = start.plus({ minutes: 3 });
  const total = end.diff(start).as("milliseconds");
  const now = DateTime.now();
  const progress = now.diff(start).as("milliseconds") / total;
  return (
    <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
      <Image src={photo} width={100} height={100} alt="" />
      {phone}
      <LinearProgressBar progress={progress} duration={total} />
    </div>
  );
}
