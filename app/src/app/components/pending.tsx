"use client";
import { DateTime } from "luxon";
import Image from "next/image";
import { CircularProgressBar } from "react-percentage-bar/dist";

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
  const progress = (now.diff(start).as("milliseconds") / total) * 100;
  return (
    <div className="flex items-center space-x-10">
      <Image
        src={photo}
        width={150}
        height={150}
        alt=""
        priority={true}
        style={{ borderRadius: "50%" }}
      />
      <CircularProgressBar
        progress={progress}
        duration={total}
        trackColor="rgba(14, 165, 233, 0.4)"
        color="#04E400"
        radius={75}
        shadow={true}
      />
    </div>
  );
}
