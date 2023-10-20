"use client";
import Image from "next/image";

export default function Complete({
  ts,
  phone,
  photo,
  status,
}: {
  ts: number;
  phone: string;
  photo: string;
  status: string;
}) {
  return (
    <div>
      <Image src={photo} width={200} height={200} alt="" />
    </div>
  );
}
