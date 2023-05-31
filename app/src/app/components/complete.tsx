"use client";
import Image from "next/image";

export default function Complete({ photo }: { photo: string }) {
  return (
    <div>
      <Image src={photo} width={200} height={200} alt="" />
    </div>
  );
}
