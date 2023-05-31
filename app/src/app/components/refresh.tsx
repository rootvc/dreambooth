"use client";

import { useRouter } from "next/navigation";
import { useTimeout } from "usehooks-ts";

export default function Refresh({ seconds }: { seconds: number }) {
  const router = useRouter();
  const refreshData = () => {
    router.refresh();
  };
  useTimeout(refreshData, seconds * 1000);
  return null;
}
