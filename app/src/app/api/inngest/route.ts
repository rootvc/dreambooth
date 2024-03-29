import { inngest } from "@/events/client";
import { serve } from "inngest/next";

import boothMotion from "@/events/booth/motion";
import boothPhotos from "@/events/booth/photos";
import smsNotify from "@/events/sms/notify";
import trainComplete from "@/events/train/complete";
import trainMonitor from "@/events/train/monitor";
import trainQueue from "@/events/train/queue";
import trainStart from "@/events/train/start";
import trainWarm from "@/events/train/warm";

export const { GET, POST, PUT } = serve(inngest, [
  trainStart,
  trainQueue,
  trainMonitor,
  trainComplete,
  trainWarm,
  smsNotify,
  boothPhotos,
  boothMotion,
]);
