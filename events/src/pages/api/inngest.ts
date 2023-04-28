import { inngest } from "@/events/client";
import { serve } from "inngest/next";

import boothPhotos from "@/events/booth/photos";
import smsNotify from "@/events/sms/notify";
import trainComplete from "@/events/train/complete";
import trainMonitor from "@/events/train/monitor";
import trainQueue from "@/events/train/queue";
import trainStart from "@/events/train/start";

export default serve(inngest, [
  trainStart,
  trainQueue,
  trainMonitor,
  trainComplete,
  smsNotify,
  boothPhotos,
]);
