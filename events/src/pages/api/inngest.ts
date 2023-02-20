import { inngest } from "@/events/client";
import trainStart from "@/events/train/start";
import { serve } from "inngest/next";

export default serve(inngest, [trainStart]);
