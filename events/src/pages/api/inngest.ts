import { inngest } from "@/events/client";
import pythonExecute from "@/events/python/execute";
import { serve } from "inngest/next";

export default serve(inngest, [pythonExecute]);
