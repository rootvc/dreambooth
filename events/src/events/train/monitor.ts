import {
  DescribeTrainingJobCommand,
  SageMakerClient,
  UpdateTrainingJobCommand,
} from "@aws-sdk/client-sagemaker";
import { defineFunction } from "../tools";

const client = new SageMakerClient({ region: process.env.AWS_REGION });

export default defineFunction(
  "Monitor a training run",
  "dreambooth/train.monitor",
  async ({
    tools: { run, send, redis, sleep },
    event: {
      data: { id, name },
    },
  }) => {
    while (true) {
      const response = await run(
        "get training job status",
        async () =>
          await client.send(
            new DescribeTrainingJobCommand({ TrainingJobName: name })
          )
      );
      if (
        ["Completed", "Failed", "Stopped"].includes(response.TrainingJobStatus)
      ) {
        break;
      }
      await sleep("1 minute");
    }

    const length = await run("get queue length", async () => redis.llen(id));
    if (length === 0) {
      await run(
        "update training job",
        async () =>
          await client.send(
            new UpdateTrainingJobCommand({
              TrainingJobName: name,
              ResourceConfig: {
                KeepAlivePeriodInSeconds: 300,
              },
            })
          )
      );
    }

    await send("dreambooth/train.complete", { id });
  }
);
