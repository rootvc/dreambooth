import os

import runpod
from runpod.endpoint import Job

from dreambooth_old.train.trainers.base import BaseTrainer

runpod.api_key = os.environ["RUNPOD_AI_API_KEY"]


class RunpodTrainer(BaseTrainer):
    API_ID = "cfxm61b8e7uzcb"

    def __init__(self, id: str) -> None:
        super().__init__(id)
        self.endpoint = runpod.Endpoint(self.API_ID)

    async def run(self) -> Job:
        return self.endpoint.run({"env": self.env, "id": self.id})

    async def run_and_wait(self):
        return (await self.run()).output()

    async def run_and_report(self):
        return (await self.run()).job_id
