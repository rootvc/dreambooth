import site
from pathlib import Path

src = Path(__file__).absolute.parent.parent.parent
site.addsitedir(src)

from sanic import Sanic, response

from dreambooth.train.trainer import TrainJob

app = Sanic()


@app.route("/api/start", methods=["POST"])
async def start(request):
    id = request.json["id"]
    resp = await TrainJob(id).run_and_report(id)
    return response.json(resp)
