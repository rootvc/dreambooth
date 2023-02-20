import site
from pathlib import Path

src = Path(__file__).absolute.parent.parent.parent
site.addsitedir(src)

from sanic import Sanic

from dreambooth.train import trainer

app = Sanic()


@app.route("/api/start", methods=["POST"])
async def start(request):
    id = request.json["id"]
    trainer.run(id)
