import site
from pathlib import Path

from sanic import Sanic
from sanic.response import json

app = Sanic()

src = Path(__file__).absolute.parent.parent.parent
site.addsitedir(src)


@app.route("/api/inngest")
async def inngest(request, path=""):
    return json({"hello": path})
