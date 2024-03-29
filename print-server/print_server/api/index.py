import asyncio
import subprocess
from pathlib import Path
import time

from cloudpathlib import S3Path
from sanic import Sanic, response

PRINTER_NAME = "Dai_Nippon_Printing_DP_QW410"  # "Send_to_Preview___yasyf_"
BUCKET = "rootvc-photobooth"

app = Sanic(name="print-server")

photos = Path.home() / ".dreambooth" / "photos"
photos.mkdir(parents=True, exist_ok=True)


def _download_from_s3(id):
    print(f"Downloading {id} from S3")
    path = photos / f"{id}_{int(time.time())}.png"
    S3Path(f"s3://{BUCKET}/output/{id}/grid.png").download_to(path)
    return str(path)


def _send_to_printer(id):
    print(f"Sending {id} to printer")
    subprocess.run(
        ["lpr", "-P", PRINTER_NAME, "-o", "media='dnp4x4'", _download_from_s3(id)],
        check=True,
    )


@app.route("/", methods=["POST"])
async def start(request):
    id = request.json["id"]
    await asyncio.to_thread(_send_to_printer, id)
    return response.json({"status": "ok"})
