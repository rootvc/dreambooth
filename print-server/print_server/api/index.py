import asyncio
import subprocess
import tempfile
from pathlib import Path

from cloudpathlib import S3Path
from sanic import Sanic, response

PRINTER_NAME = "Send_to_Preview___yasyf_"  # "QW410_4x6___dnpwcm"
BUCKET = "rootvc-photobooth"

app = Sanic(name="print-server")


def _download_from_s3(id):
    print(f"Downloading {id} from S3")
    path = Path(tempfile.mkdtemp()) / f"{id}.png"
    S3Path(f"s3://{BUCKET}/output/{id}/grid.png").download_to(path)
    return path


def _send_to_printer(id):
    print(f"Sending {id} to printer")
    subprocess.run(["lpr", "-P", PRINTER_NAME, _download_from_s3(id)], check=True)


@app.route("/", methods=["POST"])
async def start(request):
    id = request.json["id"]
    await asyncio.to_thread(_send_to_printer, id)
    return response.json({"status": "ok"})
