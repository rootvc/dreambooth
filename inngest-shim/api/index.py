import asyncio
import subprocess
import tempfile

from cloudpathlib import S3Path
from sanic import Sanic, response

PRINTER_NAME = "dreambooth"
BUCKET = "rootvc-photobooth"

app = Sanic()


def _download_from_s3(id):
    print(f"Downloading {id} from S3")
    tmpdir = tempfile.mkdtemp()
    S3Path(f"{BUCKET}/output/{id}").download_to(tmpdir)
    return tmpdir


def _send_to_printer(id):
    print(f"Sending {id} to printer")
    subprocess.run(["lpr", "-P", PRINTER_NAME])


@app.route("/print", methods=["POST"])
async def start(request):
    request.json["id"]
    asyncio.to_thread()
    return response.json(resp)
