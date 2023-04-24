import asyncio
import subprocess

from sanic import Sanic, response

PRINTER_NAME = "dreambooth"

app = Sanic()


def _send_to_printer(id):
    print("Sending to printer")
    subprocess.run(["lpr", "-P", PRINTER_NAME])


@app.route("/print", methods=["POST"])
async def start(request):
    request.json["id"]
    asyncio.to_thread()
    return response.json(resp)
