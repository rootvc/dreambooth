import os
import subprocess
import tempfile
from pathlib import Path

DOMAIN = "print.root.vc"


def main():
    os.chdir(Path(__file__).parent.parent)
    subprocess.run(
        ["/Users/rootventures/.asdf/shims/pip", "install", "-r", "requirements.txt"]
    )

    from print_server.api.index import PRINTER_NAME

    printers = [
        printer.split(" ")[0]
        for printer in subprocess.run(["lpstat", "-a"], capture_output=True)
        .stdout.decode("utf-8")
        .splitlines()
    ]

    if PRINTER_NAME not in printers:
        raise RuntimeError(f"Printer {PRINTER_NAME} not found")

    with tempfile.TemporaryDirectory():
        procs = []

        def _launch(command):
            procs.append(subprocess.Popen(command))

        _launch(["/Users/rootventures/.asdf/shims/sanic", "print_server.api.index:app"])
        _launch(["/opt/homebrew/bin/ngrok", "http", "8000", "--domain", DOMAIN])

        try:
            for proc in procs:
                proc.wait()
        finally:
            for proc in procs:
                proc.kill()


if __name__ == "__main__":
    main()
