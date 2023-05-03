import os
import subprocess
import tempfile
from pathlib import Path

DOMAIN = "print.root.vc"


def main():
    os.chdir(Path(__file__).parent.parent)
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

    from print_server.api.index import PRINTER_NAME

    printers = [
        printer.split(" ")[0]
        for printer in subprocess.run(
            ["lpstat", "-a"], capture_output=True
        ).stdout.decode("utf-8")
    ]

    if PRINTER_NAME[0] not in printers:
        raise RuntimeError(f"Printer {PRINTER_NAME[0]} not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        procs = []

        def _launch(command):
            fd = open((Path(tmpdir) / command[0]).with_suffix(".log"), "wb")
            procs.append(subprocess.Popen(command, stdout=fd, stderr=fd))

        _launch(["sanic", "print_server.api.index:app"])
        _launch(["ngrok", "http", "8000", "--domain", DOMAIN])

        proc = subprocess.Popen(
            ["tail", "-f", f"{tmpdir}/*.log"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(proc)

        try:
            for line in iter(proc.stdout.readline, b""):
                print(line.decode("utf-8"), end="")
        except KeyboardInterrupt:
            for proc in procs:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
