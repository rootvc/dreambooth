import csv
import hashlib
from pathlib import Path

import requests
from cloudpathlib import CloudPath
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

PROMPTS = "/Users/rootventures/Documents/sparkbooth/prompts.txt"
IMAGES = "/Users/rootventures/Documents/sparkbooth/singles"
BUCKET = "rootvc-photobooth"
N_EXPECTED = 4
INNGEST_ENDPOINT = "9ZLArGIEZ6tpV0-kulgO_fKAE4Nd-UC_F6CMIiFuQWjfaKhyLZot4LTx9L_e5rrO4GXdYuPGKCzSoq0usM04Qw"

HANDLED = set()


class UploadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f"Received {event.src_path}")

        if event.src_path != PROMPTS:
            return

        reader = csv.DictReader(open(event.src_path, "r"), delimiter="\t")
        row = list(reader)[-1]

        file_id = Path(row["filename"]).stem
        phone = row["email"]
        key = hashlib.md5(phone.encode()).hexdigest()

        if file_id in HANDLED:
            return
        else:
            HANDLED.add(file_id)

        print(f"Uploading {file_id} to {key}")

        for idx in range(N_EXPECTED):
            CloudPath(f"s3://{BUCKET}/dataset/{key}/{idx}.jpg").upload_from(
                Path(f"{IMAGES}/{file_id}-{idx + 1}.jpg"), force_overwrite_to_cloud=True
            )

        requests.post(
            f"https://inn.gs/e/{INNGEST_ENDPOINT}", json={"phone": phone, "key": key}
        )


def main():
    observer = Observer()
    observer.schedule(UploadHandler(), Path(PROMPTS).parent)
    observer.start()

    print("Watching for changes...")

    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
