import requests
from gpiozero import MotionSensor

WEBHOOK = "https://inn.gs/e/H8gA6o2t2fClT2KDfseXbFjZA-Co3cQvfJwwucNKNamWuZNybXNgY7o9xV3u0fqdJ2xjZW102Q-LhAr1rTyctw"


def send_motion_event():
    requests.post(WEBHOOK)


pir: MotionSensor = MotionSensor(4)

while True:
    pir.wait_for_active()
    send_motion_event()
    pir.wait_for_inactive()
