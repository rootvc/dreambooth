from modal import Image as DockerImage
from modal import Secret, Stub, Volume, gpu

stub = Stub()
volume = stub.volume = Volume.persisted("model-cache")

fn_kwargs = {
    "image": DockerImage.from_registry("rootventures/train-dreambooth-modal:latest")
    .run_commands(
        "pip uninstall -y opencv", "rm -rf /usr/local/lib/python3.10/dist-packages/cv2"
    )
    .pip_install(
        "opencv-contrib-python-headless==4.8.0.74",
        "opencv-python-headless==4.8.0.74",
        "onnxruntime-openvino",
        "insightface",
        "snoop",
    ),
    "gpu": gpu.L4(count=4),
    "memory": 22888,
    "cpu": 4.0,
    "volumes": {"/root/cache": volume},
    "secret": Secret.from_name("dreambooth"),
    "timeout": 60 * 30,
    "cloud": "gcp",
}
