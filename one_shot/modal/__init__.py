from modal import Image as DockerImage
from modal import Secret, Stub, Volume, gpu

stub = Stub("dreambooth-one-shot")
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
    )
    .apt_install("libwebp-dev")
    .run_commands(
        "pip uninstall -y pillow",
        'CC="cc -mavx2" pip install -U --force-reinstall pillow-simd --no-binary :all: -C webp=enable',
    )
    .env(
        {
            "TORCH_HOME": "/root/cache/torch",
        }
    ),
    "gpu": gpu.A100(count=1),
    "memory": 22888,
    "cpu": 4.0,
    "volumes": {"/root/cache": volume},
    "secret": Secret.from_name("dreambooth"),
    "timeout": 60 * 30,
    "cloud": "gcp",
    # "container_idle_timeout": 60 * 10,
}
