from pathlib import Path

import torch
from flask import (
    Flask,
    jsonify,
    request,
)

from PIL import Image
from torchvision import transforms

from src.models.cnn import MNISTCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = (
    Path("artifacts")
    / "checkpoints"
    / "best.pt"
)


# --------------------------------------------------
# Flask App
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Model Loading
# --------------------------------------------------

model = MNISTCNN()

if MODEL_PATH.exists():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint[
                    "state_dict"
                ]
            )

        else:
            model.load_state_dict(
                checkpoint
            )

    else:
        model.load_state_dict(
            checkpoint
        )

model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=1
        ),
        transforms.Resize(
            (28, 28)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        ),
    ]
)


def preprocess_image(
    image: Image.Image,
):
    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": "MNISTCNN",
        }
    )


@app.route(
    "/predict",
    methods=["POST"],
)
def predict():

    if "file" not in request.files:

        return (
            jsonify(
                {
                    "error": (
                        "No image uploaded"
                    )
                }
            ),
            400,
        )

    file = request.files["file"]

    image = Image.open(
        file.stream
    )

    x = preprocess_image(image)

    with torch.no_grad():

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1,
        )

        prediction = int(
            torch.argmax(
                probs,
                dim=1,
            ).item()
        )

        confidence = float(
            probs.max().item()
        )

    return jsonify(
        {
            "prediction": prediction,
            "confidence": confidence,
        }
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )