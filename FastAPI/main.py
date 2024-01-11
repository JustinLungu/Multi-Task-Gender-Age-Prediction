# FastAPI endpoint written by Ivo

from typing import List

import PIL
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse

import mnist_classifier


class DigitConfidence(BaseModel):
    digit: int = 7
    confidence: float = 0.2


class DigitPredictions(BaseModel):
    predictions: List[DigitConfidence]


app = FastAPI(
    title="Handwritten Digit Classifier",
    summary="An API endpoint to classify handwritten digits using a CNN. Trained on MNIST.",
    description="""
# An API endpoint to access a CNN trained on MNIST.
# Model usage
The model is trained on 28x28 images of handwritten digits. 
Consequently, it is designed to receive images that are mostly square and cover exactly 1 digit.
It is not built to identify digits in an image.

## Limitations
The model may give overconfident and erroneous predictions when symbols are submitted that are not a handwritten digit.
Printed digits rather may work, but OCR would be better suited for that. 

The model is sourced from https://github.com/arun477/mnist_classifier/tree/main.
    """,
    version="alpha",
)


@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')


@app.get("/hello_world", description="Hello world endpoint.")
async def hello_world():
    return "Hello world!"


def process_image(file):
    image = Image.open(file.file)
    image = image.resize((28, 28))  # Resize to MNIST image size
    image = image.convert("L")  # Convert to grayscale
    raw_image = image
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    return torch.from_numpy(image).float().reshape(1, 28, 28), raw_image


@app.post("/predict", description="Image classifier endpoint. Add {'image': binary_image} to json body to send "
                                  "request. Image should be a black handwritten digit against a white background. "
                                  "Returns class confidences.",
          response_model=DigitPredictions,
          response_description="Confidence for each of the possible digits 0-9. Confidences range from 0-1.")
async def predict(image: UploadFile):
    try:
        tensor_image, raw_image = process_image(image)
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")
    confs: np.ndarray = mnist_classifier.predict(tensor_image)
    digit_confs = [DigitConfidence(digit=i, confidence=conf) for i, conf in enumerate(confs)]

    return DigitPredictions(predictions=digit_confs)

