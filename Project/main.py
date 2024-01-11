# FastAPI endpoint written by Ivo

from typing import List

import PIL
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse

import FastAPI.multitask_model as multitask_model


class Predictions(BaseModel):
    age: float
    gender: int


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


@app.post("/predict", description="Image classifier endpoint. Add {'image': binary_image} to json body to send "
                                  "request. Image should be a black handwritten digit against a white background. "
                                  "Returns class confidences.",
          response_model=Predictions,
          response_description="Confidence for each of the possible digits 0-9. Confidences range from 0-1.")
async def predict(image: UploadFile):
    try:
        tensor_image = multitask_model.preprocess(image)
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")
    pred_age, pred_gen = multitask_model.predict(tensor_image)

    return Predictions(age=pred_age, gen=pred_gen)
