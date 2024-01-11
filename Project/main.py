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
    gen: str


app = FastAPI(
    title="Multi-task Learning Age and Gender prediction ",
    summary="An API endpoint to predict age and gender using a Multitask Convolutional Neural Network (CNN)",
    description="""

# Model usage

This API provides a tool for predicting gender and age based on input images using a 
Multitask CNN. This API is designed to offer seamless integration into various use cases. 
For instance, insights of age and gender can be leveraged to assess the effectiveness of advertisement placements with respect 
to customer gender and age groups, enabling businesses to make data-driven decisions on their customer profiles, optimize marketing strategies, 
and tailor their advertising efforts to better resonate with their target audience demographic.

The model is trained on 100x100 images of male and female human faces, all of various ages ranging from 0 to 116 from the UTKface dataset 
(https://susanqq.github.io/UTKFace/). 
Consequently, it is designed to receive images that are mostly square and display a singular face.
 

The model is sourced from https://github.com/JustinLungu/Multi-Task-Gender-Age-Prediction.git.
    """,
    version="alpha",
)


@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')


# @app.get("/hello_world", description="Hello world endpoint.")
# async def hello_world():
#     return "Hello world!"


@app.post("/predict", description="Image classifier endpoint. Add {'image': binary_image} to json body to send "
                                  "request. Image should be of a singular human face. "
                                  "Returns age and gender prediction.",
          response_model=Predictions,
          response_description="Prediction values for age and gender")
async def predict(image: UploadFile):
    try:
        tensor_image = multitask_model.preprocess(image)
        print(tensor_image.shape)
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")
    #multitask_model.load_model()
    pred_age, pred_gen = multitask_model.predict(tensor_image)

    return Predictions(age=pred_age, gen=pred_gen)
