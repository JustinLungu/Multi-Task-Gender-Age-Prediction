#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate, DataLoader
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from Model_Code.MultiTaskModel import MultiTaskModel

# model = None


def preprocess(file):
    image = Image.open(file.file).convert('L')
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(0.5, 0.5),
    ])
    image = transform(image)
    image = torch.stack([image])
    return image


# def load_model():
model = MultiTaskModel()
model.load_state_dict(torch.load('FastAPI/final_model.pth'))
print(model)
model.eval()
print("Model loaded")


def predict(img):
    with torch.no_grad():
        age_pred, gen_pred = model(img)  # Forward pass
        gen_pred = "male" if gen_pred.squeeze().item() <= 0.5 else "female"
    return age_pred, gen_pred
