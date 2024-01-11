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

loaded_model = None


def preprocess(file):
    image = Image.open(file).convert('L')
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(0.5, 0.5),
    ])
    image = transform(image)
    return image


def load_model():
    loaded_model = MultiTaskModel()
    loaded_model.load_state_dict(torch.load('trained_multi_model.pth'))
    loaded_model.eval()


def predict(img):
    with torch.no_grad():
        age_pred, gen_pred = loaded_model(img)  # Forward pass
        gen_pred = (gen_pred.squeeze() > 0.5).float()
    return age_pred, gen_pred
