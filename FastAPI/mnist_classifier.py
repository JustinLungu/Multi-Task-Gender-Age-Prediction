#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import fastcore.all as fc
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate, DataLoader
import torch.optim as optim


def transform_ds(b):
    b[x] = [TF.to_tensor(ele) for ele in b[x]]
    return b


bs = 1024
class DataLoaders:
    def __init__(self, train_ds, valid_ds, bs, collate_fn, **kwargs):
        self.train = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, **kwargs)
        self.valid = DataLoader(valid_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, **kwargs)

def collate_fn(b):
    collate = default_collate(b)
    return (collate[x], collate[y])


class Reshape(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.reshape(self.dim)


def conv(ni, nf, ks=3, s=2, act=nn.ReLU, norm=None):
    layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=s, padding=ks//2)]
    if norm:
        layers.append(norm)
    if act:
        layers.append(act())
    return nn.Sequential(*layers)

def _conv_block(ni, nf, ks=3, s=2, act=nn.ReLU, norm=None):
    return nn.Sequential(
        conv(ni, nf, ks=ks, s=1, norm=None, act=act),
        conv(nf, nf, ks=ks, s=s, norm=norm, act=act),
    )

class ResBlock(nn.Module):
    def __init__(self, ni, nf, s=2, ks=3, act=nn.ReLU, norm=None):
        super().__init__()
        self.convs = _conv_block(ni, nf, s=s, ks=ks, act=act, norm=norm)
        self.idconv = fc.noop if ni==nf else conv(ni, nf, ks=1, s=1, act=None)
        self.pool = fc.noop if s==1 else nn.AvgPool2d(2, ceil_mode=True)
        self.act = act()
    
    def forward(self, x):
        return self.act(self.convs(x) + self.idconv(self.pool(x)))


def cnn_classifier():
    return nn.Sequential(
        ResBlock(1, 8, norm=nn.LayerNorm([8, 14, 14])),
        ResBlock(8, 16, norm=nn.LayerNorm([16, 7, 7])),
        ResBlock(16, 32, norm=nn.LayerNorm([32, 4, 4])),
        ResBlock(32, 64, norm=nn.LayerNorm([64, 2, 2])),
        ResBlock(64, 64,  norm=nn.LayerNorm([64, 1, 1])),
        conv(64, 10, act=False),
        nn.Flatten(),
    )


def kaiming_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight)        


loaded_model = cnn_classifier()
loaded_model.load_state_dict(torch.load('classifier.pth'));
loaded_model.eval();


def predict(img):
    with torch.no_grad():
        img = img[None,]
        pred = loaded_model(img)[0]
        pred_probs = F.softmax(pred, dim=0)
        pred = [{"digit": i, "conf": f'{prob*100:.2f}%', 'logits': pred[i]} for i, prob in enumerate(pred_probs)]
        pred = sorted(pred, key=lambda ele: ele['digit'], reverse=False)
    return pred_probs

