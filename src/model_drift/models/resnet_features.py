import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np

from torchmetrics.utilities.data import to_onehot
from torchvision.models import (
    ResNet34_Weights,
    resnet34,
)

class Resnet_Features(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.01)

        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))
        self.model = backbone

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, 1)
        out = out.squeeze()
        return out

    def predict_step(self, image_batch, batch_idx, **kwargs):
        batch = image_batch["image"]
        z = self.forward(batch)
        return z