# base from https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py

import os
import numpy as np
import yaml

import wandb
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchmetrics import Accuracy

from utils.triplet_loss_utils import TripletLoss
from utils.optimizer import get_optimizer, get_lr_scheduler_config
from utils.weights_initializer import weights_init_kaiming, weights_init_classifier

class SimpleModel(LightningModule):
    def __init__(
        self,
        config: dict,
        pretrained: bool = False,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = timm.create_model(model_name=config['model_name'], pretrained=pretrained, num_classes=num_classes)

        # if skeleton, accept 4 channels instead of 3
        if config['preprocess_lvl']==3 and config['model_name'].startswith('resnet'):
            self.model.conv1 = nn.Conv2d(4, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size,
                                         stride=self.model.conv1.stride, padding=self.model.conv1.padding, bias=False)
        
        self.train_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_loss = nn.CrossEntropyLoss()
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.gradient = None
        self.outdir = self.config['outdir']
    
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        for name, module in self.model.named_modules():
            # for resnet layer 4 captures the high level features
            if name == 'layer4':
                x = module(x)
                x.register_hook(self.activations_hook)
                return x
        return None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        
        out = self(x)
        _, pred = out.max(1)

        loss = self.train_loss(out, target)
        acc = self.train_acc(pred, target)

        # self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        out = self(x)
        _, pred = out.max(1)

        loss = self.val_loss(out, target)
        acc = self.val_acc(pred, target)
        # self.log_dict({'val/loss': loss, 'val/acc': acc})

        # Log validation loss and accuracy
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
            x, target = batch

            x = x.to(torch.device('cpu'))
            target = target.to(torch.device('cpu'))

            x.requires_grad = True
        
            out = self(x)

            _, pred = out.max(1)
            if pred.numel() == 1:
                print(f"BATCH {batch_idx} PREDICTION: {pred.item()}")
            else:
                print(f"BATCH {batch_idx} PREDICTIONS: {pred.tolist()}")

    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
