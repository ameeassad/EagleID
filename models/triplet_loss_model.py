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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data.artportalen_goleag import ArtportalenDataModule

from utils.triplet_loss_utils import TripletLoss
from utils.optimizer import get_optimizer, get_lr_scheduler_config
from utils.weights_initializer import weights_init_kaiming, weights_init_classifier

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_metric_learning import losses, miners
from torch import nn
import timm

from wildlife_tools.similarity.cosine import CosineSimilarity
from utils.metrics import evaluate_map, compute_average_precision

from utils.re_ranking import re_ranking

class TripletSimpleModel(pl.LightningModule):
    def __init__(self, backbone_model_name, embedding_size=128, margin=0.2, mining_type="semihard", lr=0.001):
        super().__init__()
        # Backbone (ResNet without the final FC layer)
        self.backbone = timm.create_model(model_name=backbone_model_name, pretrained=True, num_classes=0)
        # Embedder (to project features into the desired embedding space)
        self.embedder = nn.Linear(self.backbone.feature_info[-1]["num_chs"], embedding_size)
        # self.fc = nn.Linear(self.model.output_size, embedding_size)  # Embedding layer
        self.loss_fn = losses.TripletMarginLoss(margin=margin)
        self.miner = miners.TripletMarginMiner(margin=margin, type_of_triplets=mining_type)
        self.lr = lr

    def forward(self, x):
        features = self.backbone(x)  # Extract features using the backbone
        embeddings = self.embedder(features)  # Project features into the embedding space
        return embeddings  # Project features to embedding space

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)
        mined_triplets = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, mined_triplets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class TripletModel(pl.LightningModule):
    def __init__(self, backbone_model_name="resnet50", config=None, pretrained=True, embedding_size=128, margin=0.2, mining_type="semihard", lr=0.001, preprocess_lvl=0, re_ranking=True, outdir="results"):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.re_ranking = re_ranking
        if config:
            backbone_model_name=config['backbone_name']
            embedding_size=config['triplet_loss']['embedding_size']
            margin=config['triplet_loss']['margin']
            mining_type=config['triplet_loss']['mining_type']
            preprocess_lvl=config['preprocess_lvl'],
            self.re_ranking=config['re_ranking']
            self.distance_matrix = config['triplet_loss']['distance_matrix']
            outdir=config['outdir']
        else:
            backbone_model_name=backbone_model_name
            embedding_size=embedding_size
            margin=margin
            mining_type=mining_type
            preprocess_lvl=preprocess_lvl
            self.re_ranking=re_ranking
            self.distance_matrix = 'euclidean'
            outdir="logs"
            
        # Backbone (ResNet without the final FC layer)
        self.backbone = timm.create_model(model_name=backbone_model_name, pretrained=pretrained, num_classes=0)

        if preprocess_lvl >= 3:
            # Modify the first convolutional layer to accept 4 or 18 channels instead of 3
            if preprocess_lvl == 3: 
                num_channels = 4
            elif preprocess_lvl == 4: 
                num_channels = 18
            else:
                num_channels = 3
            if hasattr(self.backbone, 'conv1'):
                self.backbone.conv1 = nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=self.backbone.conv1.out_channels,
                    kernel_size=self.backbone.conv1.kernel_size,
                    stride=self.backbone.conv1.stride,
                    padding=self.backbone.conv1.padding,
                    bias=False
                )
                # Reinitialize the weights of the new convolutional layer
                nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Embedder (to project features into the desired embedding space)
        self.embedder = nn.Linear(self.backbone.feature_info[-1]["num_chs"], embedding_size)
        # self.fc = nn.Linear(self.model.output_size, embedding_size)  # Embedding layer
        self.loss_fn = losses.TripletMarginLoss(margin=margin)
        self.miner = miners.TripletMarginMiner(margin=margin, type_of_triplets=mining_type)
        self.lr = lr

    # Can experiment with different embedders or need to adjust the embedding layer frequently.
    def forward(self, x):
        features = self.backbone(x) # Extract features using the backbone
        embeddings = self.embedder(features) # Project features into the embedding space
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalization
        return embeddings

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)
        mined_triplets = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, mined_triplets)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.query_embeddings = []
        self.query_labels = []
        self.gallery_embeddings = []
        self.gallery_labels = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, target = batch
        embeddings = self(x)
        if dataloader_idx == 0:
            # Query data
            self.query_embeddings.append(embeddings)
            self.query_labels.append(target)
        else:
            # Gallery data
            self.gallery_embeddings.append(embeddings)
            self.gallery_labels.append(target)

    def on_validation_epoch_end(self):
        # Concatenate all embeddings and labels
        query_embeddings = torch.cat(self.query_embeddings)
        query_labels = torch.cat(self.query_labels)
        gallery_embeddings = torch.cat(self.gallery_embeddings)
        gallery_labels = torch.cat(self.gallery_labels)

        # Compute distance matrix
        if self.re_ranking:
            distmat = re_ranking(query_embeddings, gallery_embeddings, k1=20, k2=6, lambda_value=0.3)
        else:
            distmat = self.compute_distance_matrix(query_embeddings, gallery_embeddings)

        # Compute mAP
        # mAP = torchreid.metrics.evaluate_rank(distmat, query_labels.cpu().numpy(), gallery_labels.cpu().numpy(), use_cython=False)[0]['mAP']
        mAP = evaluate_map(distmat, query_labels, gallery_labels)
        self.log('val/mAP', mAP)

    def compute_distance_matrix(self, query_embeddings, gallery_embeddings, wildlife=True):
        if self.distance_matrix == "euclidean":
            # Compute Euclidean distance between query and gallery embeddings
            distmat = torch.cdist(query_embeddings, gallery_embeddings)
        elif self.distance_matrix == "cosine":
            if wildlife:
                similarity_function = CosineSimilarity()
                similarity = similarity_function(query_embeddings, gallery_embeddings)['cosine']
                distmat = 1 - similarity # Convert similarity to distance if necessary
                print(f"Distance matrix type should be np for rerankin: {type(distmat)}")
                return distmat
            else:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
                cosine_similarity = torch.mm(query_embeddings, gallery_embeddings.t())
                distmat = 1 - cosine_similarity # Convert similarity to distance if necessary
                print(f"Distance matrix type should be np for reranking: {type(distmat)}")
                return distmat

    def configure_optimizers(self):
        if self.config:
            optimizer = get_optimizer(self.config, self.parameters())
            lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer
    
    

# class TripletLossModel(LightningModule):
#     def __init__(
#         self,
#         config: dict,
#         model_name: str = 'resnet18',
#         pretrained: bool = False,
#         num_classes: int | None = None,
#         outdir: str = 'results',
#         margin: float = 0.3,
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.config = config

#         self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
#         self.train_loss = TripletLoss(margin=margin) 
#         self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
#         self.val_loss = TripletLoss(margin=margin)
#         self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
#         self.gradient = None
#         self.outdir = outdir

#         self.triplet_loss = TripletLoss(margin=margin)
    
#     def activations_hook(self, grad):
#         self.gradient = grad

#     def get_gradient(self):
#         return self.gradient

#     def get_activations(self, x):
#         for name, module in self.model.named_modules():
#             if name == 'layer4':
#                 x = module(x)
#                 x.register_hook(self.activations_hook)
#                 return x
#         return None

#     def forward(self, x):
#         # Get features from the backbone
#         features = self.model(x)
#         # Optionally normalize features for triplet loss
#         normalized_features = nn.functional.normalize(features, p=2, dim=1)
#         # Get classification logits
#         logits = self.fc(features)
#         return logits, normalized_features


#     def training_step(self, batch, batch_idx):
#         x, target = batch
#         logits, features = self(x)

#         loss, _, _ = self.train_loss(features, target)
        
#         # Accuracy
#         _, pred = logits.max(1)
#         acc = self.train_acc(pred, target)

#         self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, target = batch
#         out = self(x)
#         _, pred = out.max(1)

#         loss = self.val_loss(out, target)
#         acc = self.val_acc(pred, target)
#         self.log_dict({'val/loss': loss, 'val/acc': acc})

#     def test_step(self, batch, batch_idx):
#             x, target = batch

#             x = x.to(torch.device('cpu'))
#             target = target.to(torch.device('cpu'))

#             x.requires_grad = True
        
#             out = self(x)

#             _, pred = out.max(1)
#             if pred.numel() == 1:
#                 print(f"BATCH {batch_idx} PREDICTION: {pred.item()}")
#             else:
#                 print(f"BATCH {batch_idx} PREDICTIONS: {pred.tolist()}")

    
#     def configure_optimizers(self):
#         optimizer = get_optimizer(self.config, self.parameters())
#         lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
#         return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    

