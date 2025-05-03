import os
import numpy as np
import yaml

import wandb
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics import Accuracy

from utils.triplet_loss_utils import TripletLoss
from utils.optimizer import get_optimizer, get_lr_scheduler_config
from utils.weights_initializer import weights_init_kaiming, weights_init_classifier

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_metric_learning import losses, miners
from torch import nn

from utils.metrics import evaluate_map, compute_average_precision, similarity_matrix
from utils.re_ranking import re_ranking
from data.data_utils import calculate_num_channels
from utils.metrics import compute_distance_matrix, evaluate_recall_at_k, wildlife_accuracy



class ResNetPlusModel(pl.LightningModule):
    def __init__(self, 
                 backbone_model_name="resnet50", 
                 config=None, 
                 pretrained=True, # Use ImageNet pre-trained weights
                 embedding_size=256, 
                 margin=0.2, 
                 mining_type="semihard", 
                 lr=0.001, 
                 preprocess_lvl=0, 
                 re_ranking=True, 
                 outdir="results"):
        super().__init__()
        self.config = config
        if config:
            backbone_model_name=config['backbone_name'] if config['backbone_name'] else backbone_model_name
            self.embedding_size=int(config['embedding_size'])
            if config['arcface_loss']['activate']:
                self.n_classes=config['arcface_loss']['n_classes']
                margin=config['arcface_loss']['margin']
                scale=config['arcface_loss']['scale']
                self.arcface_loss = True
            else:
                margin=config['triplet_loss']['margin']
                mining_type=config['triplet_loss']['mining_type']
                self.arcface_loss = False
            self.preprocess_lvl=int(config['preprocess_lvl'])
            self.re_ranking=config['re_ranking']
            self.distance_matrix = config['triplet_loss']['distance_matrix']
            self.lr = config['solver']['BASE_LR']
            outdir=config['outdir']
            if not config['use_wandb']:
                self.save_hyperparameters()
            self.val_viz = config.get('val_viz', False)
            self.subset_val = ',' in config['wildlife_name']
        else:
            # Hardcode any changes
            backbone_model_name=backbone_model_name
            self.embedding_size=embedding_size
            margin=margin
            mining_type=mining_type
            self.preprocess_lvl=preprocess_lvl
            self.re_ranking=re_ranking
            self.distance_matrix = 'euclidean'
            self.lr = lr
            outdir=outdir
            self.val_viz = False
            self.subset_val = False
        self.distmat = None
        self.incl_metadata = True

        total_channels = calculate_num_channels(self.preprocess_lvl)

        self.backbone = timm.create_model(
            model_name=backbone_model_name, 
            pretrained=pretrained,
            num_classes=0,  # disable final classification layer
            features_only=True  # returns multiple feature maps
        )        
        
        if self.preprocess_lvl > 2:
            old_conv = self.backbone.conv1
            # print(f"Original conv1 weight shape: {old_conv.weight.shape}")  # Should be [64, 3, 7, 7]
            new_conv = nn.Conv2d(
                in_channels=total_channels, 
                out_channels=old_conv.out_channels,  # 64
                kernel_size=old_conv.kernel_size,  # (7, 7)
                stride=old_conv.stride,  # Usually (2, 2) for ResNet
                padding=old_conv.padding,  # Usually (3, 3)
                bias=old_conv.bias is not None  # Usually False for conv1 followed by BN
            )
            # print(f"New conv weight shape: {new_conv.weight.shape}")  # Should be [64, total_channels, 7, 7]
            if pretrained and total_channels >= 3:
                if self.preprocess_lvl == 4:
                # Repeat pretrained weights for every 3 input channels
                    with torch.no_grad():
                        repeat_times = total_channels // 3  # Number of full 3-channel groups
                        remainder = total_channels % 3      # Leftover channels
                        if repeat_times > 0:
                            # Repeat the [64, 3, 7, 7] weights along in_channels dimension
                            repeated_weights = old_conv.weight.data.repeat(1, repeat_times, 1, 1)
                            # Shape becomes [64, 3*repeat_times, 7, 7]
                            new_conv.weight.data[:, :3*repeat_times, :, :] = repeated_weights
                        if remainder > 0:
                            # Handle remaining channels by copying the first 'remainder' channels
                            new_conv.weight.data[:, 3*repeat_times:3*repeat_times+remainder, :, :] = \
                                old_conv.weight.data[:, :remainder, :, :]
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    # Copy the pretrained weights for the first 3 channels
                    new_conv.weight.data[:, :3, :, :] = old_conv.weight.data.clone()
                    # Initialize the additional channels
                    if total_channels > 3:
                        nn.init.kaiming_normal_(new_conv.weight.data[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            else:
                # If not pretrained, initialize all weights
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            self.backbone.conv1 = new_conv
       
        # Feature processing (Pooling)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(2048, embedding_size)
        # self.embedding = nn.Linear(self.backbone.feature_info[-1]['num_chs'], 
        #                            self.embedding_size)
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone.feature_info[-1]['num_chs'], self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.Dropout(p=0.4)  # Regularization
        )
        
        # Loss and mining
        self.loss_fn = losses.TripletMarginLoss(margin=margin)
        if self.arcface_loss:
            self.loss_fn = losses.ArcFaceLoss(num_classes=self.n_classes, embedding_size=self.embedding_size, margin=margin, scale=scale)
        self.miner = miners.TripletMarginMiner(margin=margin, type_of_triplets=mining_type)

    def forward(self, x):
        """Input shape: (B, 3+N, H, W)"""
        # print(f"Input shape: {x.shape}")  # Debug: Should be [batch_size, in_channels, 224, 224]
        features = self.backbone(x)  # Returns a tuple of feature maps
        # Select the last feature map (for the deepest features)
        features = features[-1]  # Take the last element of the tuple
        # print(f"Features shape before pooling: {features.shape}")  # Should be [B, C, H, W]
        
        pooled = self.global_pool(features)  # Shape: (B, 2048, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # Shape: (B, 2048)
        embeddings = self.embedding(flattened)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def training_step(self, batch, batch_idx):
        if self.incl_metadata:
            images, labels = batch['img'], batch['label']
        else:
            images, labels = batch
        embeddings = self(images)
        if not self.arcface_loss: # TripletLoss
            mined_triplets = self.miner(embeddings, labels)
            loss = self.loss_fn(embeddings, labels, mined_triplets)
        else: # ArcFaceLoss
            loss = self.loss_fn(embeddings, labels)
        self.log("train/loss", loss, batch_size=images.size(0), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_start(self): # to run only once: on_train_start / for every val: on_validation_start
        self.eval()
        self.on_validation_epoch_start()  # Initialize query/gallery embeddings and labels
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.trainer.val_dataloaders[0]):
                if self.incl_metadata:
                    x, target = batch['img'], batch['label']
                else:
                    x, target = batch
                # Generate random embeddings for the query set
                random_embeddings = torch.randn(x.size(0), self.embedding_size, device=x.device)
                self.query_embeddings.append(random_embeddings)
                self.query_labels.append(target)
            for batch_idx, batch in enumerate(self.trainer.val_dataloaders[1]):
                if self.incl_metadata:
                    x, target = batch['img'], batch['label']
                else:
                    x, target = batch
                # Generate random embeddings for the gallery set
                random_embeddings = torch.randn(x.size(0), self.embedding_size, device=x.device)
                self.gallery_embeddings.append(random_embeddings)
                self.gallery_labels.append(target)

            # Perform validation metric calculation using random embeddings
            # Compute the distance matrix using the random embeddings
            query_embeddings = torch.cat(self.query_embeddings)
            gallery_embeddings = torch.cat(self.gallery_embeddings)
            query_labels = torch.cat(self.query_labels)
            gallery_labels = torch.cat(self.gallery_labels)

            # Use a suitable distance metric for mAP calculation
            distmat = compute_distance_matrix(self.distance_matrix, query_embeddings, gallery_embeddings)
            random_mAP = evaluate_map(distmat, query_labels, gallery_labels)
            
            # Log the random baseline mAP
            print(f"Random mAP: {random_mAP}")
            self.log("random_val/mAP", random_mAP)
        # Switch back to training mode
        self.train()
    
    def on_validation_epoch_start(self):
        self.query_embeddings = []
        self.query_labels = []
        self.gallery_embeddings = []
        self.gallery_labels = []

        if self.incl_metadata:
            self.query_path = []
            self.query_identity = []
            self.gallery_path = []
            self.gallery_identity = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.incl_metadata:
            x, target, img_path, identity = batch['img'], batch['label'], batch['path'], batch['identity']
        else:
            x, target = batch
        embeddings = self(x)
        if dataloader_idx == 0:
            # Query data
            self.query_embeddings.append(embeddings)
            self.query_labels.append(target)
            if self.incl_metadata:
                self.query_path.append(img_path)
                self.query_identity.append(identity)
        else:
            # Gallery data
            self.gallery_embeddings.append(embeddings)
            self.gallery_labels.append(target)
            if self.incl_metadata:
                self.gallery_path.append(img_path)
                self.gallery_identity.append(identity)

    def on_validation_epoch_end(self):
        # Concatenate all embeddings and identities
        query_embeddings = torch.cat(self.query_embeddings)
        # query_labels = torch.cat(self.query_labels)
        gallery_embeddings = torch.cat(self.gallery_embeddings)
        # gallery_labels = torch.cat(self.gallery_labels)
        if not self.incl_metadata:
            raise ValueError("incl_metadata must be True to collect query_identity and gallery_identity")
        query_identities = [item for sublist in self.query_identity for item in sublist]
        gallery_identities = [item for sublist in self.gallery_identity for item in sublist]

        distmat = compute_distance_matrix(self.distance_matrix, query_embeddings, gallery_embeddings)
        sim_mx = 1 - distmat

        accuracy = wildlife_accuracy(sim_mx, query_identities=query_identities, gallery_identities=gallery_identities)
        self.log('val/accuracy', accuracy)

        if self.re_ranking:
            distmat_rr = re_ranking(query_embeddings, gallery_embeddings, k1=20, k2=6, lambda_value=0.3)
            mAP_rr = evaluate_map(distmat_rr, query_identities=query_identities, gallery_identities=gallery_identities)
            self.log('val/mAP_rr', mAP_rr)
            recall_at_k_rr = evaluate_recall_at_k(distmat_rr, query_identities=query_identities, gallery_identities=gallery_identities, k=5)
            self.log(f'val/Recall@5_rr', recall_at_k_rr)

        # Compute mAP
        # mAP = torchreid.metrics.evaluate_rank(distmat, query_labels.cpu().numpy(), gallery_labels.cpu().numpy(), use_cython=False)[0]['mAP']
        mAP = evaluate_map(distmat, query_identities=query_identities, gallery_identities=gallery_identities)
        self.log('val/mAP', mAP)
        
        recall_at_k = evaluate_recall_at_k(distmat, query_identities=query_identities, gallery_identities=gallery_identities, k=5)
        self.log(f'val/Recall@5', recall_at_k)

        if self.subset_val:
            self.raptor_metrics(query_embeddings, gallery_embeddings, query_identities, gallery_identities)

        if self.val_viz:
            self.distmat = distmat
            self.query_path_epoch = self.query_path  # set externally by DataModule or Trainer
            self.query_identity_epoch = self.query_identity
            self.gallery_path_epoch = self.gallery_path 
            self.gallery_identity_epoch = self.gallery_identity

    def raptor_metrics(self, query_embeddings, gallery_embeddings, query_identities, gallery_identities):
        gallery_identity_flat = gallery_identities
        query_identity_flat = query_identities
        target_prefixes = ['raptors']
        valid_gallery_indices = [
            i for i, identity in enumerate(gallery_identity_flat)
            if isinstance(identity, str) and any(p.lower() in identity.lower() for p in target_prefixes)
        ]
        if valid_gallery_indices:
            gallery_embeddings_subset = gallery_embeddings[valid_gallery_indices]
            gallery_identities_subset = [gallery_identity_flat[i] for i in valid_gallery_indices]
            valid_identities = [gallery_identity_flat[i] for i in valid_gallery_indices]
            valid_query_indices = [
                i for i, identity in enumerate(query_identity_flat)
                if isinstance(identity, str) and identity in valid_identities
            ]
            if valid_query_indices:
                query_embeddings_subset = query_embeddings[valid_query_indices]
                query_identities_subset = [query_identity_flat[i] for i in valid_query_indices]
                # print("▶ Valid gallery identities:", valid_identities[:5])
                # print("▶ Valid query identities:  ", [query_identity_flat[i] for i in valid_query_indices][:5])
                # print("▶ Identity intersection:   ", list(set(query_identities_subset) & set(gallery_identities_subset)))
                if len(query_embeddings_subset) > 0 and len(gallery_embeddings_subset) > 0:
                    distmat_subset = compute_distance_matrix(self.distance_matrix, query_embeddings_subset, gallery_embeddings_subset)
                    mAP_subset = evaluate_map(distmat_subset, query_identities_subset, gallery_identities_subset)
                    self.log('val/mAP_raptors', mAP_subset)
                else:
                    print("No valid query-gallery pairs after filtering. Skipping mAP_raptors.")
                    self.log('val/mAP_raptors', 0.0)
            else:
                print("No matching query identities found. Skipping mAP_raptors.")
                self.log('val/mAP_raptors', 0.0)
        else:
            print("No gallery identities match 'raptors'. Skipping mAP_raptors.")
            self.log('val/mAP_raptors', 0.0)

    def configure_optimizers(self):
        if self.config:
            optimizer = get_optimizer(self.config, self.parameters())
            lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer


