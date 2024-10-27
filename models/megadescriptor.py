import timm
import itertools
from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer


import torch
import torch.nn as nn

from utils.triplet_loss_utils import TripletLoss
from utils.optimizer import get_optimizer, get_lr_scheduler_config
from utils.weights_initializer import weights_init_kaiming, weights_init_classifier

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_metric_learning import losses, miners
from torch import nn

from wildlife_tools.similarity.cosine import CosineSimilarity
from utils.metrics import evaluate_map, compute_average_precision

from utils.re_ranking import re_ranking

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pytorch_metric_learning import losses

from data.data_utils import calculate_num_channels
from utils.metrics import compute_distance_matrix

class MegadescriptorModel(pl.LightningModule):
    def __init__(self, 
                 backbone_model_name="swin_base_patch4_window7_224", 
                 img_size=224, 
                 num_classes=1000, 
                 config=None, 
                 pretrained=True, 
                 embedding_size=768, 
                 margin=0.5, 
                 scale=64, 
                 lr=0.001, 
                 preprocess_lvl=0, 
                 re_ranking=True, 
                 outdir="results"):
        super().__init__()
        self.config = config

        if config:
            backbone_model_name=config['backbone_name']
            embedding_size=int(config['embedding_size'])
            self.num_classes = config['num_classes']
            self.margin=config['megadescriptor']['margin']
            self.scale=config['megadescriptor']['scale']
            preprocess_lvl=int(config['preprocess_lvl'])
            img_size=config['img_size']
            self.re_ranking=config['re_ranking']
            self.lr = config['solver']['BASE_LR']
            self.distance_matrix = config['triplet_loss']['distance_matrix']
            outdir=config['outdir']
        else:
            backbone_model_name=backbone_model_name
            embedding_size=embedding_size
            self.num_classes = num_classes
            self.margin=margin
            self.scale=scale
            preprocess_lvl=preprocess_lvl
            img_size=img_size
            self.re_ranking=re_ranking
            self.lr = lr
            self.distance_matrix = 'euclidean'
            outdir=outdir
        self.save_hyperparameters()

        # Backbone model
        if self.backbone not in ['swin_large_patch4_window7_224','swin_base_patch4_window7_224','swin_large_patch4_window12_384','swin_tiny_patch4_window7_224']:
            raise ValueError(f"Backbone model {self.backbone} not supported.")
        self.backbone = timm.create_model(backbone_model_name, num_classes=0, pretrained=pretrained)
        
        # Adjust input channels if necessary based on preprocess level
        if preprocess_lvl >= 3:
            num_channels = calculate_num_channels(preprocess_lvl)
            if hasattr(self.backbone, 'patch_embed'):
                self.backbone.patch_embed.proj = nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=self.backbone.patch_embed.proj.out_channels,
                    kernel_size=self.backbone.patch_embed.proj.kernel_size,
                    stride=self.backbone.patch_embed.proj.stride,
                    padding=self.backbone.patch_embed.proj.padding,
                    bias=False
                )
                # Reinitialize weights for the adjusted layer
                nn.init.kaiming_normal_(self.backbone.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')
        else:
            num_channels = 3

        # Embedder: Linear layer to project backbone output to embedding size
        with torch.no_grad():
            dummy_input = torch.randn(1, num_channels, 224, 224)
            pretrained_embedding_size = self.backbone(dummy_input).shape[1]
            print(f"Pretrained embedding size: {pretrained_embedding_size}")
            print(f"Input embedding size: {embedding_size}")
        self.embedding_size = embedding_size
        self.embedder = nn.Linear(self.backbone.num_features, embedding_size)
        
        # ArcFace Loss: Initializes the margin-based loss function
        self.loss_fn = losses.ArcFaceLoss(num_classes=self.num_classes, embedding_size=self.embedding_size, margin=self.margin, scale=self.scale)

    def forward(self, x):
        features = self.backbone(x)  # Get features from the backbone
        embeddings = self.embedder(features)  # Project to embedding space
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize the embeddings
        return embeddings

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)  # Extract embeddings
        loss = self.loss_fn(embeddings, labels)  # Compute ArcFace loss
        self.log("train_loss", loss)  # Log the loss for monitoring
        return loss

    def on_validation_epoch_start(self):
        self.query_embeddings = []
        self.query_labels = []
        self.gallery_embeddings = []
        self.gallery_labels = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, target = batch
        embeddings = self(x)
        self.log('val_loss', self.loss_fn(embeddings, target))
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
            distmat = compute_distance_matrix(self.distance_matrix, query_embeddings, gallery_embeddings)

        # Compute mAP
        # mAP = torchreid.metrics.evaluate_rank(distmat, query_labels.cpu().numpy(), gallery_labels.cpu().numpy(), use_cython=False)[0]['mAP']
        mAP = evaluate_map(distmat, query_labels, gallery_labels)
        self.log('val/mAP', mAP)

    def configure_optimizers(self):
        if self.config:
            optimizer = get_optimizer(self.config, self.parameters())
            lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            return optimizer


class MegadescriptorModel(pl.LightningModule):
    def __init__(self, backbone_model_name="swin_base_patch4_window7_224", img_size=224, config=None, pretrained=True, embedding_size=768, margin=0.5, scale=64, lr=0.001, preprocess_lvl=0, re_ranking=True, outdir="results"):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if config:
            backbone_model_name=config['backbone_name']
            embedding_size=int(config['triplet_loss']['embedding_size'])
            margin=config['triplet_loss']['margin']
            mining_type=config['triplet_loss']['mining_type']
            preprocess_lvl=int(config['preprocess_lvl'])
            img_size=config['img_size']
            self.re_ranking=config['re_ranking']
            self.distance_matrix = config['triplet_loss']['distance_matrix']
            outdir=config['outdir']
        else:
            backbone_model_name=backbone_model_name
            embedding_size=embedding_size
            margin=margin
            mining_type=mining_type
            preprocess_lvl=preprocess_lvl
            img_size=img_size
            self.re_ranking=re_ranking
            self.distance_matrix = 'euclidean'
            outdir=outdir

        #  Swin-L/p4-w12-384
        # Backbone and loss configuration
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            embedding_size = backbone(dummy_input).shape[1]


        # # Optimize parameters in backbone and in objective using single optimizer.
        # params = itertools.chain(backbone.parameters(), objective.parameters())
        # Optimizer and scheduler configuration
        # params = chain(backbone.parameters(), objective.parameters())
        # optimizer = SGD(params=params, lr=0.001, momentum=0.9)
        # min_lr = optimizer.defaults.get("lr") * 1e-3
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=min_lr)


        # Setup training
        # trainer = BasicTrainer(
        #     dataset=dataset,
        #     model=backbone,
        #     objective=objective,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     batch_size=64,
        #     accumulation_steps=2,
        #     num_workers=2,
        #     epochs=100,
        #     device='cuda',
        # )

class MiewIdNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='swinv2_base_window12_192_22k',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 pretrained=True,
                 in_channels=3,  # Parameter for the number of input channels
                 **kwargs):
        """
        SwinV2 Model initialization
        """
        super(MiewIdNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.model_name = model_name

        # Create the backbone using timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Modify the patch embedding layer if the number of input channels is different
        if in_channels != 3:
            # Get the original patch embedding layer
            patch_embed = self.backbone.patch_embed

            # Create a new patch embedding layer with the desired number of input channels
            new_patch_embed = nn.Conv2d(
                in_channels=in_channels,
                out_channels=patch_embed.proj.out_channels,
                kernel_size=patch_embed.proj.kernel_size,
                stride=patch_embed.proj.stride,
                padding=patch_embed.proj.padding,
                bias=patch_embed.proj.bias is not None
            )

            # Initialize the weights of the new patch embedding layer
            nn.init.kaiming_normal_(new_patch_embed.weight, mode='fan_out', nonlinearity='relu')

            # Replace the old patch embedding layer with the new one
            self.backbone.patch_embed.proj = new_patch_embed

        # Determine the number of input features for the final layer
        final_in_features = self.backbone.head.in_features

        self.final_in_features = final_in_features
        
        # Remove the classification head
        self.backbone.head = nn.Identity()
        
        self.pooling = GeM()
        self.bn = nn.BatchNorm1d(final_in_features)
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.bn.bias.requires_grad_(False)
            self.fc = nn.Linear(final_in_features, n_classes, bias=False)            
            self.bn.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
            final_in_features = fc_dim

    def forward(self, x):
        feature = self.extract_feat(x)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)

        x = self.pooling(x).view(batch_size, -1)
        x = self.bn(x)
        if self.use_fc:
            x1 = self.dropout(x)
            x1 = self.bn(x1)
            x1 = self.fc(x1)
        return x

