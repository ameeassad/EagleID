import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch import optim

from torch.optim import SGD

from utils.triplet_loss_utils import TripletLoss
from utils.optimizer import get_optimizer, get_lr_scheduler_config
from utils.weights_initializer import weights_init_kaiming, weights_init_classifier

import pytorch_lightning as pl
from pytorch_metric_learning import losses, miners

from wildlife_tools.similarity.cosine import CosineSimilarity
from utils.metrics import evaluate_map, compute_average_precision

from utils.re_ranking import re_ranking

from data.data_utils import calculate_num_channels
from utils.metrics import compute_distance_matrix

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
    
class EfficientNet(pl.LightningModule):
    # inspiration from MiewIdNet
    def __init__(self,
                n_classes=0,
                backbone_model_name='efficientnet_b0',
                config = None, pretrained=True,
                margin=0.3,
                scale=50,
                dropout=0.0,
                lr=0.001, 
                preprocess_lvl=0, 
                re_ranking=True, 
                outdir="results",
                **kwargs):
        super().__init__()
        self.config = config
        if config:
            self.backbone_model_name=config['backbone_name']
            self.n_classes=config['arcface_loss']['n_classes']
            self.preprocess_lvl=int(config['preprocess_lvl'])
            self.re_ranking=config['re_ranking']
            self.distance_matrix = config['triplet_loss']['distance_matrix']
            outdir=config['outdir']
            if not config['use_wandb']:
                self.save_hyperparameters()
        else:
            self.backbone_model_name=backbone_model_name
            self.n_classes=n_classes
            self.preprocess_lvl=preprocess_lvl
            self.re_ranking=re_ranking
            self.distance_matrix = 'euclidean'
            outdir=outdir
        

        if self.backbone_model_name not in ['efficientnet_b0','efficientnet_b3','efficientnetv2_rw']:
            raise ValueError(f"Backbone model {self.backbone_model_name} not supported.")
        self.backbone = timm.create_model(backbone_model_name, num_classes=0, pretrained=pretrained)

        # Modify the first convolutional layer to support more input channels
        # if preprocess_lvl >= 3:
        #     in_channels = calculate_num_channels(preprocess_lvl)
        #     # Get the original first conv layer
        #     original_conv = self.backbone.conv_head if hasattr(self.backbone, 'conv_head') else self.backbone.conv1

        #     # Create a new convolutional layer with the desired number of input channels
        #     new_conv = nn.Conv2d(
        #         in_channels=in_channels,
        #         out_channels=original_conv.out_channels,
        #         kernel_size=original_conv.kernel_size,
        #         stride=original_conv.stride,
        #         padding=original_conv.padding,
        #         bias=original_conv.bias is not None
        #     )
        if self.preprocess_lvl >= 3:
            in_channels = calculate_num_channels(self.preprocess_lvl)
            # Get the first convolutional layer
            first_conv = list(self.backbone.children())[0]
            if isinstance(first_conv, nn.Conv2d):
                new_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None
                )
                # Copy weights from the existing conv layer
                with torch.no_grad():
                    new_conv.weight[:, :first_conv.in_channels] = first_conv.weight
                self.backbone.conv_stem = new_conv  # or set the appropriate attribute

            # Initialize weights of the new conv layer using Kaiming normalization
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            # Replace the old convolutional layer with the new one
            if hasattr(self.backbone, 'conv_head'):
                self.backbone.conv_head = new_conv
            else:
                self.backbone.conv1 = new_conv

        # Determine the number of input features for the final layer
        # self.final_in_features = self.backbone.classifier.in_features
        self.final_in_features = self.backbone.num_features

        # Remove the classifier
        self.backbone.reset_classifier(0)
        
        
        # # Remove the classifier and global pool layer, replace with identity
        # self.backbone.classifier = nn.Identity()
        # self.backbone.global_pool = nn.Identity()
        
        self.pooling = GeM()
        self.bn = nn.BatchNorm1d(self.final_in_features)
        # initialize parameters
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        self.loss_fn = losses.ArcFaceLoss(num_classes=self.n_classes, embedding_size=self.final_in_features, margin=margin, scale=scale)
        self.lr = lr

    def forward(self, x):
        feature = self.extract_feat(x)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)
        x = self.pooling(x).view(batch_size, -1)
        x = self.bn(x)
        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self.extract_feat(images)
        loss = self.loss_fn(embeddings, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        embeddings = self.extract_feat(images)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        if dataloader_idx == 0:
            # Query data
            self.query_embeddings.append(embeddings)
            self.query_labels.append(labels)
        else:
            # Gallery data
            self.gallery_embeddings.append(embeddings)
            self.gallery_labels.append(labels)

    def on_validation_epoch_start(self):
        self.query_embeddings = []
        self.query_labels = []
        self.gallery_embeddings = []
        self.gallery_labels = []

    def on_validation_epoch_end(self):
        # Concatenate embeddings and labels
        query_embeddings = torch.cat(self.query_embeddings, dim=0)
        query_labels = torch.cat(self.query_labels, dim=0)
        gallery_embeddings = torch.cat(self.gallery_embeddings, dim=0)
        gallery_labels = torch.cat(self.gallery_labels, dim=0)

        # Compute distance matrix
        if self.re_ranking:
            distmat = re_ranking(query_embeddings, gallery_embeddings, k1=20, k2=6, lambda_value=0.3)
        else:
            distmat = compute_distance_matrix(self.distance_matrix, query_embeddings, gallery_embeddings, wildlife=True)

        # Evaluate mAP and other metrics
        mAP1 = evaluate_map(distmat, query_labels, gallery_labels, top_k=1)
        mAP5 = evaluate_map(distmat, query_labels, gallery_labels, top_k=5)
        self.log('val/mAP1', mAP1)
        self.log('val/mAP5', mAP5)
    
    def configure_optimizers(self):
        if self.config:
            optimizer = get_optimizer(self.config, self.parameters())
            lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            return optimizer