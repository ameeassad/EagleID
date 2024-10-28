import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch import optim

from .heads import ArcMarginProduct, ElasticArcFace, ArcFaceSubCenterDynamic

from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer

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


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

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
    
class EfficientNet(pl.LighntingModule):
    # inspiration from MiewIdNet
    def __init__(self,
                n_classes,
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
            backbone_model_name=config['backbone_name']
            preprocess_lvl=int(config['preprocess_lvl'])
            self.re_ranking=config['re_ranking']
            self.distance_matrix = config['triplet_loss']['distance_matrix']
            outdir=config['outdir']
        else:
            backbone_model_name=backbone_model_name
            preprocess_lvl=preprocess_lvl
            self.re_ranking=re_ranking
            self.distance_matrix = 'euclidean'
            outdir=outdir
        self.save_hyperparameters()
        

        if self.backbone not in ['efficientnet_b0','efficientnet_b3','efficientnetv2_rw']:
            raise ValueError(f"Backbone model {self.backbone} not supported.")
        self.backbone = timm.create_model(backbone_model_name, num_classes=0, pretrained=pretrained)
        self.model_name = backbone_model_name

        # Create the backbone using timm
        self.backbone = timm.create_model(backbone_model_name, pretrained=pretrained)

        # Modify the first convolutional layer to support more input channels
        if preprocess_lvl >= 3:
            in_channels = calculate_num_channels(preprocess_lvl)
            # Get the original first conv layer
            original_conv = self.backbone.conv_head if hasattr(self.backbone, 'conv_head') else self.backbone.conv1

            # Create a new convolutional layer with the desired number of input channels
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

            # Initialize weights of the new conv layer using Kaiming normalization
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            # Replace the old convolutional layer with the new one
            if hasattr(self.backbone, 'conv_head'):
                self.backbone.conv_head = new_conv
            else:
                self.backbone.conv1 = new_conv

        # Determine the number of input features for the final layer
        self.final_in_features = self.backbone.classifier.in_features
        
        # Remove the classifier and global pool layer, replace with identity
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        self.pooling = GeM()
        self.bn = nn.BatchNorm1d(self.final_in_features)
        # initialize parameters
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        self.loss_fn = losses.ArcFaceLoss(num_classes=n_classes, embedding_size=self.final_in_features, margin=margin, scale=scale)
        self.lr = lr

        self._init_params()

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        features = self.extract_feat(images)
        logits = self.fc(features)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.config:
            optimizer = get_optimizer(self.config, self.parameters())
            lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            return optimizer