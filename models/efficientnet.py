import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch import optim

from .heads import ArcMarginProduct, ElasticArcFace, ArcFaceSubCenterDynamic

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
    
class MiewIdNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 pretrained=True,
                 in_channels=3,  # Added parameter for number of input channels
                 **kwargs):
        """
        method: TimmBackbone
        model_name: 'efficientnet_b0'
        """
        super(MiewIdNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.model_name = model_name

        # Create the backbone using timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Modify the first convolutional layer to support more input channels
        if in_channels != 3:
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
        if model_name.startswith('efficientnetv2_rw'):
            final_in_features = self.backbone.classifier.in_features
        elif model_name.startswith('swinv2'):
            final_in_features = self.backbone.norm.normalized_shape[0]

        self.final_in_features = final_in_features
        
        # Remove the classifier and global pool layer, replace with identity
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
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
        if self.model_name.startswith('swinv2'):
            x = x.permute(0, 3, 1, 2)

        x = self.pooling(x).view(batch_size, -1)
        x = self.bn(x)
        if self.use_fc:
            x1 = self.dropout(x)
            x1 = self.bn(x1)
            x1 = self.fc(x1)
        return x

            
class MiewIdNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 pretrained=True,
                 **kwargs):
        """
        method: TimmBackbone
        model_name: 'efficientnet_b0'
        efficientnet_b3
        """
        super(MiewIdNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.model_name = model_name


        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if model_name.startswith('efficientnetv2_rw'):
            final_in_features = self.backbone.classifier.in_features
        if model_name.startswith('swinv2'):
            final_in_features = self.backbone.norm.normalized_shape[0]

        self.final_in_features = final_in_features
        
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        self.pooling =  GeM()
        self.bn = nn.BatchNorm1d(final_in_features)
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.bn.bias.requires_grad_(False)
            self.fc = nn.Linear(final_in_features, n_classes, bias = False)            
            self.bn.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
            final_in_features = fc_dim

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label=None):
        feature = self.extract_feat(x)
        
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)
        if self.model_name.startswith('swinv2'):
            x = x.permute(0, 3, 1, 2)

        x = self.pooling(x).view(batch_size, -1)
        x = self.bn(x)
        if self.use_fc:
            x1 = self.dropout(x)
            x1 = self.bn(x1)
            x1 = self.fc(x1)
    
        return x
    

class EfficientNet(pl.LighntingModule):







     #Load EfficientNet model from timm
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    #efficientnetv2_rw_m

# Define your custom input channels (e.g., 4 channels)
num_input_channels = 4

# Modify the conv_stem layer to have custom input channels
model.conv_stem = nn.Conv2d(
    in_channels=num_input_channels,
    out_channels=model.conv_stem.out_channels,
    kernel_size=model.conv_stem.kernel_size,
    stride=model.conv_stem.stride,
    padding=model.conv_stem.padding,
    bias=False
)

# Reinitialize the weights for conv_stem after modifying it
nn.init.kaiming_normal_(model.conv_stem.weight, mode='fan_out', nonlinearity='relu')