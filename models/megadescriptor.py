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


# Arcface loss - needs backbone output size and number of classes.
objective = ArcFaceLoss(
    num_classes=dataset.num_classes,
    embedding_size=768,
    margin=0.5,
    scale=64
    )

# Optimize parameters in backbone and in objective using single optimizer.
params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)


trainer = BasicTrainer(
    dataset=dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=20,
    device='cpu',
)

trainer.train()


class MegadescriptorModel(pl.LightningModule):
    def __init__(self, backbone_model_name="MegaDescriptor-T", img_size=224, config=None, pretrained=True, embedding_size=128, margin=0.2, mining_type="semihard", lr=0.001, preprocess_lvl=0, re_ranking=True, outdir="results"):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.re_ranking = re_ranking
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

        if pretrained==True and img_size != 224:
            raise ValueError("Image size must be 224 for pretrained MegaDescriptor models.")
        
        backbone_model_name = f'hf-hub:BVRA/{backbone_model_name}-{img_size}'

        # Download MegaDescriptor-T backbone from HuggingFace Hub
        self.backbone = timm.create_model(backbone_model_name, num_classes=0, pretrained=pretrained)

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
    