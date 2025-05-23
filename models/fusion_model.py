import wandb
import timm
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

from utils.metrics import evaluate_map

from utils.re_ranking import re_ranking
from data.data_utils import calculate_num_channels
from utils.metrics import compute_distance_matrix, evaluate_recall_at_k, wildlife_accuracy

class FusionModel(pl.LightningModule):
    def __init__(self, 
                 backbone_model_name="resnet18", 
                 config=None, pretrained=True, 
                 embedding_size=128, margin=0.2, 
                 mining_type="semihard", 
                 lr=0.001, 
                 preprocess_lvl=0, 
                 re_ranking=True, 
                 outdir="results"):
        super().__init__()
        self.config = config
        if config:
            backbone_model_name=config['backbone_name'] if config['backbone_name'] else 'resnet18'
            self.embedding_size=int(config['embedding_size'])
            margin=config['triplet_loss']['margin']
            mining_type=config['triplet_loss']['mining_type']
            self.preprocess_lvl=int(config['preprocess_lvl'])
            self.re_ranking=config['re_ranking']
            self.distance_matrix = config['triplet_loss']['distance_matrix']
            outdir=config['outdir']
            if not config['use_wandb']:
                self.save_hyperparameters()
        else:
            backbone_model_name=backbone_model_name
            self.embedding_size=embedding_size
            margin=margin
            mining_type=mining_type
            self.preprocess_lvl=preprocess_lvl
            self.re_ranking=re_ranking
            self.distance_matrix = 'euclidean'
            outdir=outdir
            
        self.backbone = timm.create_model(model_name=backbone_model_name, pretrained=pretrained, num_classes=0, global_pool='', features_only=True)
        if self.preprocess_lvl >= 3:

            if self.preprocess_lvl == 3:
                num_kp_groups = calculate_num_channels(self.preprocess_lvl) - 3
                in_chans = 1 # skeleton
            elif self.preprocess_lvl == 4:
                # num_kp_groups = calculate_num_channels(self.preprocess_lvl) - 3 // 3
                num_kp_groups = 5
                in_chans = 3 # parts each RGB channel
            elif self.preprocess_lvl == 5:
                num_kp_groups = calculate_num_channels(self.preprocess_lvl) - 3
                in_chans = 1 # heatmap each 1 channel
            # self.kp_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.backbone_kp = timm.create_model("resnet18", pretrained=False, num_classes=0, in_chans=in_chans, global_pool='', features_only=True)
            self.kp_pools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(num_kp_groups)])
            # self.keypoint_branch = nn.Sequential(
            #     nn.Conv2d(in_channels=num_kp_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(inplace=True),
            #     self.backbone_img.maxpool,
            #     self.backbone_img.layer1,
            #     self.backbone_img.layer2,
            # )

        # Pooling
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding layer
        # if self.preprocess_lvl >= 3:
        #     total_feature_dim = self.backbone.feature_info[-1]['num_chs'] + self.backbone_kp.feature_info[-1]['num_chs']
        if self.preprocess_lvl >= 3:
            total_feature_dim = (
                self.backbone.feature_info[-1]['num_chs'] + 
                (num_kp_groups * self.backbone_kp.feature_info[-1]['num_chs'])  # Multiply by num_kp_groups
            )
        else:
            total_feature_dim = self.backbone.feature_info[-1]['num_chs']
        self.embedding = nn.Linear(total_feature_dim, self.embedding_size)

        self.loss_fn = losses.TripletMarginLoss(margin=margin)
        self.miner = miners.TripletMarginMiner(margin=margin, type_of_triplets=mining_type)
        self.lr = lr

    def forward(self, x):
        # Process RGB image through global backbone
        x_rgb = x[:, :3, :, :]
        features_rgb = self.backbone(x_rgb)[-1]  # Shape: (B, C_rgb, H, W)
        
        if self.preprocess_lvl >= 3:
            # Process each keypoint part separately through the keypoint backbone
            x_kp = x[:, 3:, :, :]
            kp_features = []
            if self.preprocess_lvl == 3:
                kp_feature = self.backbone_kp(x_kp)[-1]  # Single grayscale channel
                kp_pooled = self.kp_pools[0](kp_feature).view(kp_feature.size(0), -1)
                kp_features.append(kp_pooled) 
            elif self.preprocess_lvl == 4:
                for i in range(x_kp.shape[1] // 3):  # Assuming 3 channels per part
                    kp_part = x_kp[:, i*3:(i+1)*3, :, :]
                    kp_feature = self.backbone_kp(kp_part)[-1]
                    kp_pooled = self.kp_pools[i](kp_feature).view(kp_feature.size(0), -1)  # Shape: (B, C_kp)
                    kp_features.append(kp_pooled)
            elif self.preprocess_lvl == 5:
                for i in range(x_kp.shape[1]):  # Each channel is a separate grayscale keypoint
                    kp_part = x_kp[:, i:i+1, :, :]  # Select each single-channel keypoint
                    kp_feature = self.backbone_kp(kp_part)[-1]
                    kp_pooled = self.kp_pools[i](kp_feature).view(kp_feature.size(0), -1)  # Shape: (B, C_kp)
                    kp_features.append(kp_pooled)
                
            # Concatenate keypoint features along the channel dimension
            kp_features = torch.cat(kp_features, dim=1)  # Shape: (B, C_kp_total)

            # Concatenate RGB features with keypoint features
            features_rgb_pooled = self.global_pool(features_rgb).view(features_rgb.size(0), -1)  # Shape: (B, C_rgb)
            features = torch.cat((features_rgb_pooled, kp_features), dim=1)  # Shape: (B, C_rgb + C_kp_total)
        else:
            features = self.global_pool(features_rgb).view(features_rgb.size(0), -1)

        # Embedding
        embeddings = self.embedding(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalization
        return embeddings
    
    def on_train_start(self): # to run only once: on_train_start / for every val: on_validation_start
        self.eval()
        self.on_validation_epoch_start()  # Initialize query/gallery embeddings and labels
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.trainer.val_dataloaders[0]):
                x, target = batch
                # Generate random embeddings for the query set
                random_embeddings = torch.randn(x.size(0), self.embedding_size, device=x.device)
                self.query_embeddings.append(random_embeddings)
                self.query_labels.append(target)
            for batch_idx, batch in enumerate(self.trainer.val_dataloaders[1]):
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
            distmat = compute_distance_matrix('euclidean', query_embeddings, gallery_embeddings)
            random_mAP = evaluate_map(distmat, query_labels, gallery_labels)
            
            # Log the random baseline mAP
            print(f"Random mAP: {random_mAP}")
            self.log("random_val/mAP", random_mAP)
        # Switch back to training mode
        self.train()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)
        mined_triplets = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, mined_triplets)
        self.log("train/loss", loss,  on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
            distmat = compute_distance_matrix(self.distance_matrix, query_embeddings, gallery_embeddings, wildlife=True)

        # Compute mAP
        # mAP = torchreid.metrics.evaluate_rank(distmat, query_labels.cpu().numpy(), gallery_labels.cpu().numpy(), use_cython=False)[0]['mAP']
        mAP = evaluate_map(distmat, query_labels, gallery_labels)
        mAP1 = evaluate_map(distmat, query_labels, gallery_labels, top_k=1)
        mAP5 = evaluate_map(distmat, query_labels, gallery_labels, top_k=5)
        self.log('val/mAP', mAP)
        self.log('val/mAP1', mAP1)
        self.log('val/mAP5', mAP5)

        recall_at_k = evaluate_recall_at_k(distmat, query_labels, gallery_labels, k=5)
        self.log(f'val/Recall@5', recall_at_k)

        accuracy = wildlife_accuracy(query_embeddings, gallery_embeddings, query_labels, gallery_labels)
        self.log(f'val/accuracy', accuracy)

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
    

