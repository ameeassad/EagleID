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

from wildlife_tools.similarity.cosine import CosineSimilarity

from utils.re_ranking import re_ranking
from data.data_utils import calculate_num_channels
from utils.metrics import compute_distance_matrix, evaluate_map, evaluate_recall_at_k, wildlife_accuracy

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from utils.triplet_loss_utils import KnnClassifier



class MegaDescriptor(pl.LightningModule):
    def __init__(self, 
                 backbone_model_name="megadescriptor", 
                 config=None, pretrained=True, 
                 embedding_size=128,
                 lr=0.001, 
                 scale=50,
                 margin=0.5,
                 n_classes =0,
                 preprocess_lvl=0, 
                 re_ranking=True, 
                 outdir="results"):
        super().__init__()
        self.config = config
        if config:
            backbone_model_name=config['backbone_name'] if config['backbone_name'] else 'resnet18'
            self.embedding_size=int(config['embedding_size'])
            margin=config['triplet_loss']['margin']
            scale=config['arcface_loss']['scale']
            self.n_classes=config['arcface_loss']['n_classes']
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
            scale=scale
            self.n_classes=n_classes
            self.preprocess_lvl=preprocess_lvl
            self.re_ranking=re_ranking
            self.distance_matrix = 'cosine'
            outdir=outdir
            
        self.backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)

        # self.embedder = nn.Linear(self.backbone.feature_info[-1]["num_chs"], embedding_size)
        # self.fc = nn.Linear(self.model.output_size, embedding_size)  # Embedding layer
        self.loss_fn = losses.ArcFaceLoss(num_classes=self.n_classes, embedding_size=self.embedding_size, margin=margin, scale=scale)
        self.lr = lr

    def forward(self, x):
        features = self.backbone(x) # Extract features using the backbone
        # embeddings = self.embedder(features)
        # embeddings = nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalization
        # return embeddings
        # features = F.normalize(features, p=2, dim=1) # MegaDescriptor-T already includes normalization?
        return features
    
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
            random_mAP1 = evaluate_map(distmat, query_labels, gallery_labels, top_k=1)
            random_mAP5 = evaluate_map(distmat, query_labels, gallery_labels, top_k=5)
            
            # Log the random baseline mAP
            print(f"Random mAP: {random_mAP1}")
            self.log("random_val/mAP1", random_mAP1)
            self.log("random_val/mAP5", random_mAP5)
        # Switch back to training mode
        self.train()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)
        loss = self.loss_fn(embeddings, labels)
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
            self.query_labels.extend(target)
        else:
            # Gallery data
            self.gallery_embeddings.append(embeddings)
            self.gallery_labels.extend(target)

    # def on_validation_epoch_end(self):
    #     # Concatenate all embeddings and labels
    #     query_embeddings = torch.cat(self.query_embeddings)
    #     query_labels = torch.cat(self.query_labels)
    #     gallery_embeddings = torch.cat(self.gallery_embeddings)
    #     gallery_labels = torch.cat(self.gallery_labels)
        
    #     # Compute distance matrix
    #     if self.re_ranking:
    #         distmat = re_ranking(query_embeddings, gallery_embeddings, k1=20, k2=6, lambda_value=0.3)
    #     else:
    #         distmat = compute_distance_matrix(self.distance_matrix, query_embeddings, gallery_embeddings, wildlife=True)

    #     # Compute mAP
    #     mAP = evaluate_map(distmat, query_labels, gallery_labels)
    #     mAP1 = evaluate_map(distmat, query_labels, gallery_labels, top_k=1)
    #     mAP5 = evaluate_map(distmat, query_labels, gallery_labels, top_k=5)
    #     self.log('val/mAP', mAP)
    #     self.log('val/mAP1', mAP1)
    #     self.log('val/mAP5', mAP5)

    #     recall_at_k = evaluate_recall_at_k(distmat, query_labels, gallery_labels, k=5)
    #     self.log(f'val/Recall@5', recall_at_k)

    #     accuracy = wildlife_accuracy(query_embeddings, gallery_embeddings, query_labels, gallery_labels)
    #     self.log(f'val/accuracy', accuracy)
    def on_validation_epoch_end(self):
        # Concatenate embeddings and get string labels
        query_embeddings = torch.cat(self.query_embeddings)
        gallery_embeddings = torch.cat(self.gallery_embeddings)
        query_labels = np.array(self.query_labels)
        gallery_labels = np.array(self.gallery_labels)

        # Compute cosine similarity
        similarity = CosineSimilarity()(query_embeddings, gallery_embeddings)
        similarity_matrix = similarity['cosine']

        # Re-ranking if enabled
        if self.re_ranking:
            from utils.re_ranking import re_ranking
            dist_matrix = compute_distance_matrix(
                torch.from_numpy(query_embeddings), 
                torch.from_numpy(gallery_embeddings), 
                metric='euclidean'
            )
            reranked_dist = re_ranking(dist_matrix.numpy())
            similarity_matrix = -reranked_dist  # Convert distance to similarity

        # KNN Classifier
        classifier = KnnClassifier(k=1, database_labels=gallery_labels)
        predictions = classifier(similarity_matrix)

        # Compute metrics
        accuracy = accuracy_score(query_labels, predictions)
        precision = precision_score(query_labels, predictions, average='weighted', zero_division=1)
        recall = recall_score(query_labels, predictions, average='weighted', zero_division=1)
        f1 = f1_score(query_labels, predictions, average='weighted', zero_division=1)

        # Log metrics
        self.log('val/accuracy', accuracy, prog_bar=True)
        self.log('val/precision', precision)
        self.log('val/recall', recall)
        self.log('val/f1', f1)

    def configure_optimizers(self):
        if self.config:
            optimizer = get_optimizer(self.config, self.parameters())
            lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer
    

