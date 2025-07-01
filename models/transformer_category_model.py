import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanAbsoluteError, QuadraticWeightedKappa
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math


from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import CoralLoss
from coral_pytorch.dataset import levels_from_labels, coral_label_from_logits


class TransformerCategory(pl.LightningModule):
    """
    Transformer-based model with improved training strategies
    Uses cross-entropy loss for multi-class classification.
    """
    
    def __init__(self, 
                 backbone_model_name="swin_base_patch4_window7_224", 
                 img_size=224, 
                 num_classes=5, 
                 config=None, 
                 pretrained=True, 
                 lr=0.001, 
                 preprocess_lvl=0, 
                 outdir="results"):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        if config:
            self.backbone_model_name = config['backbone_name'] if config['backbone_name'] else 'swin_base_patch4_window7_224'
            self.num_classes = num_classes
            preprocess_lvl = int(config['preprocess_lvl'])
            img_size = config['img_size']
            self.lr = config['solver']['BASE_LR']
            outdir = config['outdir']
            
            # Get regularization and classifier hyperparameters from config with sensible defaults
            reg_cfg = config.get('regularization', {}) if config else {}
            self.drop_rate = reg_cfg.get('drop_rate', 0.1)
            self.drop_path_rate = reg_cfg.get('drop_path_rate', 0.05)
            self.gradient_clip_val = reg_cfg.get('gradient_clip_val', 1.0)
            self.dropout_rate1 = config.get('classifier', {}).get('dropout_rate1', 0.3) if config else 0.3
            self.dropout_rate2 = config.get('classifier', {}).get('dropout_rate2', 0.0) if config else 0.0
            self.hidden_dim = config.get('classifier', {}).get('hidden_dim', 512) if config else 512
            self.label_smoothing = config.get('classifier', {}).get('label_smoothing', 0.0) if config else 0.0
            
            # Training hyperparameters
            self.weight_decay = config.get('solver', {}).get('WEIGHT_DECAY', 0.05)  # Higher weight decay
            self.warmup_epochs = config.get('solver', {}).get('WARMUP_EPOCHS', 5)
            self.warmup_freeze = config.get('solver', {}).get('WARMUP_FREEZE', 5)  # Freeze backbone for first N epochs
            self.min_lr = config.get('solver', {}).get('MIN_LR', 1e-6)
            self.layer_decay = config.get('solver', {}).get('LAYER_DECAY', 0.65)  # LLRD factor
            self.head_lr_multiplier = config.get('solver', {}).get('HEAD_LR_MULTIPLIER', 5.0)  # Head LR multiplier
        else:
            self.backbone_model_name = backbone_model_name
            self.num_classes = num_classes
            preprocess_lvl = preprocess_lvl
            img_size = img_size
            self.lr = lr
            outdir = outdir
            
            # Better default values for transformer training
            self.drop_rate = 0.1
            self.drop_path_rate = 0.05
            self.gradient_clip_val = 1.0
            self.dropout_rate1 = 0.3
            self.dropout_rate2 = 0.0
            self.hidden_dim = 512
            self.label_smoothing = 0.0  # Removed label smoothing
            self.weight_decay = 0.05
            self.warmup_epochs = 5
            self.warmup_freeze = 5
            self.min_lr = 1e-6
            self.layer_decay = 0.65
            self.head_lr_multiplier = 5.0

        # Validate backbone model
        supported_models = [
            'swin_large_patch4_window7_224',
            'swin_base_patch4_window7_224', 
            'swin_large_patch4_window12_384',
            'swin_tiny_patch4_window7_224',
            'vit_base_patch16_224',
            'vit_large_patch16_224',
            'deit_base_patch16_224',
            'deit_large_patch16_224'
        ]
        
        if self.backbone_model_name not in supported_models:
            raise ValueError(f"Backbone model {self.backbone_model_name} not supported. Supported models: {supported_models}")

        # Create backbone model with config-driven regularization
        self.backbone = timm.create_model(
            self.backbone_model_name, 
            num_classes=0,  # Remove classification head
            pretrained=pretrained,
            drop_rate=self.drop_rate,  # Configurable dropout
            drop_path_rate=self.drop_path_rate  # Configurable stochastic depth
        )
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone.forward_features(dummy_input)
            if isinstance(features, (list, tuple)):
                # Some models return multiple feature levels
                features = features[-1]  # Use the last feature level
            feature_dim = features.shape[1] if len(features.shape) == 2 else features.flatten(1).shape[1]
        
        # Head: MLP before CoralLayer for more capacity
        self.head = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate1)
        )
        self.coral_layer = CoralLayer(self.hidden_dim, self.num_classes)

        # Loss function: CORAL for ordinal regression
        class_weights = None
        if config and 'class_weights' in config:
            class_weights = torch.tensor(config['class_weights'], dtype=torch.float32)
        self.criterion = CoralLoss(num_classes=self.num_classes, class_weights=class_weights)
        
        # Metrics for ordinal regression
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_mae = MeanAbsoluteError()
        self.val_qwk = QuadraticWeightedKappa(num_classes=self.num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_preds = []
        self.val_targets = []
        
        # Initialize weights with better initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with better strategies for transformers"""
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        logits = self.coral_layer(self.head(feats))
        return logits
    
    def on_train_start(self):
        """Freeze backbone for the first few epochs to let classifier learn first"""
        print(f"Freezing backbone for first {self.warmup_freeze} epochs")
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Log the freezing state
        if hasattr(self, 'log'):
            self.log('train/backbone_frozen', 1.0, prog_bar=False)
    
    def on_train_epoch_start(self):
        """Unfreeze backbone after warmup_freeze epochs"""
        if self.current_epoch == self.warmup_freeze:
            print(f"Unfreezing backbone at epoch {self.current_epoch}")
            for param in self.backbone.parameters():
                param.requires_grad = True
            
            # Log the unfreezing state
            if hasattr(self, 'log'):
                self.log('train/backbone_frozen', 0.0, prog_bar=False)
    
    def _decode(self, logits):
        """Convert cumulative logits → rank label (integer class index)"""
        return coral_label_from_logits(logits)

    def training_step(self, batch, batch_idx):
        """Training step for CORAL ordinal regression (no MixUp/CutMix)"""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = self._decode(logits)
        self.train_acc.update(preds, labels)
        # Log loss (this updates every step)
        self.log('train/loss_step',  loss, on_step=True, prog_bar=True,  logger=True)
        self.log('train/loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        """End of training epoch - compute and log training metrics"""
        # Compute training accuracy
        train_acc = self.train_acc.compute()
        self.log('train/acc', train_acc, prog_bar=True)
        
        # Reset training metrics for next epoch
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step for CORAL ordinal regression"""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = self._decode(logits)
        self.val_acc.update(preds, labels)
        self.val_mae.update(preds, labels)
        self.val_qwk.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())
        self.log('val/loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        """End of validation epoch - compute and log all metrics"""
        acc = self.val_acc.compute()
        mae = self.val_mae.compute()
        qwk = self.val_qwk.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        self.log('val/acc', acc, prog_bar=True)
        self.log('val/mae', mae, prog_bar=True)
        self.log('val/qwk', qwk, prog_bar=True)
        self.log('val/precision', precision, prog_bar=False)
        self.log('val/recall', recall, prog_bar=False)
        self.log('val/f1', f1, prog_bar=False)
        self.val_acc.reset()
        self.val_mae.reset()
        self.val_qwk.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        if len(self.val_preds) > 0:
            cm = confusion_matrix(self.val_targets, self.val_preds)
            self.val_preds = []
            self.val_targets = []
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Age Class')
        ax.set_ylabel('Actual Age Class')
        ax.set_title('Confusion Matrix - Age Classification')
        
        # Set age class labels
        age_labels = ['Juvenile', 'Sub-adult', 'Adult', 'Mature', 'Senior']
        ax.set_xticklabels(age_labels, rotation=45)
        ax.set_yticklabels(age_labels, rotation=0)
        
        plt.tight_layout()
        return fig
    
    def get_layer_groups(self):
        """Get parameter groups for layer-wise learning rate decay with higher head LR"""
        # Separate backbone and classifier parameters
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': self.lr * self.layer_decay},  # Lower LR for backbone
            {'params': classifier_params, 'lr': self.lr * self.head_lr_multiplier}  # Higher LR for classifier
        ]
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with proper transformer training"""
        if self.config:
            from utils.optimizer import get_optimizer, get_lr_scheduler_config
            # Use layer-wise learning rate decay when config is provided
            param_groups = self.get_layer_groups()
            optimizer = get_optimizer(self.config, param_groups)
            lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            # Use layer-wise learning rate decay
            param_groups = self.get_layer_groups()
            
            optimizer = torch.optim.AdamW(
                param_groups, 
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Cosine annealing with warmup scheduler
            def cosine_schedule_with_warmup(epoch):
                if epoch < self.warmup_epochs:
                    # Linear warmup
                    return epoch / self.warmup_epochs
                else:
                    # Cosine annealing
                    progress = (epoch - self.warmup_epochs) / (self.trainer.max_epochs - self.warmup_epochs)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=cosine_schedule_with_warmup
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            } 