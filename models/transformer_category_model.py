import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

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
            
            # Get classifier hyperparameters from config with better defaults
            self.dropout_rate1 = config.get('classifier', {}).get('dropout_rate1', 0.5)  # Increased dropout
            self.dropout_rate2 = config.get('classifier', {}).get('dropout_rate2', 0.3)  # Additional dropout layer
            self.hidden_dim = config.get('classifier', {}).get('hidden_dim', 512)  # Larger hidden dim
            self.label_smoothing = config.get('classifier', {}).get('label_smoothing', 0.05)  # Reduced for small classes
            
            # Training hyperparameters
            self.weight_decay = config.get('solver', {}).get('WEIGHT_DECAY', 0.05)  # Higher weight decay
            self.warmup_epochs = config.get('solver', {}).get('WARMUP_EPOCHS', 5)
            self.min_lr = config.get('solver', {}).get('MIN_LR', 1e-6)
            self.layer_decay = config.get('solver', {}).get('LAYER_DECAY', 0.65)  # LLRD factor
        else:
            self.backbone_model_name = backbone_model_name
            self.num_classes = num_classes
            preprocess_lvl = preprocess_lvl
            img_size = img_size
            self.lr = lr
            outdir = outdir
            
            # Better default values for transformer training
            self.dropout_rate1 = 0.5
            self.dropout_rate2 = 0.3
            self.hidden_dim = 512
            self.label_smoothing = 0.05  # Reduced from 0.1
            self.weight_decay = 0.05
            self.warmup_epochs = 5
            self.min_lr = 1e-6
            self.layer_decay = 0.65

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

        # Create backbone model with better initialization
        self.backbone = timm.create_model(
            self.backbone_model_name, 
            num_classes=0,  # Remove classification head
            pretrained=pretrained,
            drop_rate=0.1,  # Add dropout to backbone
            drop_path_rate=0.1  # Add stochastic depth
        )
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone.forward_features(dummy_input)
            if isinstance(features, (list, tuple)):
                # Some models return multiple feature levels
                features = features[-1]  # Use the last feature level
            feature_dim = features.shape[1] if len(features.shape) == 2 else features.flatten(1).shape[1]
        
        # Enhanced classification head with better regularization
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(self.dropout_rate1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate2),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        # Loss function with reduced label smoothing for small class count
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Comprehensive metrics for 5-class classification - use proper epoch-level metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.top2_acc = Accuracy(task='multiclass', num_classes=self.num_classes, top_k=2)
        
        # Per-class metrics - use proper epoch-level metrics
        self.val_precision = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        
        # Store predictions for confusion matrix
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
        """Forward pass"""
        features = self.backbone.forward_features(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Use the last feature level
        
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.flatten(1)
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step with controlled augmentation"""
        images, labels = batch
        
        # Only apply MixUp/CutMix after warmup period to avoid early confusion
        current_epoch = self.current_epoch
        if (hasattr(self.trainer.datamodule.train_dataset, 'transform') and 
            hasattr(self.trainer.datamodule.train_dataset.transform, 'apply_mixup_cutmix') and
            current_epoch >= self.warmup_epochs):  # Only after warmup
            
            images, labels_a, labels_b, lam, aug_type = self.trainer.datamodule.train_dataset.transform.apply_mixup_cutmix(images, labels)
            
            # Forward pass
            logits = self(images)
            
            # Compute loss with MixUp/CutMix
            loss_a = self.criterion(logits, labels_a)
            loss_b = self.criterion(logits, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            
            # Update accuracy metric (use original labels for accuracy)
            self.train_acc.update(logits.softmax(dim=-1), labels_a)
            
            # Log augmentation type
            self.log('train/aug_type', 1.0 if aug_type != 'none' else 0.0, prog_bar=False)
        else:
            # Standard training without aggressive augmentation during warmup
            logits = self(images)
            loss = self.criterion(logits, labels)
            self.train_acc.update(logits.softmax(dim=-1), labels)
        
        # Log loss (this updates every step)
        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """End of training epoch - compute and log training metrics"""
        # Compute training accuracy
        train_acc = self.train_acc.compute()
        self.log('train/acc', train_acc, prog_bar=True)
        
        # Reset training metrics for next epoch
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step with comprehensive metrics"""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Update metrics (they will be computed at epoch end)
        self.val_acc.update(logits.softmax(dim=-1), labels)
        self.top2_acc.update(logits.softmax(dim=-1), labels)
        self.val_precision.update(logits.softmax(dim=-1), labels)
        self.val_recall.update(logits.softmax(dim=-1), labels)
        self.val_f1.update(logits.softmax(dim=-1), labels)
        
        # Store predictions for confusion matrix
        preds = logits.argmax(dim=-1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())
        
        # Log loss (this updates every step)
        self.log('val/loss', loss, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """End of validation epoch - compute and log all metrics"""
        # Compute all metrics
        acc = self.val_acc.compute()
        top2_acc = self.top2_acc.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        
        # Log all metrics
        self.log('val/acc', acc, prog_bar=True)
        self.log('val/top2_acc', top2_acc, prog_bar=True)
        self.log('val/precision', precision, prog_bar=False)
        self.log('val/recall', recall, prog_bar=False)
        self.log('val/f1', f1, prog_bar=False)
        
        # Reset metrics for next epoch
        self.val_acc.reset()
        self.top2_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        
        # Calculate confusion matrix if we have predictions
        if len(self.val_preds) > 0:
            # Calculate confusion matrix
            cm = confusion_matrix(self.val_targets, self.val_preds)
            
            # Log confusion matrix as figure if logger is available
            # if hasattr(self, 'logger') and self.logger:
            #     try:
            #         fig = self._plot_confusion_matrix(cm)
            #         self.logger.experiment.add_figure('confusion_matrix', fig, self.current_epoch)
            #     except Exception as e:
            #         print(f"Warning: Could not plot confusion matrix: {e}")
            
            # Reset for next epoch
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
        """Get parameter groups for layer-wise learning rate decay"""
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
            {'params': classifier_params, 'lr': self.lr}  # Higher LR for classifier
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