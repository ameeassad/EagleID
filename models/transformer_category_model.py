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

class TransformerCategory(pl.LightningModule):
    """
    Transformer-based model
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
            
            # Get classifier hyperparameters from config
            self.dropout_rate1 = config.get('classifier', {}).get('dropout_rate1', 0.3)
            self.hidden_dim = config.get('classifier', {}).get('hidden_dim', 256)
            self.label_smoothing = config.get('classifier', {}).get('label_smoothing', 0.1)
        else:
            self.backbone_model_name = backbone_model_name
            self.num_classes = num_classes
            preprocess_lvl = preprocess_lvl
            img_size = img_size
            self.lr = lr
            outdir = outdir
            
            # Default values if no config provided
            self.dropout_rate1 = 0.3
            self.hidden_dim = 256
            self.label_smoothing = 0.1

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

        # Create backbone model
        self.backbone = timm.create_model(
            self.backbone_model_name, 
            num_classes=0,  # Remove classification head
            pretrained=pretrained
        )
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone.forward_features(dummy_input)
            if isinstance(features, (list, tuple)):
                # Some models return multiple feature levels
                features = features[-1]  # Use the last feature level
            feature_dim = features.shape[1] if len(features.shape) == 2 else features.flatten(1).shape[1]
        
        # Simplified classification head for 5 classes
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate1),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Comprehensive metrics for 5-class classification
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.top2_acc = Accuracy(task='multiclass', num_classes=self.num_classes, top_k=2)
        
        # Per-class metrics
        self.val_precision = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        
        # Store predictions for confusion matrix
        self.val_preds = []
        self.val_targets = []
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the classifier layers"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
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
        """Training step with support for MixUp and CutMix"""
        images, labels = batch
        
        # Apply MixUp or CutMix if the transform supports it
        if hasattr(self.trainer.datamodule.train_dataset, 'transform') and hasattr(self.trainer.datamodule.train_dataset.transform, 'apply_mixup_cutmix'):
            images, labels_a, labels_b, lam, aug_type = self.trainer.datamodule.train_dataset.transform.apply_mixup_cutmix(images, labels)
            
            # Forward pass
            logits = self(images)
            
            # Compute loss with MixUp/CutMix
            loss_a = self.criterion(logits, labels_a)
            loss_b = self.criterion(logits, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            
            # Calculate accuracy (use original labels for accuracy)
            acc = self.train_acc(logits.softmax(dim=-1), labels_a)
            
            # Log augmentation type
            self.log('train/aug_type', 1.0 if aug_type != 'none' else 0.0, prog_bar=False)
        else:
            # Standard training without MixUp/CutMix
            logits = self(images)
            loss = self.criterion(logits, labels)
            acc = self.train_acc(logits.softmax(dim=-1), labels)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with comprehensive metrics"""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate various accuracy metrics
        acc = self.val_acc(logits.softmax(dim=-1), labels)
        top2_acc = self.top2_acc(logits.softmax(dim=-1), labels)
        
        # Calculate precision, recall, and F1
        precision = self.val_precision(logits.softmax(dim=-1), labels)
        recall = self.val_recall(logits.softmax(dim=-1), labels)
        f1 = self.val_f1(logits.softmax(dim=-1), labels)
        
        # Store predictions for confusion matrix
        preds = logits.argmax(dim=-1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())
        
        # Log comprehensive metrics
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        self.log('val/top2_acc', top2_acc, prog_bar=True)
        self.log('val/precision', precision, prog_bar=False)
        self.log('val/recall', recall, prog_bar=False)
        self.log('val/f1', f1, prog_bar=False)
        
        return loss
    
    def on_validation_epoch_end(self):
        """End of validation epoch - compute confusion matrix"""
        if len(self.val_preds) > 0:
            # Calculate confusion matrix
            cm = confusion_matrix(self.val_targets, self.val_preds)
            
            # # Log confusion matrix as figure
            # if hasattr(self, 'logger') and self.logger:
            #     fig = self._plot_confusion_matrix(cm)
            #     self.logger.experiment.add_figure('confusion_matrix', fig, self.current_epoch)
            
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
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        if self.config:
            from utils.optimizer import get_optimizer, get_lr_scheduler_config
            optimizer = get_optimizer(self.config, self.parameters())
            lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            } 