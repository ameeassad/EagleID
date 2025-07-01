import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall, F1Score
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import CoralLoss

class AgeModel(LightningModule):
    def __init__(self, config: dict, pretrained: bool = False, num_classes: int | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        if num_classes is None:
            num_classes = config.get('num_classes', 5)  # Default to 5 if not in config
        self.num_classes = num_classes

        # Backbone without classifier
        self.model = timm.create_model(model_name=config['backbone_name'], pretrained=pretrained, num_classes=0)

        # If skeleton, accept 4 channels instead of 3
        if config['preprocess_lvl'] == 3 and config['backbone_name'].startswith('resnet'):
            self.model.conv1 = nn.Conv2d(4, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size,
                                         stride=self.model.conv1.stride, padding=self.model.conv1.padding, bias=False)

        # CORAL layer: output K-1 logits for K ordinal classes
        feature_dim = self.model.num_features
        self.model.reset_classifier(0)  # Remove original classifier
        self.coral = CoralLayer(feature_dim, num_classes)

        # Class balancing weights (from config or uniform)
        class_weights = config.get('class_weights', [1.0] * (num_classes - 1))
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.coral_loss = CoralLoss(num_classes=num_classes, class_weights=self.class_weights)

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_top2 = Accuracy(task='multiclass', num_classes=num_classes, top_k=2)

    def forward(self, x):
        features = self.model.forward_features(x)
        logits = self.coral(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, target = batch
        logits = self(x)
        loss = self.coral_loss(logits, target)
        pred = (logits > 0).sum(dim=1)
        acc = self.train_acc(pred, target)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        logits = self(x)
        loss = self.coral_loss(logits, target)
        pred = (logits > 0).sum(dim=1)
        acc = self.val_acc(pred, target)
        precision = self.val_precision(pred, target)
        recall = self.val_recall(pred, target)
        f1 = self.val_f1(pred, target)
        top2 = self.val_top2(logits, target)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/precision', precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/recall', recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/top2_acc', top2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(self.config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config} 