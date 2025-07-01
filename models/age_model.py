import torch, torch.nn as nn, timm
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import CoralLoss
from coral_pytorch.dataset import levels_from_labelbatch

class AgeModel(pl.LightningModule):
    def __init__(self, config: dict, pretrained: bool = False, num_classes: int | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.num_classes = num_classes or config.get("num_classes", 5)

        # ---------------- backbone -----------------
        self.backbone = timm.create_model(
            config["backbone_name"],
            pretrained=pretrained,
            num_classes=0          # strip original head
        )
        feat_dim = self.backbone.num_features

        # accept 4-channel input if needed
        if (config["preprocess_lvl"] == 3
                and config["backbone_name"].startswith("resnet")):
            self.backbone.conv1 = nn.Conv2d(
                4, self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=False,
            )

        # ---------------- CORAL head ----------------
        self.coral = CoralLayer(feat_dim, self.num_classes)

        # ---------------- loss ----------------------
        self.coral_loss = CoralLoss()          # no kwargs
        # optional manual class weighting (vector len = K)
        cw = torch.tensor(config.get("class_weights",
                                     [1.0] * self.num_classes))
        self.register_buffer("class_weights", cw)

        # ---------------- metrics ------------------
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_prec  = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_rec   = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1    = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_top2  = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=2)

    # ---------- forward ----------
    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.ndim == 4:
            features = features.mean(dim=[2, 3])  # Global average pool
        logits = self.coral(features)
        return logits

    # ---------- helpers ----------
    @staticmethod
    def logits_to_pred(logits):
        """Convert CORAL logits to predicted class index (0-indexed)."""
        return (torch.sigmoid(logits) > 0.5).sum(1) - 1

    # ---------- training ----------
    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        # Ensure y is 0-indexed for levels_from_labelbatch
        assert y.min() >= 0, f"Labels must be 0-indexed, got min label {y.min()}"
        levels = levels_from_labelbatch(y, self.num_classes).to(logits.device)

        loss = self.coral_loss(logits, levels)
        loss = loss * self.class_weights[y]      # manual weighting (optional)
        loss = loss.mean()

        pred = self.logits_to_pred(logits)
        acc  = self.train_acc(pred, y)

        self.log("train/loss", loss,  on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  acc,   on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ---------- validation ----------
    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        levels = levels_from_labelbatch(y, self.num_classes).to(logits.device)

        loss = self.coral_loss(logits, levels).mean()
        pred = self.logits_to_pred(logits)

        # metrics
        self.val_acc.update(pred, y)
        self.val_prec.update(pred, y)
        self.val_rec.update(pred, y)
        self.val_f1.update(pred, y)
        self.val_top2.update(torch.softmax(torch.nn.functional.pad(logits, (0,1), value=0),
                                           dim=-1), y)   # rebuild 5-way probs for top-k

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict({
            "val/acc":  self.val_acc.compute(),
            "val/precision": self.val_prec.compute(),
            "val/recall":    self.val_rec.compute(),
            "val/f1":        self.val_f1.compute(),
            "val/top2_acc":  self.val_top2.compute(),
        }, prog_bar=True)
        for m in (self.val_acc, self.val_prec, self.val_rec, self.val_f1, self.val_top2):
            m.reset()

    # ---------- optim ---------
    def configure_optimizers(self):
        from utils.optimizer import get_optimizer, get_lr_scheduler_config
        opt = get_optimizer(self.config, self.parameters())
        sched = get_lr_scheduler_config(self.config, opt)
        return {"optimizer": opt, "lr_scheduler": sched}
