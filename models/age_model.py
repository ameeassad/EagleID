import torch, torch.nn as nn, timm
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanAbsoluteError
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import CoralLoss
from coral_pytorch.dataset import levels_from_labelbatch

class AgeModel(pl.LightningModule):
    def __init__(self, config: dict, pretrained: bool = False, num_classes: int | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.num_classes = num_classes or config.get("num_classes", 5)
        
        # Two-stage training parameters
        self.coral_warmup_epochs = config.get("coral_warmup_epochs", 3)
        self.coral_warmup_lr = config.get("coral_warmup_lr", 0.001)

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
        
        # Initialize CORAL layer properly for two-stage training
        self._init_coral_layer()

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
        self.val_mae   = MeanAbsoluteError()



    # ---------- helpers ----------
    @staticmethod
    def logits_to_pred(logits):
        """Convert CORAL logits to predicted class index (0-indexed)."""
        # logits: [B, K-1]
        prob_gt = torch.sigmoid(logits)              # P(y > t)
        
        # Correct CORAL decoder: P(y = k) = P(y > k-1) - P(y > k)
        # For K classes, we need K-1 thresholds
        # P(y = 0) = 1 - P(y > 0)
        # P(y = k) = P(y > k-1) - P(y > k) for 0 < k < K-1
        # P(y = K-1) = P(y > K-2)
        
        batch_size = logits.shape[0]
        num_classes = logits.shape[1] + 1  # K = K-1 + 1
        
        # Initialize class probabilities
        class_probs = torch.zeros(batch_size, num_classes, device=logits.device)
        
        # P(y = 0) = 1 - P(y > 0)
        class_probs[:, 0] = 1.0 - prob_gt[:, 0]
        
        # P(y = k) = P(y > k-1) - P(y > k) for 0 < k < K-1
        for k in range(1, num_classes - 1):
            class_probs[:, k] = prob_gt[:, k-1] - prob_gt[:, k]
        
        # P(y = K-1) = P(y > K-2)
        class_probs[:, -1] = prob_gt[:, -1]
        
        # Debug: Print logits and probabilities for first few samples
        if logits.shape[0] > 0:
            print(f"DEBUG: Logits shape: {logits.shape}")
            print(f"DEBUG: Sample logits: {logits[0]}")
            print(f"DEBUG: Sample prob_gt: {prob_gt[0]}")
            print(f"DEBUG: Sample class_probs: {class_probs[0]}")
            print(f"DEBUG: Sample argmax: {class_probs[0].argmax().item()}")
        
        return class_probs.argmax(dim=1)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.ndim == 4:
            features = features.mean(dim=[2, 3])  # Global average pool
        logits = self.coral(features)
        
        # Debug: Print feature and logit statistics
        if x.shape[0] > 0 and self.current_epoch == 0:
            print(f"DEBUG: Features shape: {features.shape}")
            print(f"DEBUG: Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
            print(f"DEBUG: Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
        
        return logits
    
    def _init_coral_layer(self):
        """Initialize CORAL layer for two-stage training"""
        # Initialize weights to small values for better threshold learning
        if hasattr(self.coral, 'weight'):
            torch.nn.init.xavier_uniform_(self.coral.weight, gain=0.1)
        
        # Put the thresholds on the log-odds scale where you expect them to lie
        if hasattr(self.coral, 'bias') and self.coral.bias is not None:
            # e.g. [-2, -1, 0, +1]       (σ ~ [0.12, 0.27, 0.50, 0.73])
            init = torch.linspace(2, -2, steps=self.num_classes - 1)
            self.coral.bias.data.copy_(init)
            print("CORAL bias initialised to:", init.cpu().tolist())
        else:
            print("CORAL layer has no bias parameter")
        
        print(f"Initialized CORAL layer for two-stage training")
    
    def on_train_start(self):
        """Start two-stage training: freeze backbone, train CORAL layer first"""
        print(f"Starting two-stage training: CORAL warmup for {self.coral_warmup_epochs} epochs")
        
        # Freeze backbone completely
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen for CORAL layer training")
        
        # Log the freezing state
        if hasattr(self, 'log'):
            self.log('train/backbone_frozen', 1.0, prog_bar=False)
    
    def on_train_epoch_start(self):
        """Unfreeze backbone after CORAL warmup epochs"""
        if self.current_epoch == self.coral_warmup_epochs:
            print(f"Unfreezing backbone at epoch {self.current_epoch} after CORAL warmup")
            for param in self.backbone.parameters():
                param.requires_grad = True
            
            # Log the unfreezing state
            if hasattr(self, 'log'):
                self.log('train/backbone_frozen', 0.0, prog_bar=False)

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
        assert y.min() >= 0 and y.max() < self.num_classes, f"Labels must be in [0, {self.num_classes-1}], got min {y.min().item()}, max {y.max().item()}"
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
        
        # Ensure MAE metric is on the same device as predictions
        if self.val_mae.device != pred.device:
            self.val_mae = self.val_mae.to(pred.device)
        self.val_mae.update(pred, y)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        mae = self.val_mae.compute()
        self.log_dict({
            "val/acc":  self.val_acc.compute(),
            "val/precision": self.val_prec.compute(),
            "val/recall":    self.val_rec.compute(),
            "val/f1":        self.val_f1.compute(),
            "val/top2_acc":  self.val_top2.compute(),
        }, prog_bar=True)
        self.log('val/mae', mae, prog_bar=True)
        for m in (self.val_acc, self.val_prec, self.val_rec, self.val_f1, self.val_top2, self.val_mae):
            m.reset()

    # ---------- optim ---------
    def configure_optimizers(self):
        from utils.optimizer import get_optimizer, get_lr_scheduler_config
        
        # Create two parameter groups with different learning rates
        backbone_params = [p for n, p in self.named_parameters() if "backbone" in n]
        head_params = [p for n, p in self.named_parameters() if "coral" in n]
        
        base_lr = self.config["solver"]["BASE_LR"]
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": base_lr},        # e.g. 1e-3
            {"params": head_params, "lr": base_lr * 0.01},     # e.g. 1e-5 (100x smaller)
        ])
        
        # param group 0 → backbone, group 1 → coral
        # freeze backbone LR for warm-up
        lambda_fns = [
            lambda epoch: 0.0 if epoch < self.coral_warmup_epochs else 1.0,
            lambda epoch: 1.0,
        ]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_fns)
        return {"optimizer": opt, "lr_scheduler": scheduler}
