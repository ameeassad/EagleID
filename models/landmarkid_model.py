import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchvision.ops import MLP

class LandmarkIDModel(pl.LightningModule):
    def __init__(
        self,
        num_landmarks: int = 5,  # Number of landmark heatmaps (k)
        backbone_name: str = "resnet50",
        embedding_size: int = 256,
        num_classes: int = 750,  # For cross-entropy loss
        alpha: float = 1.0,  # Heatmap reconstruction loss weight
        beta: float = 0.0005,  # Center loss weight
        pretrained: bool = True,
        lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1. Modified Backbone (3 + k input channels)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,  # We handle initialization manually
            num_classes=0,
            in_chans=3 + num_landmarks,  # RGB + heatmaps
            global_pool="",
            features_only=False,
        )
        
        # Initialize first layer: pretrained RGB weights + random heatmap channels
        if pretrained:
            pretrained_model = timm.create_model(backbone_name, pretrained=True)
            old_weights = pretrained_model.conv1.weight.data
            new_weights = self.backbone.conv1.weight.data.clone()
            # Copy pretrained RGB weights (first 3 channels)
            new_weights[:, :3, :, :] = old_weights
            # Random initialization for landmark channels (remaining k channels)
            nn.init.kaiming_normal_(new_weights[:, 3:, :, :], mode="fan_out")
            self.backbone.conv1.weight = nn.Parameter(new_weights)

        # 2. Embedding Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(self.backbone.num_features, embedding_size)
        
        # 3. Heatmap Reconstruction Decoder
        self.heatmap_decoder = nn.Sequential(
            nn.Conv2d(embedding_size, 256, 1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_landmarks, 1),
            nn.Sigmoid(),  # Output heatmaps in [0,1]
        )

        # 4. Loss Components
        self.loss_fn = {
            "triplet": losses.TripletMarginLoss(),
            "center": losses.CenterLoss(num_classes, embedding_size),
            "ce": nn.CrossEntropyLoss(label_smoothing=0.1),
            "recon": nn.BCELoss(),
        }

        # Track training stage
        self.current_stage = "stage1"
        self.automatic_optimization = False  # Manual optimization for flexibility

    def forward(self, x):
        # Input x: (B, 3+k, H, W) - RGB + heatmaps
        features = self.backbone.forward_features(x)
        pooled = self.global_pool(features).flatten(1)
        embeddings = self.embedding(pooled)
    

    def configure_optimizers(self):
        # Will be reinitialized when stage changes
        return None

    def on_train_epoch_start(self):
        # Stage transitions
        if self.current_epoch == self.hparams.stage1_epochs:
            self._enter_stage2a()
        elif self.current_epoch == self.hparams.stage1_epochs + self.hparams.stage2_decoder_epochs:
            self._enter_stage2b()

    def _enter_stage1(self):
        """Initial stage: train only first conv + classifier"""
        self.current_stage = "stage1"
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Unfreeze first conv layer and classifier
        for param in self.backbone.conv1.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.backbone.conv1.parameters()) + 
            list(self.classifier.parameters()),
            lr=self.hparams.lr
        )

    def _enter_stage2a(self):
        """Stage 2a: train only decoder"""
        self.current_stage = "stage2a"
        
        # Freeze all except decoder
        for param in self.parameters():
            param.requires_grad = False
        for param in self.heatmap_decoder.parameters():
            param.requires_grad = True
            
        self.optimizer = torch.optim.Adam(
            self.heatmap_decoder.parameters(),
            lr=self.hparams.lr
        )

    def _enter_stage2b(self):
        """Stage 2b: train entire model with lower LR"""
        self.current_stage = "stage2b"
        
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True
            
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr * 0.1  # 10x smaller LR
        )

    def training_step(self, batch, batch_idx):
        # Manually handle optimization
        opt = self.optimizer
        opt.zero_grad()
        
        # Forward pass
        loss = self._calculate_loss(batch)
        
        # Backward and optimize
        loss.backward()
        opt.step()
        
        return loss

    def _calculate_loss(self, batch):
        # Implement your loss calculations here
        x, heatmaps_gt, labels = batch
        embeddings = self(x)
        
        # Stage-specific losses
        if self.current_stage == "stage1":
            return self._reid_loss(embeddings, labels)
            
        # Stage2 losses include reconstruction
        reconstructed = self.heatmap_decoder(embeddings)
        return self._reid_loss(embeddings, labels) + \
               self._reconstruction_loss(reconstructed, heatmaps_gt)