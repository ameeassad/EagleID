import os
import numpy as np
import yaml

import wandb
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchmetrics import Accuracy

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import ArtportalenDataModule, unnormalize

from utils import TripletLoss
from utils import get_optimizer, get_lr_scheduler_config, weights_init_kaiming, weights_init_classifier

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class TripletLossModel(LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = False,
        num_classes: int | None = None,
        outdir: str = 'results',
        margin: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        
        self.train_loss = TripletLoss(margin=margin) 
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_loss = TripletLoss(margin=margin)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.gradient = None
        self.outdir = outdir

        self.triplet_loss = TripletLoss(margin=margin)
    
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        for name, module in self.model.named_modules():
            if name == 'layer4':
                x = module(x)
                x.register_hook(self.activations_hook)
                return x
        return None

    def forward(self, x):
        # Get features from the backbone
        features = self.model(x)
        # Optionally normalize features for triplet loss
        normalized_features = nn.functional.normalize(features, p=2, dim=1)
        # Get classification logits
        logits = self.fc(features)
        return logits, normalized_features


    def training_step(self, batch, batch_idx):
        x, target = batch
        logits, features = self(x)

        loss, _, _ = self.train_loss(features, target)
        
        # Accuracy
        _, pred = logits.max(1)
        acc = self.train_acc(pred, target)

        self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if config['use_gradcam']:
            with torch.enable_grad():
                x, target = batch
                out = self(x)
                _, pred = out.max(1)

                loss = self.val_loss(out, target)
                acc = self.val_acc(pred, target)
                self.log_dict({'val/loss': loss, 'val/acc': acc})

                unnormalized_x = unnormalize(x[0].cpu(), config['transforms']['mean'], config['transforms']['std']).permute(1, 2, 0).numpy()
                unnormalized_x = np.clip(unnormalized_x, 0, 1)  # Ensure the values are within [0, 1]


                cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
                targets = [ClassifierOutputTarget(class_idx) for class_idx in target]
                grayscale_cam = cam(input_tensor=x, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(unnormalized_x, grayscale_cam, use_rgb=True)
                img = Image.fromarray((visualization * 255).astype(np.uint8))

                # Log image to 
                if config['use_wandb']:
                    wandb_img = wandb.Image(visualization, caption=f"GradCAM Batch {batch_idx} Image 0")
                    self.logger.experiment.log({"GradCAM Images": wandb_img})

                
                # save locally
                os.makedirs(self.outdir, exist_ok=True)
                img.save(os.path.join(self.outdir, f'cam_image_val_batch{batch_idx}_img0.png'))
                
                # To save all images in batch:
                # for i in range(len(x)):
                #     grayscale_cam_img = 
                # grayscale_cam[i]
                #     visualization = show_cam_on_image(x[i].cpu().numpy().transpose(1, 2, 0), grayscale_cam_img, use_rgb=True)
                #     img = Image.fromarray((visualization * 255).astype(np.uint8))
                #     os.makedirs(self.hparams.outdir, exist_ok=True)
                #     img.save(os.path.join(self.hparams.outdir, f'cam_image_val_batch{batch_idx}_img{i}.png'))
                
                # self.model.train()
        else:
            x, target = batch
            out = self(x)
            _, pred = out.max(1)

            loss = self.val_loss(out, target)
            acc = self.val_acc(pred, target)
            self.log_dict({'val/loss': loss, 'val/acc': acc})

    def test_step(self, batch, batch_idx):
            x, target = batch

            x = x.to(torch.device('cpu'))
            target = target.to(torch.device('cpu'))

            x.requires_grad = True
        
            out = self(x)

            _, pred = out.max(1)
            if pred.numel() == 1:
                print(f"BATCH {batch_idx} PREDICTION: {pred.item()}")
            else:
                print(f"BATCH {batch_idx} PREDICTIONS: {pred.tolist()}")

    
    def configure_optimizers(self):
        optimizer = get_optimizer(config, self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}