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

class ResNetPlusModel(LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = True, # Use ImageNet pre-trained weights
        num_classes: int | None = None,
        outdir: str = 'results',
        frozen_layers: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=0)  # No classification head yet

        # Freeze the ResNet backbone (except last 3 layers)
        self.frozen_layers = frozen_layers
        self.resnet_layers = list(self.model.named_parameters())
        for name, param in self.resnet_layers[:-1*self.frozen_layers]:
            param.requires_grad = False

        # Bottleneck
        num_bottleneck = 512
        self.fc = nn.Sequential(
            nn.Linear(2048, num_bottleneck),
            nn.BatchNorm1d(num_bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc.apply(weights_init_kaiming)  # Apply Kaiming initialization

        # Classifier 
        self.classifier = nn.Sequential(
            nn.Linear(num_bottleneck, num_classes)
        )
        self.classifier.apply(weights_init_classifier)  # Apply classifier initialization
        
        self.train_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_loss = nn.CrossEntropyLoss()
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.gradient = None
        self.outdir = outdir
    
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
         # Forward pass through ResNet-50 backbone
        features = self.model(x)

        x = features.view(features.size(0), -1) # Flatten output
        x = self.fc(x)  # Bottleneck
        x = self.classifier(x)  # Classifier

        return x

    def training_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        _, pred = out.max(1)

        loss = self.train_loss(out, target)
        acc = self.train_acc(pred, target)
        self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if config['use_gradcam']:
            for param in self.model.parameters():
                param.requires_grad = True
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
            # Re-freeze the model parameters after computing the Grad-CAM
            for name, param in self.resnet_layers[:-1*self.frozen_layers]:
                param.requires_grad = False
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
        # Only optimize the parameters of the added fully connected layers
        optimizer = get_optimizer(config, filter(lambda p: p.requires_grad, self.parameters()))
        lr_scheduler_config = get_lr_scheduler_config(config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
