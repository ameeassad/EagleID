import argparse
import shutil
import os
import yaml
import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning import Trainer
import numpy as np
from PIL import Image
import wandb
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from data.artportalen_goleag import ArtportalenDataModule
from models.simple_model import SimpleModel
from models.resnet_plus_model import ResNetPlusModel

# re-ranking option

def get_args():
    parser = argparse.ArgumentParser(description='Inference without GradCAM visualization.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config.')
    # parser.add_argument('--gpu', type=bool, default=False, help='Gpu true?.')
    parser.add_argument('--gpu', default=False, action=argparse.BooleanOptionalAction)
    
    return parser.parse_args()

def main():
    
    args = get_args()

    config_file_path = yaml.safe_load(args.config)
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    data = ArtportalenDataModule(data_dir=config['dataset'], preprocess_lvl=config['preprocess_lvl'], batch_size=config['batch_size'], size=config['img_size'], mean=config['transforms']['mean'], std=config['transforms']['std'])
    data.prepare_testing_data(config['dataset'])
    dataloader = data.test_dataloader()

    model = SimpleModel(config=config, pretrained=False, num_classes=data.num_classes)
    if args.gpu:
        checkpoint = torch.load(config['checkpoint'])
    else:
        checkpoint = torch.load(config['checkpoint'], map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device('cpu'))
    

    trainer = Trainer(accelerator="cpu")
    # trainer.fit(model, data)
    trainer.test(model, dataloaders=dataloader, ckpt_path=config['checkpoint'])
    trainer.validate(model, dataloaders=data.val_dataloader())

if __name__ == '__main__':
    main()
    # python inference.py --checkpoint checkpoints/model-j75sihxi-best-epoch049/model.ckpt --dataset testing/images --num-classes 5 --gpu False

