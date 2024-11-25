# https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py

import argparse
from pprint import pprint
import yaml, shutil, os

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import wandb

from data.combined_datasets import get_dataset
from models.simple_model import SimpleModel
from models.fusion_model import FusionModel
from models.megadescriptor import MegaDescriptor
from models.transformer_model import TransformerModel
from models.efficientnet import EfficientNet
from models.resnet_plus_model import ResNetPlusModel
from models.triplet_loss_model import TripletModel
from utils.gradcam_callback import GradCAMCallback

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inference without GradCAM visualization.')
    parser.add_argument('--config', type=str, required=True, default="./config.yaml", help='Path to config yaml file')
    parser.add_argument('--gpu', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

def get_gpu_settings(
    gpu_ids: list[int], n_gpu: int
) -> tuple[str, int | list[int] | None, str | None]:
    """Get gpu settings for pytorch-lightning trainer:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags

    Args:
        gpu_ids (list[int])
        n_gpu (int)

    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    """
    if not torch.cuda.is_available():
        if n_gpu>0 and torch.backends.mps.is_available():
            return "mps", 1, None
        else:
            return "cpu", 1, None

    if gpu_ids is not None:
        devices = gpu_ids
        strategy = "ddp" if len(gpu_ids) > 1 else None
    elif n_gpu is not None:
        # int
        devices = n_gpu
        strategy = "ddp" if n_gpu > 1 else None
    else:
        devices = 1
        strategy = None

    torch.set_float32_matmul_precision('high')

    return "gpu", devices, strategy

if __name__ == '__main__':
    args = get_args()

    config_file_path = yaml.safe_load(args.config)
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    if type(config['only_cache']) != list:
        config['only_cache'] = [config['only_cache'], config['only_cache']]
     
    # setup dataset
    data =  get_dataset(config)

    model_classes = {
        'TripletModel': TripletModel,
        'FusionModel': FusionModel,
        'TransformerModel': TransformerModel,
        'EfficientNet': EfficientNet,
        'MegaDescriptor': MegaDescriptor,
    }
    model_name = config['model_architecture']
    model_class = model_classes.get(model_name)

    if model_class is None:
        raise ValueError(f"Unknown model architecture: {model_name}")

    # setup model
    if config['checkpoint']:  # pretrained=True because will load from checkpoint
        print(f"Loading model {model_name} from checkpoint")
        model = model_class(config=config, pretrained=False)
        if args.gpu:
            checkpoint = torch.load(config['checkpoint'])
            model.load_state_dict(checkpoint["state_dict"])
        else:
            checkpoint = torch.load(config['checkpoint'], map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(torch.device('cpu'))
    else:
        print(f"Testing {model_name} without any training.")
        model = model_class(config=config, pretrained=True)


    trainer = Trainer(accelerator="cpu")
    # trainer.fit(model, data)
    # trainer.test(model, dataloaders=data.val_dataloader(), ckpt_path=config['checkpoint'])
    trainer.validate(model, dataloaders=data.val_dataloader())
        