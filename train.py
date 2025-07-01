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
from models.age_model import AgeModel
from models.fusion_model import FusionModel
from models.megadescriptor import MegaDescriptor
from models.transformer_model import TransformerModel
from models.transformer_category_model import TransformerCategory
from models.efficientnet import EfficientNet
from models.resnet_plus_model import ResNetPlusModel
from models.triplet_loss_model import TripletModel
from utils.gradcam_callback import GradCAMCallback
from utils.viz_callback import SimilarityVizCallback
from utils.augmentation_callback import AugmentationCallback

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument(
        '--config', type=str, required=True, default="./config.yaml", help='Path to config yaml file'
    )
    # parser.add_argument(
    #     '--test-local', action='store_true', help='Test local training'
    # )
    args = parser.parse_args()
    return args

def get_basic_callbacks(checkpoint_interval: int = 1) -> list:
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    
    # Determine the monitor metric based on model type
    if config.get('model_architecture') in ['TransformerCategory', 'SimpleModel']:
        monitor_metric = 'val/acc'
        monitor_mode = 'max'
    else:
        monitor_metric = 'val/mAP'
        monitor_mode = 'max'
    
    ckpt_callback = ModelCheckpoint(
        dirpath=f"checkpoints-{config['project_name']}",
        filename='epoch{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=monitor_metric,
        mode=monitor_mode,
        every_n_epochs=checkpoint_interval,
        save_last=True,
    )
    callbacks = [ckpt_callback, lr_callback]

    # Add augmentation callback for transformer models
    if config.get('model_architecture') == 'TransformerCategory':
        # Enable advanced augmentations after epoch 10
        aug_callback = AugmentationCallback(enable_epoch=10)
        callbacks.append(aug_callback)

    if config['early_stopping']['enabled']==True:
        early_stop_callback = EarlyStopping(
            monitor=config['early_stopping']['monitor'],  # Monitored metric
            min_delta=config['early_stopping']['min_delta'],      # Minimum change to qualify as an improvement
            patience=config['early_stopping']['patience'],         # Number of epochs with no improvement after which training will be stopped
            verbose=config['early_stopping']['verbose'],
            mode=config['early_stopping']['mode']           # Mode for the monitored metric ('min' or 'max')
        )
        callbacks.append(early_stop_callback)

    if config['use_gradcam']:
        gradcam_callback = GradCAMCallback(
            model=model, 
            config=config, 
            outdir=config['outdir'], 
            log_every_n_epochs=10 
        )
        callbacks.append(gradcam_callback)

    if config['val_viz']:
        viz_callback = SimilarityVizCallback(
                    config=config, 
                    outdir=config['outdir'], 
                    log_every_n_epochs=1 
                )
        callbacks.append(viz_callback)


    return callbacks


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

    torch.set_float32_matmul_precision('high') # prioritizes speed over reproducibility
    # torch.set_float32_matmul_precision('highest') #slower but more deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning

    return "gpu", devices, strategy


def get_trainer(config) -> Trainer:
    callbacks = get_basic_callbacks(checkpoint_interval=int(config['save_interval']))
    accelerator, devices, strategy = get_gpu_settings(config['gpu_ids'], config['n_gpu'])

    if config['use_wandb']:
        wandb_logger = WandbLogger(project=config['project_name'], log_model=True)
        wandb_logger.watch(model, log='all', log_freq=10)
        # add multiple hyperparameters
        # wandb_logger.experiment.config.update({"model_architecture": config['model_architecture'], 
        #                                         "checkpoint": config['checkpoint'],
        #                                         "preprocess_lvl": config['preprocess_lvl'],
        #                                         "batch_size": config['batch_size'],
        #                                         "img_size": config['img_size'], 
        #                                         "seed": config['seed'],
        #                                         "transforms": str(config['transforms']['mean']) + " / " + str(config['transforms']['std']),
        #                                         "optimizer": config['solver']['OPT'],
        #                                         "weight_decay": config['solver']['WEIGHT_DECAY'],
        #                                         "momentum": config['solver']['MOMENTUM'],
        #                                         "base_lr": config['solver']['BASE_LR'],
        #                                         "lr_scheduler": config['solver']['LR_SCHEDULER'],
        #                                         "lr_decay_rate": config['solver']['LR_DECAY_RATE'],
        #                                         "lr_step_size": config['solver']['LR_STEP_SIZE'],
        #                                         "lr_step_milestones": config['solver']['LR_STEP_MILESTONES']
        #                                         })
    else:
        wandb_logger = None

    trainer_args = {
        'max_epochs': config['epochs'],
        'log_every_n_steps': 1,
        'callbacks': callbacks,
        'default_root_dir': config['outdir'],
        'accelerator': accelerator,
        'devices': devices,
        'logger': wandb_logger,
        'deterministic': True,
        'profiler': 'simple',
        'num_sanity_val_steps': -1, # -1 to check all validation data, 0 to turn off
        'precision': "32-true",
    }

    if strategy is not None:
        trainer_args['strategy'] = strategy

    trainer = Trainer(**trainer_args)
    return trainer


if __name__ == '__main__':
    args = get_args()

    config_file_path = yaml.safe_load(args.config)
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    seed_everything(config['seed'], workers=True)

    if type(config['only_cache']) != list:
        config['only_cache'] = [config['only_cache'], config['only_cache']]

    if config['use_wandb']:
        wandb.init(project=config['project_name'])
        wandb.config.update(config)
        print(config)
        wandb.define_metric("val/mAP", summary="max")
        

    # setup dataset
    data =  get_dataset(config)
    
    model_classes = {
        'TripletModel': TripletModel,
        'FusionModel': FusionModel,
        'TransformerModel': TransformerModel,
        'TransformerCategory': TransformerCategory,
        'SimpleModel': SimpleModel,
        'EfficientNet': EfficientNet,
        'MegaDescriptor': MegaDescriptor,
        'ResNetPlusModel': ResNetPlusModel,
        'AgeModel': AgeModel,
    }
    model_name = config['model_architecture']
    model_class = model_classes.get(model_name)

    if model_class is None:
        raise ValueError(f"Unknown model architecture: {model_name}")

    # setup model
    if config['checkpoint']:  # pretrained=True because will load from checkpoint
        print(f"Loading model {model_name} from checkpoint")
        model = model_class(config=config, pretrained=False)
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print(f"Start training {model_name} from pretrained model")
        model = model_class(config=config, pretrained=True)

    trainer = get_trainer(config)

    print('Args:')
    pprint(args.__dict__)
    print('configuration:')
    pprint(config)

    trainer.fit(model, data)

        