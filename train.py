# https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py

import argparse
from pprint import pprint
import yaml, shutil, os

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from data.combined_datasets import get_dataset
from models.simple_model import SimpleModel
from models.resnet_plus_model import ResNetPlusModel
from models.triplet_loss_model import TripletModel
from utils.gradcam_callback import GradCAMCallback
from sklearn.model_selection import ParameterGrid

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument(
        '--config', type=str, required=True, default="./config.yaml", help='Path to config yaml file'
    )
    args = parser.parse_args()
    return args

def get_basic_callbacks(checkpoint_interval: int = 1) -> list:
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='epoch{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=1,
        every_n_epochs=checkpoint_interval,
        save_last=True,
    )
    callbacks = [ckpt_callback, lr_callback]

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
            log_every_n_epochs=1 
        )
        callbacks.append(gradcam_callback)

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
        if torch.backends.mps.is_available():
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


def get_trainer(config) -> Trainer:
    callbacks = get_basic_callbacks(checkpoint_interval=int(config['save_interval']))
    accelerator, devices, strategy = get_gpu_settings(config['gpu_ids'], config['n_gpu'])

    if config['use_wandb']:
        wandb_logger = WandbLogger(project=config['project_name'], log_model=True)
        wandb_logger.watch(model, log='all', log_freq=100)
        # add multiple hyperparameters
        wandb_logger.experiment.config.update({"model_architecture": config['model_architecture'], 
                                                "checkpoint": config['checkpoint'],
                                                "preprocess_lvl": config['preprocess_lvl'],
                                                "batch_size": config['batch_size'],
                                                "img_size": config['img_size'], 
                                                "seed": config['seed'],
                                                "transforms": str(config['transforms']['mean']) + " / " + str(config['transforms']['std']),
                                                "optimizer": config['solver']['OPT'],
                                                "weight_decay": config['solver']['WEIGHT_DECAY'],
                                                "momentum": config['solver']['MOMENTUM'],
                                                "base_lr": config['solver']['BASE_LR'],
                                                "lr_scheduler": config['solver']['LR_SCHEDULER'],
                                                "lr_decay_rate": config['solver']['LR_DECAY_RATE'],
                                                "lr_step_size": config['solver']['LR_STEP_SIZE'],
                                                "lr_step_milestones": config['solver']['LR_STEP_MILESTONES']
                                                })
    else:
        wandb_logger = None

    trainer_args = {
        'max_epochs': config['epochs'],
        'callbacks': callbacks,
        'default_root_dir': config['outdir'],
        'accelerator': accelerator,
        'devices': devices,
        'logger': wandb_logger,
        'deterministic': True,
        'profiler': 'simple',
        'num_sanity_val_steps': -1, # -1 to check all validation data, 0 to turn off
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

    # setup dataset
    data =  get_dataset(config)

    ##
    param_grid = config['param_grid']['use_grid']
    if param_grid == True:
        config['max_epochs'] = 20
        param_grid_triplet = {
            'embedding_sizes': config['param_grid']['embedding_sizes'],
            'margins': config['param_grid']['triplet']['margins'],
            'mining_types': config['param_grid']['triplet']['mining_types'],
            'learning_rates': config['param_grid']['learning_rates'],
            }
        param_grid_arcface = {
            'scales': config['param_grid']['arcface']['scales'],
            'margins': config['param_grid']['arcface']['margins'],
            }
        grid = ParameterGrid(param_grid_triplet)
    else:
        grid = range(1)

    results = []

    # setup model
    for params in grid:
        if param_grid and config['model_architecture']=='TripletModel':
            # triplet
            config['embedding_size'] = params['embedding_sizes']
            config['triplet_loss']['margin'] = params['margins']
            config['triplet_loss']['mining_type'] = params['mining_types']
            config['solver']['BASE_LR'] = params['learning_rates']
        elif param_grid and config['model_architecture']=='ArcFaceModel':
            # arcface
            config['embedding_size'] = params['embedding_sizes']
            config['arcface']['scale'] = params['scales']
            config['arcface']['margin'] = params['margins']
            config['solver']['BASE_LR'] = params['learning_rates']

        if config['checkpoint']:
            print(f"Loading model {config['model_architecture']} from checkpoint")
            if config['model_architecture']=='TripletModel':
                # pretrained=True
                model = TripletModel(config=config, pretrained=False)
            else:
                model = SimpleModel(config=config, pretrained=False)

            checkpoint = torch.load(config['checkpoint'])
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print(f"Start training {config['model_architecture']} from pretrained model")
            if config['model_architecture']=='TripletModel':
                model = TripletModel(config=config, pretrained=True)
            else:
                model = SimpleModel(config=config, pretrained=True)

        trainer = get_trainer(config)

        print('Args:')
        pprint(args.__dict__)
        print('configuration:')
        pprint(config)

        
        trainer.fit(model, data)
        
        if param_grid:
            results.append((params, trainer.callback_metrics['train_loss']))

    if param_grid:
        best_params = sorted(results, key=lambda x: x[1])[0]
        print(f'Best parameters: {best_params[0]}, Validation Loss: {best_params[1]}')