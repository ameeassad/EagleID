# https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py

import argparse
from pprint import pprint
import yaml, shutil, os
from init_paths import lib_path

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from dataset import ArtportalenDataModule
from models import SimpleModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inference with GradCAM visualisation.')
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
        save_top_k=-1,
        every_n_epochs=checkpoint_interval,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=config['early_stopping']['monitor'],  # Monitored metric
        min_delta=config['early_stopping']['min_delta'],      # Minimum change to qualify as an improvement
        patience=config['early_stopping']['patience'],         # Number of epochs with no improvement after which training will be stopped
        verbose=config['early_stopping']['verbose'],
        mode=config['early_stopping']['mode']           # Mode for the monitored metric ('min' or 'max')
    )
    return [ckpt_callback, lr_callback, early_stop_callback]



def get_trainer(config) -> Trainer:
    accelerator, devices, strategy = "cpu", 1, None
    trainer_args = {
        'max_epochs': config['epochs'],
        'callbacks': None,
        'default_root_dir': config['outdir'],
        'accelerator': accelerator,
        'devices': devices,
        'logger': None,
        'deterministic': True,
    }

    if strategy is not None:
        trainer_args['strategy'] = strategy

    trainer = Trainer(**trainer_args)
    return trainer


if __name__ == '__main__':
    args = get_args()
    
    config_file_path = yaml.safe_load(os.path.join(lib_path, 'configs',args.config))
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    seed_everything(config['seed'], workers=True)

    data = ArtportalenDataModule(data_dir=config['dataset'], batch_size=config['batch_size'], size=config['img_size'], test=True)
    data.prepare_testing_data(config['dataset'])
    dataloader = data.test_dataloader()

    model = SimpleModel(model_name=config['model_name'], pretrained=False, num_classes=data.num_classes, outdir=config['outdir'])
    if config['n_gpu']>0:
        checkpoint = torch.load(config['checkpoint'])
    else:
        checkpoint = torch.load(config['checkpoint'], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    

    trainer = get_trainer(config)

    print('Args:')
    pprint(args.__dict__)
    print('configuration:')
    pprint(config)

    # 
    trainer.fit(model, data)
