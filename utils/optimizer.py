import torch
import math
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def get_optimizer(config, parameters) -> torch.optim.Optimizer:
    if config['solver']['OPT'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=config['solver']['BASE_LR'], weight_decay=config['solver']['WEIGHT_DECAY'])
    elif config['solver']['OPT'] == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=config['solver']['BASE_LR'], weight_decay=config['solver']['WEIGHT_DECAY'])
    elif config['solver']['OPT'] == 'sgd':
        optimizer = torch.optim.SGD(
            parameters, lr=config['solver']['BASE_LR'], weight_decay=config['solver']['WEIGHT_DECAY'], momentum=config['solver']['MOMENTUM']
        )
    else:
        raise NotImplementedError(f"Optimizer {config['solver']['OPT']} not implemented. Supported optimizers: adam, adamw, sgd")

    return optimizer


def get_lr_scheduler_config(config, optimizer: torch.optim.Optimizer) -> dict:
    if config['solver']['LR_SCHEDULER'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config['solver']['LR_STEP_SIZE'], gamma=config['solver']['LR_DECAY_RATE']
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif config['solver']['LR_SCHEDULER'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config['solver']['LR_STEP_MILESTONES'], gamma=config['solver']['LR_DECAY_RATE']
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif config['solver']['LR_SCHEDULER'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'train/loss',
            'interval': 'epoch',
            'frequency': 1,
        }
    elif config['solver']['LR_SCHEDULER'] == 'cosine_annealing':
        # Number of epochs for cosine decay (after ramp-up)
        total_epochs = config.get('epochs', 20)  # Default to 20 if not specified
        ramp_up_epochs = config['solver'].get('lr_ramp_ep', 0)
        T_max = total_epochs - ramp_up_epochs  # Decay over remaining epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config['solver'].get('lr_min', 1e-06)  # Minimum LR
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif config['solver']['LR_SCHEDULER'] == 'cosine_with_warmup':
        # Get warmup epochs from config
        warmup_epochs = config['solver'].get('WARMUP_EPOCHS', 5)
        total_epochs = config.get('epochs', 100)
        min_lr = config['solver'].get('MIN_LR', 1e-6)
        
        # Create warmup scheduler (linear warmup from 0 to BASE_LR)
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.0, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        # Create cosine annealing scheduler (from BASE_LR to min_lr)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
        # Combine warmup and cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    else:
        raise NotImplementedError(f"LR scheduler {config['solver']['LR_SCHEDULER']} not implemented. Supported schedulers: step, multistep, reduce_on_plateau, cosine_annealing, cosine_with_warmup")

    return lr_scheduler_config
