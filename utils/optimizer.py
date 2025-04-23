import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def get_optimizer(config, parameters) -> torch.optim.Optimizer:
    if config['solver']['OPT'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=config['solver']['BASE_LR'], weight_decay=config['solver']['WEIGHT_DECAY'])
    elif config['solver']['OPT'] == 'sgd':
        optimizer = torch.optim.SGD(
            parameters, lr=config['solver']['BASE_LR'], weight_decay=config['solver']['WEIGHT_DECAY'], momentum=config['solver']['MOMENTUM']
        )
    else:
        raise NotImplementedError()

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
    if config['solver']['LR_SCHEDULER'] == 'cosine_annealing':
        total_epochs = config.get('epochs', 60)
        ramp_up_epochs = config['solver'].get('lr_ramp_ep', 8)
        lr_max = config['solver'].get('lr_max', 1e-4)
        lr_start = config['solver'].get('lr_start', 3e-6)
        lr_min = config['solver'].get('lr_min', 1e-6)

        # Set optimizer LR to lr_start
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_start

        # Linear warmup: lr_start → lr_max over ramp_up_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=lr_start / lr_max,
            end_factor=1.0,
            total_iters=ramp_up_epochs
        )

        # Cosine annealing from lr_max → lr_min over remaining epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - ramp_up_epochs,
            eta_min=lr_min
        )

        # Chain them
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[ramp_up_epochs]
        )

        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    else:
        raise NotImplementedError

    return lr_scheduler_config
