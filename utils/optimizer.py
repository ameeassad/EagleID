import torch

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
    else:
        raise NotImplementedError

    return lr_scheduler_config
