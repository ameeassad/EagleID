from wildlife_datasets import analysis, datasets, loader
from data.raptors_wildlife import Raptors, WildlifeReidDataModule

import os
import sys

def get_dataset(config):
    if config['wildlife_name'] == 'raptors':
        dataset = Raptors(root=config['dataset'], include_video=False)
        data = WildlifeReidDataModule(metadata=dataset.df, config = config)
    elif config['wildlife_name'] == 'whaleshark':
        dataset = datasets.WhaleSharkID(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config, only_cache=True)
    elif config['wildlife_name'] == 'ATRW':
        if os.path.exists(config['dataset']) is False:
            os.makedirs(config['dataset'])
            datasets.ATRW.get_data(config['dataset'])
        dataset = datasets.ATRW(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'BirdIndividualID':
        dataset = datasets.BirdIndividualID(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'GiraffeZebraID':
        # try to get zebras only because giraffes is really close up
        dataset = datasets.GiraffeZebraID(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'HyenaID2022':
        dataset = datasets.HyenaID2022(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'IPanda50':
        dataset = datasets.IPanda50(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'LeopardID2022':
        dataset = datasets.LeopardID2022(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'LionData':
        dataset = datasets.LionData(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'MPDD':
        dataset = datasets.MPDD(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'NyalaData':
        dataset = datasets.NyalaData(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'PolarBearVidID':
        dataset = datasets.PolarBearVidID(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'SealIDSegmented':
        dataset = datasets.SealIDSegmented(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'SeaTurtleID2022':
        # don't use bbox (it is only face)
        dataset = datasets.SeaTurtleID2022(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    elif config['wildlife_name'] == 'StripeSpotter': 
        # don't use bbox
        dataset = datasets.StripeSpotter(config['dataset'])
        data = WildlifeReidDataModule(metadata=dataset.df, config=config)
    else:
        raise ValueError(f"Unknown dataset {config['wildlife_name']}")
    
    return data