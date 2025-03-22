from wildlife_datasets import analysis, datasets, loader
from data.raptors_wildlife import Raptors, GoldensWildlife
from data.wildlife_dataset import WildlifeDataModule
import os
import sys
import wandb
import pandas as pd

def get_dataset(config, hardcode=None):
    if hardcode is not None:
        config['wildlife_name'] = hardcode['wildlife_name']
        # config['dataset']= hardcode['dataset']
        # config['cache_path']= hardcode['cache_path']

    # if os.path.exists(config['dataset']) is False:
    #         os.makedirs(config['dataset'])

    # single datasets
    if type(config['wildlife_name']) != list:
        if config['wildlife_name'] == 'raptors':
            if config['use_wandb'] == True:
                wandb.config.update({"animal_cat": "bird", 
                                    "dataset": '/proj/nobackup/aiforeagles/raptor_individuals_cropped/',
                                    "cache_path": '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_raptors.csv'
                                    }, allow_val_change=True)
            else:
                config['animal_cat'] = 'bird'
                config['dataset']= '/proj/nobackup/aiforeagles/raptor_individuals_cropped/'
                config['cache_path']= '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_raptors.csv'

            dataset = Raptors(root=config['dataset'], include_video=False)
            data = WildlifeDataModule(metadata=dataset.df, config = config)

        elif config['wildlife_name'] == 'goleag':
            if config['use_wandb'] == True:
                wandb.config.update({"animal_cat": "bird", 
                                    "dataset": '/proj/nobackup/aiforeagles/raptor_individuals_cropped/',
                                    "cache_path": '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/goleag_cache.csv'
                                    }, allow_val_change=True)
            else:
                config['animal_cat'] = 'bird'
                config['dataset']= '/proj/nobackup/aiforeagles/raptor_individuals_cropped/'
                config['cache_path']= '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/goleag_cache.csv'

            dataset = GoldensWildlife(root=config['dataset'], include_video=False)
            data = WildlifeDataModule(metadata=dataset.df, config = config)
        
        elif config['wildlife_name'] == 'whaleshark':
            if config['use_wandb'] == True:
                wandb.config.update({"animal_cat": "fish", 
                                    "dataset": '/proj/nobackup/aiforeagles/EDA-whaleshark/',
                                    "cache_path": '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_whaleshark.csv'
                                    }, allow_val_change=True)
            elif not config['dataset'].startswith('/Users'):
                config['animal_cat'] = 'fish'
                config['dataset']= '/proj/nobackup/aiforeagles/EDA-whaleshark/'
                config['cache_path']= '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_EDAwhaleshark.csv'

            dataset = datasets.WhaleSharkID(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config, only_cache=True)
        
        elif config['wildlife_name'] == 'ATRW':
            # datasets.ATRW.get_data(config['dataset'])
            if config['use_wandb'] == True:
                wandb.config.update({"animal_cat": "mammal", 
                                    "dataset": '/proj/nobackup/aiforeagles/ATRW/',
                                    "cache_path": '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_ATRW.csv'
                                    }, allow_val_change=True)
            elif not config['dataset'].startswith('/Users'):
                config['animal_cat'] = 'mammal'
                config['dataset']= '/proj/nobackup/aiforeagles/ATRW/'
                config['cache_path']= '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_ATRW.csv'

            dataset = datasets.ATRW(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
        
        elif config['wildlife_name'] == 'BirdIndividualID':
            if config['use_wandb'] == True:
                wandb.config.update({"animal_cat": "bird", 
                                    "dataset": '/proj/nobackup/aiforeagles/BirdIndividualID/',
                                    "cache_path": '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_BirdIndividualID.csv'
                                    }, allow_val_change=True)
            elif not config['dataset'].startswith('/Users'):
                config['animal_cat'] = 'bird'
                config['dataset']= '/proj/nobackup/aiforeagles/BirdIndividualID/'
                config['cache_path']= '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_BirdIndividualID.csv'
            dataset = datasets.BirdIndividualID(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            
        elif config['wildlife_name'] == 'GiraffeZebraID':
            # try to get zebras only because giraffes is really close up
            dataset = datasets.GiraffeZebraID(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'HyenaID2022':
            dataset = datasets.HyenaID2022(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'IPanda50':
            dataset = datasets.IPanda50(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'LeopardID2022':
            dataset = datasets.LeopardID2022(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'LionData':
            dataset = datasets.LionData(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'MPDD':
            dataset = datasets.MPDD(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'NyalaData':
            dataset = datasets.NyalaData(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'PolarBearVidID':
            dataset = datasets.PolarBearVidID(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'SealIDSegmented':
            dataset = datasets.SealIDSegmented(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
        elif config['wildlife_name'] == 'SeaTurtleID2022':
            # don't use bbox (it is only face)
            dataset = datasets.SeaTurtleID2022(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'reptile'
        elif config['wildlife_name'] == 'StripeSpotter': 
            # don't use bbox
            dataset = datasets.StripeSpotter(config['dataset'])
            data = WildlifeDataModule(metadata=dataset.df, config=config)
            config['animal_cat'] = 'mammal'
    else:  # combined datasets -- only works with cache
        if 'raptors' and 'BirdIndividualID' in config['wildlife_name'] and len(config['wildlife_name']) == 2:
            if config['use_wandb'] == True:
                wandb.config.update({"animal_cat": ['bird','bird'], 
                                    "dataset": '/proj/nobackup/aiforeagles/',
                                    "cache_path": '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_birds.csv',
                                    "cache_only": True, }, allow_val_change=True)
            else:
                config['animal_cat'] = ['bird', 'bird']
                config['dataset']= '/proj/nobackup/aiforeagles/'
                config['cache_path']= '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_birds.csv'
                config['cache_only']= True

                config['dataset']= '/Users/amee/Documents/code/master-thesis/datasets/'
                config['cache_path']= '/Users/amee/Documents/code/master-thesis/EagleID/dataset/dataframe/cache_birds.csv'

            raptor_path = os.path.join(config['dataset'], 'raptor_individuals_cropped')
            birds_path = os.path.join(config['dataset'], 'BirdIndividualID')

            dataset1 = Raptors(root=raptor_path, include_video=False)
            dataset1.df['wildlife_name'] = 'raptors'
            dataset1.df['path'] = dataset1.df['path'].apply(lambda x: os.path.join('raptor_individuals_cropped', x))
            dataset2 = datasets.BirdIndividualID(birds_path)
            dataset2.df['wildlife_name'] = 'BirdIndividualID'
            dataset2.df['path'] = dataset2.df['path'].apply(lambda x: os.path.join('BirdIndividualID', x))
            dataset_df = pd.concat([dataset1.df, dataset2.df], ignore_index=True)

            data = WildlifeDataModule(metadata=dataset_df, config = config)
        # elif 'raptors' and 'ATRW' in config['wildlife_name'] and len(config['wildlife_name']) == 2:
        #     if config['use_wandb'] == True:
        #         wandb.config.update({"animal_cat": ['bird','bird'], 
        #                             "dataset": '/proj/nobackup/aiforeagles/',
        #                             "cache_path": '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_birds.csv',
        #                             "cache_only": True, }, allow_val_change=True)
        #     else:
        #         config['animal_cat'] = ['bird', 'bird']
        #         config['dataset']= '/proj/nobackup/aiforeagles/'
        #         config['cache_path']= '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_birds.csv'
        #         config['cache_only']= True


        #     raptor_path = os.path.join(config['dataset'], 'raptor_individuals_cropped')
        #     birds_path = os.path.join(config['dataset'], 'ATRW')

        #     dataset1 = Raptors(root=raptor_path, include_video=False)
        #     dataset1.df['wildlife_name'] = 'raptors'
        #     dataset1.df['path'] = dataset1.df['path'].apply(lambda x: os.path.join('raptor_individuals_cropped', x))
        #     dataset2 = datasets.BirdIndividualID(birds_path)
        #     dataset2.df['wildlife_name'] = 'ATRW'
        #     dataset2.df['path'] = dataset2.df['path'].apply(lambda x: os.path.join('ATRW', x))
        #     dataset_df = pd.concat([dataset1.df, dataset2.df], ignore_index=True)

        #     data = WildlifeDataModule(metadata=dataset_df, config = config)
        elif 'raptors' and 'BirdIndividualID' and 'ATRW' and 'whaleshark' in config['wildlife_name'] and len(config['wildlife_name']) == 4:
            if config['use_wandb'] == True:
                wandb.config.update({"animal_cat": ['bird','bird', 'mammal', 'fish'], 
                                    "dataset": '/proj/nobackup/aiforeagles/',
                                    "cache_path": '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_multispecies.csv',
                                    "cache_only": True, }, allow_val_change=True)
            else:
                config['animal_cat'] = ['bird', 'bird', 'mammal', 'fish']
                config['dataset']= '/proj/nobackup/aiforeagles/'
                config['cache_path']= '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_multispecies.csv'
                config['cache_only']= True


            raptor_path = os.path.join(config['dataset'], 'raptor_individuals_cropped')
            birds_path = os.path.join(config['dataset'], 'BirdIndividualID')
            atrw_path = os.path.join(config['dataset'], 'ATRW')
            whaleshark_path = os.path.join(config['dataset'], 'EDA-whaleshark')

            dataset1 = Raptors(root=raptor_path, include_video=False)
            dataset1.df['wildlife_name'] = 'raptors'
            dataset1.df['path'] = dataset1.df['path'].apply(lambda x: os.path.join('raptor_individuals_cropped', x))
            dataset2 = datasets.BirdIndividualID(birds_path)
            dataset2.df['wildlife_name'] = 'BirdIndividualID'
            dataset2.df['path'] = dataset2.df['path'].apply(lambda x: os.path.join('BirdIndividualID', x))
            dataset3 = datasets.WhaleSharkID(whaleshark_path)
            dataset3.df['wildlife_name'] = 'whaleshark'
            dataset3.df['path'] = dataset3.df['path'].apply(lambda x: os.path.join('EDA-whaleshark', x))
            dataset4 = datasets.ATRW(atrw_path)
            dataset4.df['wildlife_name'] = 'ATRW'
            dataset4.df['path'] = dataset4.df['path'].apply(lambda x: os.path.join('ATRW', x))

            dataset_df = pd.concat([dataset1.df, dataset2.df, dataset3.df, dataset4.df], ignore_index=True)

            data = WildlifeDataModule(metadata=dataset_df, config = config)
        else:
            raise ValueError(f"Unknown dataset {config['wildlife_name']}")
    
    return data