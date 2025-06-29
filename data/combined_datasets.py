from wildlife_datasets import analysis, datasets, loader
from data.raptors_wildlife import Raptors, GoldensWildlife
from data.wildlife_dataset import WildlifeDataModule
from data.artportalen_goleag import ArtportalenDataModule
import os
import sys
import wandb
import pandas as pd

def get_dataset(config, hardcode=None, sweep=False):
    if hardcode is not None:
        config['wildlife_name'] = hardcode['wildlife_name']

    raw = config['wildlife_name']
    if isinstance(raw, str):
        names = {n.strip() for n in raw.split(',')}
    else:
        names = set(raw)

    def _update_wandb(cat, ds, cache):
        if sweep and config.get('use_wandb'):
            wandb.config.update({
                "animal_cat": cat,
                "dataset":    ds,
                "cache_path": cache
            }, allow_val_change=True)
        config['animal_cat'] = cat
        config['dataset']    = ds
        config['cache_path'] = cache

    # — SINGLE‑DATASET CASES
    if names == {'raptors'}:
        _update_wandb('bird',
                      '/proj/nobackup/aiforeagles/raptor_individuals_cropped/',
                      '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_raptors_split.csv')
        ds = Raptors(root=config['dataset'], include_video=False)
        return WildlifeDataModule(metadata=ds.df, config=config)
    
    if names == {'goleag'}:
        _update_wandb('bird',
                      '/proj/nobackup/aiforeagles/raptor_individuals_cropped/',
                      '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_goleag_split.csv')
        ds = GoldensWildlife(root=config['dataset'], include_video=False)
        return WildlifeDataModule(metadata=ds.df, config=config)
    
    if names == {'artportalen'}:
        _update_wandb('bird',
                      '/proj/nobackup/aiforeagles/artportalen/artportalen_goeag/',
                      '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_artportalen.csv')
        
        # Create ArtportalenDataModule with the same parameters as in the notebook
        data_module = ArtportalenDataModule(
            data_dir=config['dataset'],
            preprocess_lvl=config['preprocess_lvl'],
            batch_size=config['batch_size'],
            size=config['img_size'],
            mean=config['transforms']['mean'],
            std=config['transforms']['std'],
            cache_dir=config['cache_dir']
        )
        
        # Use the existing CSV files
        train_csv = '/proj/nobackup/aiforeagles/artportalen/final_train_sep_sightings.csv'
        val_csv = '/proj/nobackup/aiforeagles/artportalen/train_sep_sightings.csv'
        
        data_module.setup_from_csv(train_csv, val_csv)
        return data_module
    
    if names == {'BirdIndividualID'}:
        _update_wandb('bird',
                      '/proj/nobackup/aiforeagles/BirdIndividualID/',
                      '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_BirdIndividualID.csv')
        ds = datasets.BirdIndividualID(config['dataset'])
        return WildlifeDataModule(metadata=ds.df, config=config)
        
    if names == {'whaleshark'}:
        _update_wandb('fish',
                      '/proj/nobackup/aiforeagles/EDA-whaleshark/',
                      '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_whaleshark.csv')
        ds = datasets.WhaleSharkID(config['dataset'])
        return WildlifeDataModule(metadata=ds.df, config=config, only_cache=True)

    if names == {'ATRW'}:
        _update_wandb('mammal',
                      '/proj/nobackup/aiforeagles/ATRW/',
                      '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/cache_ATRW.csv')
        ds = datasets.ATRW(config['dataset'])
        return WildlifeDataModule(metadata=ds.df, config=config, only_cache=True)

    if names == {'seal'}:
        ds = datasets.SealID(config['dataset'])
        return WildlifeDataModule(metadata=ds.df, config=config)
    
    if names == {'NDD20'}:
        ds = datasets.NDD20(config['dataset'])
        return WildlifeDataModule(metadata=ds.df, config=config)

    if names == {'NyalaData'}:
        ds = datasets.NyalaData(config['dataset'])
        return WildlifeDataModule(metadata=ds.df, config=config)
    
    if names == {'ELPephant'}:
        ds = datasets.ELPephants(config['dataset'])
        return WildlifeDataModule(metadata=ds.df, config=config)

    # — MULTI‑DATASET (cache‑only) CASES
    # note: we assume `all(config['only_cache']) == True` if use cache‑only
    if len(names)>1 and all(config.get('only_cache', [])):
        df = pd.read_csv(config['cache_path'])
        return WildlifeDataModule(metadata=df, config=config)
    
    raise ValueError(f"Invalid wildlife_name: {config['wildlife_name']} / {names}")