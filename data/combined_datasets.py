from wildlife_datasets import analysis, datasets, loader
from data.raptors_wildlife import Raptors, GoldensWildlife
from data.wildlife_dataset import WildlifeDataModule
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
                      '/proj/nobackup/aiforeagles/EagleID/dataset/dataframe/goleag_cache.csv')
        ds = GoldensWildlife(config['dataset'], include_video=False)
        return WildlifeDataModule(metadata=ds.df, config=config)
    
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
    
    if names == {'elephant'}:
        ds = datasets.ELPephants(config['dataset'])
        return WildlifeDataModule(metadata=ds.df, config=config)

    # — MULTI‑DATASET (cache‑only) CASES
    # note: we assume `all(config['only_cache']) == True` if use cache‑only
    if len(names)>1 and all(config.get('only_cache', [])):
        df = pd.read_csv(config['cache_path'])
        return WildlifeDataModule(metadata=df, config=config)
    
    raise ValueError(f"Invalid wildlife_name: {config['wildlife_name']} / {names}")