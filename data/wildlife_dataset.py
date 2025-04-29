import json
import os
import math
import ast
import pandas as pd
import numpy as np
from scipy import io
from pathlib import Path
import shutil

from wildlife_datasets import datasets, splits
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from typing import Callable

import cv2
import pycocotools.mask as mask_coco
from PIL import Image

from wildlife_tools.tools import realize
from wildlife_tools.data.dataset import WildlifeDataset

import data. transforms as t
from data.transforms import RGBTransforms, ResizeAndPadRGB, ValTransforms, SynchTransforms, resize_and_pad, rotate_image
from data.data_utils import CustomClosedSetSplit, StratifiedSpeciesSplit, SplitQueryDatabase, analyze_split, RandomIdentitySampler, PaddedBatchSampler
from data.raptors_wildlife import RaptorsWildlife

from preprocess.preprocess_utils import create_mask, create_skeleton_channel, create_multichannel_heatmaps
from preprocess.component_gen import component_generation_module
from preprocess.mmpose_fill import get_keypoints_info, get_skeleton_info


class Wildlife(WildlifeDataset):
    def __init__(
        self,
        metadata: pd.DataFrame | None = None,
        root: str | None = None,
        split: Callable | None = None,
        transform: callable = None,
        img_load: str = "full",
        col_path: str = "path",
        col_label: str = "identity", 
        load_label: bool = True,
        load_metadata: bool = False,
    ):    
        metadata = metadata.sort_values(by=['identity', 'image_id']).reset_index(drop=True)  # Sort for consistency

        super().__init__(
            metadata=metadata,
            root=root,
            split = split,
            transform=transform,
            img_load=img_load,
            col_path=col_path,
            col_label=col_label,
            load_label=load_label
        )
        # New Child Variables
        self.load_metadata = load_metadata

    def get_image(self, path):
        """
        Custom image loader, customized based on preprocessing level for raptor images.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def __getitem__(self, idx):
        data = self.metadata.iloc[idx]
        if self.root:
            img_path = os.path.join(self.root, data[self.col_path])
        else:
            img_path = data[self.col_path]
        img = self.get_image(img_path)

        # SEGMENTATIONS required: segmentation data is wrt. original uncropped image
        if self.img_load in ["full_mask", "full_hide", "bbox_mask", "bbox_hide", "bbox_mask_skeleton", "bbox_mask_components", "bbox_mask_heatmaps"]:
            if not ("segmentation" in data):
                raise ValueError(f"{self.img_load} selected but no segmentation found.")
            
            if type(data["segmentation"]) == str:
                segmentation = eval(data["segmentation"])
            else:
                segmentation = data["segmentation"]
            seg_coco = segmentation

            if 'height' in data:
                height = data['height']
                width = data['width']
            else:
                height = img.size[1]
                width = img.size[0]

            if isinstance(segmentation, list):
                # Convert polygon to RLE
                height, width = int(data['height']), int(data['width'])
                # height, width = img.size[1], img.size[0]
                rle = mask_coco.frPyObjects(segmentation, height, width)
                segmentation = mask_coco.merge(rle)

        if self.img_load in ["bbox", "bbox_mask", "bbox_hide", "bbox_mask_skeleton", "bbox_mask_components", "bbox_mask_heatmaps"]:
            if not ("bbox" in data):
                raise ValueError(f"{self.img_load} selected but no bbox found.")
            if type(data["bbox"]) == str:
                bbox = json.loads(data["bbox"])
            else:
                bbox = data["bbox"]

        # Load full image as it is.
        if self.img_load == "full":
            img = img

        # Mask background using segmentation mask.
        elif self.img_load == "full_mask":
            mask = mask_coco.decode(segmentation).astype("bool")
            img = Image.fromarray(img * mask[..., np.newaxis])

        # Hide object using segmentation mask
        elif self.img_load == "full_hide":
            mask = mask_coco.decode(segmentation).astype("bool")
            img = Image.fromarray(img * ~mask[..., np.newaxis])

        # Crop to bounding box
        elif self.img_load == "bbox":
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Mask background using segmentation mask and crop to bounding box.
        elif self.img_load in ["bbox_mask", "bbox_mask_skeleton", "bbox_mask_components", "bbox_mask_heatmaps"]:
            mask = mask_coco.decode(segmentation).astype("bool")
            img = Image.fromarray(img * mask[..., np.newaxis])
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Hide object using segmentation mask and crop to bounding box.
        elif self.img_load == "bbox_hide":
            mask = mask_coco.decode(segmentation).astype("bool")
            img = Image.fromarray(img * ~mask[..., np.newaxis])
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Crop black background around images
        elif self.img_load == "crop_black":
            y_nonzero, x_nonzero, _ = np.nonzero(img)
            img = img.crop(
                (
                    np.min(x_nonzero),
                    np.min(y_nonzero),
                    np.max(x_nonzero),
                    np.max(y_nonzero),
                )
            )

        else:
            raise ValueError(f"Invalid img_load argument: {self.img_load}")
        
        if self.img_load in ["bbox_mask_skeleton", "bbox_mask_components", "bbox_mask_heatmaps"]:
            if not ("keypoints" in data):
                raise ValueError(f"{self.img_load} selected but no keypoints found.")
            if type(data["keypoints"]) == str:
                keypoints = eval(data["keypoints"])
            else:
                keypoints = data["keypoints"]
        
        if self.img_load == "bbox_mask_skeleton":
            x_min = math.floor(bbox[0])
            y_min = math.floor(bbox[1])
            w = math.ceil(bbox[2])
            h = math.ceil(bbox[3])
            bbox = [x_min, y_min, w, h]
            # Create skeleton channel:
            connections = get_skeleton_info()
            # Convert connections from string to list if necessary
            if isinstance(connections, str):
                connections = ast.literal_eval(connections)
            skeleton_channel = create_skeleton_channel(keypoints, connections, height=int(height), width=int(width))
            skeleton_channel = skeleton_channel[y_min:y_min + h, x_min:x_min + w]
            # print(skeleton_channel.shape)
            # print(skeleton_channel)
        elif self.img_load == "bbox_mask_components":
            img_cv = cv2.imread(img_path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            keypoint_labels = get_keypoints_info()
            cropped_images = component_generation_module(img_cv, bbox, keypoints, keypoint_labels, True, seg_coco)
        elif self.img_load == "bbox_mask_heatmaps":
            x_min = math.floor(bbox[0])
            y_min = math.floor(bbox[1])
            w = math.ceil(bbox[2])
            h = math.ceil(bbox[3])
            bbox = [x_min, y_min, w, h]
            # Create keypoint heatmaps
            keypoint_heatmaps = create_multichannel_heatmaps(keypoints, int(height), int(width), w, h, 25)
            # Crop each heatmap to the bounding box
            cropped_heatmaps = [heatmap[y_min:y_min + h, x_min:x_min + w] for heatmap in keypoint_heatmaps]

        if self.transform: # handles augmentations and concatenations (if relevant)
            if self.img_load == "bbox_mask_skeleton":
                img, skeleton_channel = self.transform[0](img, skeleton_channel)
                img = self.transform[1](img, skeleton_channel) #concatenated 
            elif self.img_load == "bbox_mask_heatmaps":
                img, heatmap_channels = self.transform[0](img, cropped_heatmaps)
                img = self.transform[1](img, heatmap_channels) #concatenated 
            elif self.img_load == "bbox_mask_components":
                img = self.transform[0](img)
                img = self.transform[1](img, cropped_images) #concatenated 

            else:
                # img = resize_and_pad(img, self.size)
                for i in range(len(self.transform)):
                    img = self.transform[i](img)

        # if self.split:
        #     return img, self.labels[idx], bool(data['query'])
        if self.load_metadata:
            return {
                'img': img,
                'label': self.labels[idx],
                'path': data['path'],
                'identity' : data['identity'],
            }
        elif self.load_label: # default is True
            return img, self.labels[idx]
        else:
            return img

    def get_df(self) -> pd.DataFrame:
        return self.metadata

class WildlifeDataModule(pl.LightningDataModule):
    def __init__(self, 
                 metadata, 
                 config = None, 
                 data_dir="", 
                 preprocess_lvl=0, 
                 batch_size=8, 
                 size=224, 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225], 
                 num_workers=2, 
                 cache_path="../dataset/dataframe/cache.csv", 
                 animal_cat='bird', 
                 splitter ='closed', 
                 only_cache=False,
                 wildlife_names=None,
                 classic_transform=False,
                 precompute=False,
                ):
        # if using config, does not consider rest of parameters
        super().__init__()
        self.config = config
        self.metadata = metadata
        if config:
            self.data_dir = config['dataset']
            self.preprocess_lvl = int(config["preprocess_lvl"])
            self.batch_size = int(config['batch_size'])
            self.size = int(config['img_size'])
            self.mean = (config['transforms']['mean'], config['transforms']['mean'], config['transforms']['mean']) if isinstance(config['transforms']['mean'], float) else tuple(config['transforms']['mean'])
            self.std = (config['transforms']['std'], config['transforms']['std'], config['transforms']['std']) if isinstance(config['transforms']['std'], float) else tuple(config['transforms']['std'])
            self.split_ratio = config['split_ratio'] # percent of individuals used for training
            self.num_workers = config['num_workers']
            self.cache_path = config['cache_path']
            self.animal_cat = config['animal_cat']
            self.splitter = config['splitter']
            self.only_cache = config['only_cache']
            self.wildlife_names = config['wildlife_name']
            self.classic_transform = config.get("custom_transform", False)
            self.precompute = config.get("precompute", False)
        else:
            self.data_dir = data_dir
            self.num_workers = num_workers
            self.preprocess_lvl = preprocess_lvl
            self.batch_size = batch_size
            self.size = size
            self.mean = (mean, mean, mean) if isinstance(mean, float) else tuple(mean)
            self.std = (std, std, std) if isinstance(std, float) else tuple(std)
            self.split_ratio = 0.8
            self.cache_path = cache_path
            self.animal_cat = animal_cat
            self.splitter = splitter
            self.only_cache = only_cache
            self.wildlife_names = wildlife_names
            self.classic_transform = classic_transform
            self.precompute = precompute
            
        self.load_metadata = True

        if len(self.wildlife_names.split(',')) > 1:
            multispecies = True
            self.wildlife_names = self.wildlife_names.replace(',', '_').replace(' ', '_')

        # Handle transforms
        if self.preprocess_lvl == 3:
            resize_and_pad = t.ResizeAndPadBoth(self.size, skeleton=True)
            sync_transform = t.SynchTransforms(mean=self.mean, std=self.std, degrees=45, color_and_gaussian=True)
            sync_val_transform = t.SynchTransforms(mean=self.mean, std=self.std, degrees=0, color_and_gaussian=False)
            self.train_transforms =  [resize_and_pad, sync_transform]
            self.val_transforms = [resize_and_pad, sync_val_transform]  # everything except for color / gaussian transforms aka no someOf transforms
        elif self.preprocess_lvl == 4:
            resize_and_pad = t.ResizeAndPadRGB(self.size)
            sync_transform = t.ComponentGenerationTransforms(mean=self.mean, std=self.std, degrees=45, color_and_gaussian=True)
            sync_val_transform = t.ComponentGenerationTransforms(mean=self.mean, std=self.std, degrees=0, color_and_gaussian=False)
            self.train_transforms =  [resize_and_pad, sync_transform]
            self.val_transforms = [resize_and_pad, sync_val_transform]  # everything except for color / gaussian transforms
        elif self.preprocess_lvl == 5:
            resize_and_pad = t.ResizeAndPadBoth(self.size, skeleton=False)
            sync_transform = t.SynchMultiChannelTransforms(mean=self.mean, std=self.std, degrees=45, color_and_gaussian=True)
            sync_val_transform = t.SynchMultiChannelTransforms(mean=self.mean, std=self.std, degrees=0, color_and_gaussian=False)
            self.train_transforms =  [resize_and_pad, sync_transform]
            self.val_transforms = [resize_and_pad, sync_val_transform]
        else:
            if self.classic_transform:
                transforms_list = [transforms.Resize([224, 224]), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
                self.train_transforms = transforms_list
                self.val_transforms = transforms_list
            else:
                resize_and_pad = t.ResizeAndPadRGB(self.size)
                rgb_transform = t.RGBTransforms(mean=self.mean, std=self.std, degrees=15, color_and_gaussian=True)
                rgb_val_tranform = t.RGBTransforms(mean=self.mean, std=self.std, degrees=0, color_and_gaussian=False)
                self.train_transforms =  [resize_and_pad, rgb_transform]
                self.val_transforms = [resize_and_pad, rgb_val_tranform]

        # Add date if present in metadata
        if 'date' in self.metadata.columns and self.metadata['date'].isna().any():
            self.metadata.loc[pd.isna(self.metadata['date']), 'date'] = "unknown" 

        # If only dealing with cache (no real time preprocessing), read the cache and clean it up (segs and keypoints possibly)
        # Else, just copy the metadata and work with that
        if isinstance(self.only_cache, bool):
            self.only_cache = [self.only_cache, self.only_cache]
        if self.only_cache[0]:
            cache_df = pd.read_csv(self.cache_path)
            print(f"Dataset size before pre-processing and cleaning: {len(cache_df)}")
            df_all = cache_df.copy()
            df_all = self.clean_segmentation(df_all)
            if self.only_cache[1] and self.preprocess_lvl>2:
                df_all = df_all[df_all['keypoints'].apply(lambda x: not isinstance(x, float))]
        else:
            df_all = metadata.copy()
            print(f"Dataset size before pre-processing and cleaning: {len(df_all)}")

        # Preprocessing
        if self.preprocess_lvl > 0 and not self.only_cache[0]: # 1: bounding box cropped image or 2: masked image
            from preprocess.segmenter import add_segmentations

            df_all = add_segmentations(metadata, self.data_dir, cache_path=self.cache_path, only_cache=self.only_cache[0])
            df_all = self.clean_segmentation(df_all)

        
        if self.preprocess_lvl >= 3 and not self.only_cache[1]: # 3: masked + pose (skeleton) image in 1 channel or 4: masked + body part clusters in channels
            from preprocess.mmpose_fill import fill_keypoints, get_skeleton_info, get_keypoints_info

            self.keypoints = get_keypoints_info()
            self.skeleton = get_skeleton_info()

            df_all = fill_keypoints(df_all, self.data_dir, cache_path=self.cache_path, animal_cat=self.animal_cat, only_cache=self.only_cache[1])
            # remove 'float' keypoints because NaN is treated as float
            df_all = df_all[df_all['keypoints'].apply(lambda x: not isinstance(x, float))]

        # Split the dataset according to closed or open or default via metadata values
        if self.splitter == 'original_split':
            df_train = df_all[df_all['original_split'] == 'train']
            df_test = df_all[df_all['original_split'] == 'test']
            analyze_split(df_all, df_train.index, df_test.index)
        elif self.splitter == 'metadata_split':
            df_train = df_all[df_all['metadata_split'] == 'train']
            df_test = df_all[df_all['metadata_split'] == 'test']
            analyze_split(df_all, df_train.index, df_test.index)
        else: # Will create new metadata split values and save to file _split.csv!
            if self.splitter == 'closed':
                # Closed-set split x(same individuals in train/test)
                splitter = splits.ClosedSetSplit(self.split_ratio)
            elif self.splitter == 'open':
                # Open-set split (some individuals only in test)
                splitter = splits.OpenSetSplit(self.split_ratio, 0.1)
            elif self.splitter == 'closed_species_stratified':
                splitter = StratifiedSpeciesSplit(self.split_ratio)
            elif self.splitter == 'custom_closed':
                splitter = CustomClosedSetSplit(self.split_ratio)
            else:
                print(f"Unknown splitter: {self.splitter}. Using closed-set split.")
                splitter = splits.ClosedSetSplit(self.split_ratio)
            for idx_train, idx_test in splitter.split(df_all):
                analyze_split(df_all, idx_train, idx_test)
            df_train, df_test = df_all.loc[idx_train], df_all.loc[idx_test]

            assert all(idx in df_all.index for idx in idx_test), "Test indices do not match df_all indices"

            # Also handle query / gallery in dataframe (only if metadata_split NOT selected)
            # if 'query' not in df_all.columns: # want to overwrite even if exists....
            df_all['query'] = 0 # Reset
            # Split query set and gallery set for evaluation via SplitQueryDatabase
            self.val_split = SplitQueryDatabase()
            df_test = self.val_split(df_test)
            df_test['query'] = df_test['query'].astype(int) # Ensure integer before boolean
            # df_all['query'] = df_test['query']  # save it to original
            df_all.loc[df_test.index, 'query'] = df_test['query'] # save it to original; must ensure that df_test.index matches exactly with df_all.index!

            # Save train/test split back to cache (only if metadata_split NOT selected)
            df_all['metadata_split'] = ''
            df_all.loc[idx_train, 'metadata_split'] = 'train'
            df_all.loc[idx_test, 'metadata_split'] = 'test'
            original_path = Path(self.cache_path)
            df_all.to_csv(original_path.with_name(original_path.stem + '_split.csv'), index=False)

        # Happens for all split options, assuming 'query' column is present
        if df_all['query'].isna().any():
            print("Warning: 'query' column contains NaN values. Filling with False.")
            df_all['query'] = df_all['query'].fillna(False)
        df_test['query'] = df_test['query'].astype(bool)
        df_query = df_test[df_test['query']]
        df_gallery = df_test[~df_test['query']]

        # Reset indices
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        if config:
            config['arcface_loss']['n_classes'] = len(df_train['identity'].unique())

        print("Training Set")
        print(f"Length: {len(df_train)}")
        print(f"Number of individuals (classes): {len(df_train['identity'].unique())}")
        print(f"Mean images/individual: {df_train['identity'].value_counts().mean()}")
        print(f"Min images/individual: {df_train['identity'].value_counts().min()}")
        print(f"Max images/individual: {df_train['identity'].value_counts().max()}")

        print("Test Set")
        print(f"Length: {len(df_test)}")
        print(f"Number of individuals (classes): {len(df_test['identity'].unique())}")
        print(f"Mean images per individual: {df_test['identity'].value_counts().mean()}")
        print(f"Min images per individual: {df_test['identity'].value_counts().min()}")
        print(f"Max images per individual: {df_test['identity'].value_counts().max()}")

        # Handle for Wildlife class parameter
        if self.preprocess_lvl == 0:
            self.method = 'full'
        elif self.preprocess_lvl == 1:
            self.method = 'bbox'
        elif self.preprocess_lvl == 2: 
            self.method = "bbox_mask"
        elif self.preprocess_lvl == 3: 
            self.method = "bbox_mask_skeleton"
        elif self.preprocess_lvl == 4:
            self.method = "bbox_mask_components"
        elif self.preprocess_lvl == 5:
            self.method = "bbox_mask_heatmaps"

        if self.precompute:
            self.train_dataset = PrecomputedWildlife(metadata=df_train, root=self.data_dir, transform=self.train_transforms, col_label = 'identity', img_load=self.method, category="train_"+self.wildlife_names, load_metadata=self.load_metadata)
            self.val_query_dataset = PrecomputedWildlife(metadata=df_query, root=self.data_dir, transform=self.val_transforms, col_label = 'identity', img_load=self.method, category="query_"+self.wildlife_names, load_metadata=self.load_metadata)
            self.val_gallery_dataset = PrecomputedWildlife(metadata=df_gallery, root=self.data_dir, transform=self.val_transforms, col_label = 'identity', img_load=self.method, category="gallery_"+self.wildlife_names, load_metadata=self.load_metadata)
        else:
            self.train_dataset = Wildlife(metadata=df_train, root=self.data_dir, transform=self.train_transforms, col_label = 'identity', img_load=self.method, load_metadata=self.load_metadata)
            self.val_query_dataset = Wildlife(metadata=df_query, root=self.data_dir, transform=self.val_transforms, col_label = 'identity', img_load=self.method, load_metadata=self.load_metadata)
            self.val_gallery_dataset = Wildlife(metadata=df_gallery, root=self.data_dir, transform=self.val_transforms, col_label = 'identity', img_load=self.method, load_metadata=self.load_metadata)


        print("Round 1 Query image_ids:", df_query['image_id'].tolist())
        print("Round 1 Gallery image_ids:", df_gallery['image_id'].tolist())

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return DataLoader(self.train_dataset, batch_sampler=PaddedBatchSampler(data_source=self.train_dataset,batch_size=self.batch_size,shuffle=True),num_workers=self.num_workers)
        # If using Triplet loss need an anchor and a positive in same identity class: // handled by triplet loss function
        # sampler = RandomIdentitySampler(dataset=self.train_dataset, batch_size=self.batch_size)
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
    
    def val_dataloader(self):
        print("Query dataset length:", len(self.val_query_dataset))
        print("Gallery dataset length:", len(self.val_gallery_dataset))
        print("Query labels (first 5):", self.val_query_dataset.labels[:5])
        print("Gallery labels (first 5):", self.val_gallery_dataset.labels[:5])

        query_loader = DataLoader(self.val_query_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        gallery_loader = DataLoader(self.val_gallery_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return [query_loader, gallery_loader]
    #   return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def clean_segmentation(self, df, segmentation_col='segmentation'):
        """
        Cleans the segmentation column in the DataFrame to ensure each segmentation 
        is in the correct format (list where the first element is a float).
        
        Args:
            df (pd.DataFrame): The DataFrame containing metadata.
            segmentation_col (str): The name of the column containing segmentation data.
            
        Returns:
            pd.DataFrame: A cleaned DataFrame with valid segmentation data.
        """
        df[segmentation_col] = df[segmentation_col].apply(self.parse_segmentation)
        # Drop rows where segmentation is None (invalid)
        df_cleaned = df.dropna(subset=[segmentation_col])
        #DROP ROWS where seg is "nan" float value
        df_cleaned = df_cleaned[df_cleaned[segmentation_col].apply(lambda x: not isinstance(x, float))]
        print(f"Removed {len(df) - len(df_cleaned)} rows with invalid segmentation data.")
        return df_cleaned
    
    def parse_segmentation(self, segmentation):
        # Convert from string to list if needed
        if segmentation is None:
            print("No segmentation data found.: None")
            return None
        if isinstance(segmentation, str):
            try:
                segmentation = segmentation.replace("'", "\"")
                segmentation = json.loads(segmentation)
                # print("Parsed segmentation from string.")
            except json.JSONDecodeError:
                print("Segmentation is a string but not valid JSON.")
                return None

        if not isinstance(segmentation, list):
            print("No segmentation data found: Not a list after conversion.")
            return None
        # Check for empty lists or lists containing only empty lists
        if not segmentation or (len(segmentation) == 1 and not segmentation[0]):
            print("No segmentation data found: Empty list or list containing empty list.")
            return None
        
        # If the first element is a list of coordinates
        if isinstance(segmentation[0], list) and all(isinstance(coord, (int, float)) for coord in segmentation[0]):
            return segmentation  # Already in the correct format
        
        # If it's a flat list of coordinates, wrap it in a list
        if all(isinstance(coord, (int, float)) for coord in segmentation):
            return [segmentation]  # Convert to a list of lists
        
        # None of the conditions are met: return None
        print("Segmentation format is not recognized.")
        return None

class PrecomputedWildlife(WildlifeDataset):
    def __init__(
        self,
        metadata: pd.DataFrame | None = None,
        root: str | None = None,
        split: Callable | None = None,
        transform: callable = None,
        img_load: str = "full",
        col_path: str = "path",
        col_label: str = "identity",
        load_label: bool = True,
        load_metadata: bool = False,
        cache_dir: str = "../dataset/data_cache",
        category: str = "example_wildlife",
    ):
        
        metadata = metadata.sort_values(by=['identity', 'image_id']).reset_index(drop=True)  # Sort for consistency

        super().__init__(
            metadata=metadata,
            root=root,
            split = split,
            transform=transform,
            img_load=img_load,
            col_path=col_path,
            col_label=col_label,
            load_label=load_label
        )
        
        os.makedirs(cache_dir, exist_ok=True)
        self.load_metadata = load_metadata
        self.cache_dir = cache_dir 
        self.category = category
        self.multispecies = False

        if self.img_load not in ["bbox_mask", "bbox_mask_skeleton", "bbox_mask_components", "bbox_mask_heatmaps"]:
            raise ValueError(f"Invalid img_load argument: {self.img_load}. This class is only for preprocessed images.")

        input_category = category.split('_')
        if len(input_category) > 2: # ex. "train_raptors_BirdIndividualID"
            split = input_category[0]
            self.multispecies = True
            if 'wildlife_name' in self.metadata.columns:
                self.wildlife_names = self.metadata['wildlife_name'].unique()
            else:
                self.wildlife_names = np.array([input_category[i] for i in range(1, len(input_category))])

            self.cache_files = {
                name: {
                    "bbox_mask": os.path.join(self.cache_dir, split + "_" + name + "_mask.npz"),
                } for name in self.wildlife_names
            }
            self.data_cache = self._load_cache_multispecies()
        else:
            self.cache_files = {
                "bbox_mask": os.path.join(self.cache_dir, category + "_mask.npz"),
                "bbox_mask_skeleton": os.path.join(self.cache_dir, category +"_skeleton.npz"),
                "bbox_mask_heatmaps": os.path.join(self.cache_dir, category +"_heatmaps.npz"),
                "bbox_mask_components": os.path.join(self.cache_dir, category +"_components.npz")
            }
            self.data_cache = self._load_cache()
        print(f"Precomputed data loaded from {self.img_load} for {category}. Only to be used for processing lvl 2-5")

        print("Precomputed data loaded:")
        print(f"length of metadata: {len(self.metadata)}")
        print("first 5 rows of metadata:")
        print(self.metadata.head())

    def _load_cache_multispecies(self):
        """Load and combine mask NPZ files for all species in metadata."""
        data_cache = {'mask': {}}
        for name in self.wildlife_names:
            mask_cache_file = self.cache_files[name]["bbox_mask"]
            if not os.path.exists(mask_cache_file):
                raise FileNotFoundError(f"Mask cache file for category {name} not found: {mask_cache_file}")
            masks_data = np.load(mask_cache_file, allow_pickle=True)
            wname_masks = dict(masks_data["mask"].item())
            data_cache['mask'].update(wname_masks)
        return data_cache


    def _load_cache(self):
        """Load all precomputed data into memory as dictionaries"""
        # Load the masks cache regardless of self.img_load
        mask_cache_file = self.cache_files["bbox_mask"]
        if not os.path.exists(mask_cache_file):
            self._precompute_and_cache(mask = True)
        masks_data = np.load(mask_cache_file, allow_pickle=True)
        masks = dict(masks_data["mask"].item())
        print(f"Loaded mask cache from {mask_cache_file}: Masks count: {len(masks)}", flush=True)

        # load the primary data type for self.img_load
        primary_cache_file = self.cache_files[self.img_load]
        if not os.path.exists(primary_cache_file):
            self._precompute_and_cache()
        
        data = np.load(primary_cache_file, allow_pickle=True)
        # Extract the primary data based on the file naming convention.
        data_type = self.img_load.split('_')[-1]  # "skeleton", "heatmaps", "components", or "mask"
        cache = {data_type: dict(data[data_type].item())}
        
        count = len(cache[data_type])
        print(f"Loaded primary cache from {primary_cache_file}: {data_type.capitalize()}: {count}", flush=True)
        
        # Always include masks from the mask npz.
        cache["mask"] = masks
        return cache

    def _precompute_and_cache(self, mask=False):
        """Precompute and save only the data type required by img_load, as well as the masks."""
        p_type = "bbox_mask" if mask else self.img_load
        print(f"Starting precomputation for {p_type} ({len(self.metadata)} images)...", flush=True)
        data_type = p_type.split('_')[-1]  # "mask", "skeleton", "heatmaps", "components"
        all_masks = {}
        all_data = {}

        for idx in range(len(self.metadata)):
            data = self.metadata.iloc[idx]
            img_path = os.path.join(self.root, data[self.col_path])
            img_key = data[self.col_path] # chose this bc definitely indexable
            # img_key = os.path.splitext(os.path.basename(img_path))[0]

            # Enforce that segmentation must exist for these modes.
            if "segmentation" not in data or not data["segmentation"]:
                raise ValueError(f"{p_type} selected but no segmentation found for image {img_path}.")
            # For skeleton, heatmaps, or components, also require keypoints.
            if p_type in ["bbox_mask_skeleton", "bbox_mask_components", "bbox_mask_heatmaps"]:
                if "keypoints" not in data or not data["keypoints"]:
                    raise ValueError(f"{p_type} selected but no keypoints found for image {img_path}.")

            img = Image.open(img_path)
            bbox = self._parse_bbox(data)
            segmentation = self._parse_segmentation(data)

            # Always compute masks since they're used in __getitem__
            # all_masks[filename] = self._compute_mask(segmentation, img.size[1], img.size[0], bbox)
            all_masks[img_key] = self._compute_mask(segmentation, img.size[1], img.size[0], bbox)

            # Compute only the requested data type (if not 'masks' itself)
            if data_type == "mask":
                pass  # Only masks are needed.
            else:
                keypoints = self._parse_keypoints(data)
                if data_type == "skeleton":
                    all_data[img_key] = self._compute_skeleton(keypoints, bbox, img.size[1], img.size[0])
                elif data_type == "heatmaps":
                    all_data[img_key] = self._compute_heatmaps(keypoints, bbox, img.size[1], img.size[0])
                elif data_type == "components":
                    all_data[img_key] = self._compute_components(img, bbox, keypoints, segmentation)

        if data_type == "mask":
            np.savez_compressed(self.cache_files[p_type], mask=all_masks)
        else:
            # all_data = {f.stem.replace(f'_{data_type}', ''): np.load(f, allow_pickle=True) 
            #             for f in temp_dir.glob(f"*_{data_type}.npy")}
            np.savez_compressed(self.cache_files[p_type], masks=all_masks, **{data_type: all_data})

    def __len__(self):
        return len(self.metadata)
    
    def get_image(self, path):
        """
        Custom image loader, customized based on preprocessing level for raptor images.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def __getitem__(self, idx):
        """
        Get image from saved cache rather than recomputing it.
        """
        data = self.metadata.iloc[idx]
        if self.root:
            img_path = os.path.join(self.root, data[self.col_path])
        else:
            img_path = data[self.col_path]
        img_key = data[self.col_path]
        if self.multispecies:
            # remove the first part of the path that gets added in the cache
            img_key = '/'.join(img_key.split('/')[1:]) if '/' in img_key else img_key
        # img_key = os.path.splitext(os.path.basename(img_path))[0]
        img = self.get_image(img_path)
        
        # --- before transforms ---
        # 1. Crop image (bc saved seg and keypoint is relative to crop)
        if not ("bbox" in data):
            raise ValueError(f"{self.img_load} selected but no bbox found.")
        bbox = self._parse_bbox(data)
        img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Apply mask always
        mask = self.data_cache['mask'].get(img_key)
        if mask is None:
            raise ValueError(f"No mask found for {img_key} in cache")
        img_array = np.array(img)

        if mask.shape[:2] != img_array.shape[:2]:
            print(f"Idx {idx}: Shape mismatch! img_array {img_array.shape[:2]} vs mask {mask.shape[:2]}")
            segmentation = self._parse_segmentation(data)
            mask = self._compute_mask(segmentation, img.size[1], img.size[0])

        img_array = img_array * mask[..., None] if mask.ndim == 2 else img_array * mask
        img = Image.fromarray(img_array.astype('uint8'))


        # Apply transforms based on img_load type
        if self.img_load == "bbox_mask_skeleton":
            skeleton = self.data_cache['skeleton'].get(img_key)
        elif self.img_load == "bbox_mask_heatmaps":
            heatmaps = self.data_cache['heatmaps'].get(img_key)
        elif self.img_load == "bbox_mask_components":
            components = self.data_cache['components'].get(img_key)
        if self.transform:
            if self.img_load == "bbox_mask_skeleton" and skeleton is not None:
                img, skeleton_channel = self.transform[0](img, skeleton)
                img = self.transform[1](img, skeleton_channel)  # concatenated
            elif self.img_load == "bbox_mask_heatmaps" and heatmaps is not None:
                img, heatmap_channels = self.transform[0](img, np.stack(heatmaps) if isinstance(heatmaps, list) else heatmaps)
                # img, heatmap_channels = self.transform[0](img, np.stack(heatmaps))
                img = self.transform[1](img, heatmap_channels)  # concatenated
            elif self.img_load == "bbox_mask_components" and components is not None:
                img = self.transform[0](img)
                img = self.transform[1](img, components)  # concatenated
            else:
                # Standard image transform
                for t in self.transform:
                    img = t(img)

        # Return image and label
        if self.load_metadata:
            return {
                'img': img,
                'label': self.labels[idx],
                'path': data['path'],
                'identity' : data['identity'],
            }
        elif self.load_label: # default is True
            return img, self.labels[idx]
        else:
            return img

    def _parse_bbox(self, data):
        # if type(data["bbox"]) == str:
        #     bbox = json.loads(data["bbox"])
        # else:
        #     bbox = data["bbox"]
        # return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    
        bbox = data['bbox']
        bbox = eval(bbox) if isinstance(bbox, str) else bbox
        parsed_bbox = [int(round(coord)) for coord in bbox]
        return parsed_bbox
    
    def _parse_segmentation(self, data):
        seg = data['segmentation']
        return eval(seg) if isinstance(seg, str) else seg

    def _parse_keypoints(self, data):
        kp = data['keypoints']
        return eval(kp) if isinstance(kp, str) else kp
    
    def _compute_mask(self, segmentation, height, width, bbox=None):
        if isinstance(segmentation, list):
            rle = mask_coco.frPyObjects(segmentation, height, width)
            segmentation = mask_coco.merge(rle)
        mask = mask_coco.decode(segmentation).astype(bool)
        if bbox:
            # Convert bbox coordinates to integers
            x, y, w, h = [int(round(coord)) for coord in bbox]
            # Ensure we don't go out of bounds
            y_start = max(0, y)
            y_end = min(mask.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(mask.shape[1], x + w)
            mask = mask[y_start:y_end, x_start:x_end]
            if mask.shape != (h, w):
                print(f"Mask crop mismatch! Expected {(h, w)}, got {mask.shape}")
        return mask

    # def _compute_skeleton(self, keypoints, bbox, height, width):
    #     connections = get_skeleton_info()
    #     skeleton = create_skeleton_channel(keypoints, connections, height, width)
    #     x, y, w, h = bbox
    #     return skeleton[y:y+h, x:x+w]

    def _compute_skeleton(self, keypoints, bbox, height, width, crop_to_bbox=False):
        """
        Compute the skeleton channel, optionally cropped to bbox.
        
        Args:
            keypoints (list): List of keypoints.
            bbox (list): Bounding box [x, y, w, h].
            height (int): Full image height.
            width (int): Full image width.
            crop_to_bbox (bool): If True, compute on cropped region; if False, full image and crop after.
        
        Returns:
            np.array: Cropped skeleton channel.
        """
        if not keypoints or len(keypoints) < 2:
            x, y, w, h = [int(round(coord)) for coord in bbox]
            return np.zeros((h, w), dtype=np.float32)

        connections = get_skeleton_info()
        if crop_to_bbox:
            # Pass bbox to create_skeleton_channel for direct cropping
            skeleton = create_skeleton_channel(keypoints, connections, height, width, crop_to_bbox=bbox)
        else:
            # Compute on full image and crop afterward
            skeleton = create_skeleton_channel(keypoints, connections, height, width, crop_to_bbox=None)
            x, y, w, h = [int(round(coord)) for coord in bbox]
            skeleton = skeleton[y:y+h, x:x+w]
        
        return skeleton

    def _compute_heatmaps(self, keypoints, bbox, height, width):
        x, y, w, h = bbox
        heatmaps = create_multichannel_heatmaps(keypoints, height, width, w, h, 25)
        return heatmaps
        # return [heatmap[y:y+h, x:x+w] for heatmap in heatmaps]

    def _compute_components(self, img, bbox, keypoints, segmentation):
        img_array = np.array(img.convert("RGB"))
        keypoint_labels = get_keypoints_info()
        return component_generation_module(
            img_array, bbox, keypoints, keypoint_labels, True, segmentation
        )
