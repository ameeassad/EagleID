import json
import os
import math
import ast
import pandas as pd
import numpy as np

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
from data.data_utils import SplitQueryDatabase, analyze_split, RandomIdentitySampler
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
        # col_label_idx: str = None,
    ):    
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
        # metadata = metadata.reset_index(drop=True)
        # if col_label_idx != "tmp_idx":
        #     self.labels, self.labels_map = pd.factorize(
        #         metadata[col_label].values
        #     )
        # else:
        #     self.labels = metadata[col_label_idx].values

        # if col_label_idx == "tmp_idx":
        #     self.labels = metadata[col_label_idx].values

        # self.labels = self.labels  # Inherited from parent
        # self.labels_map = self.labels_map  # Inherited from 
        
        # self.metadata = metadata.reset_index(drop=True)
        # self.root = root
        # self.transform = transform
        # self.img_load = img_load
        # self.col_path = col_path
        # self.col_label = col_label
        # self.load_label = load_label

    # Preprocess and cache JSON or metadata:
    # Use json.loads during data preparation and store the parsed results.
    # Save processed metadata in a CSV or pickle file for direct loading.

    # Precompute masks, skeleton channels, heatmaps, or other intensive operations and save them 
    # to disk in a more accessible format (e.g., .npz, .h5, or PyTorch tensors).
    # Load precomputed data during training: img = load_preprocessed(mask_path)
    # def preprocess_data(metadata):
    #     for idx, data in metadata.iterrows():
    #         img_path = os.path.join(root, data[col_path])
    #         img = load_image(img_path)  # Load image using cv2 or Pillow
    #         if "segmentation" in data:
    #             mask = mask_coco.decode(eval(data["segmentation"])).astype("bool")
    #             save_preprocessed(mask, path)
    #         if "keypoints" in data:
    #             keypoints = eval(data["keypoints"])
    #             skeleton = create_skeleton_channel(keypoints, ...)
    #             save_preprocessed(skeleton, path)

        
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

        # print(img_path)

        if self.img_load in ["full_mask", "full_hide", "bbox_mask", "bbox_hide", "bbox_mask_skeleton", "bbox_mask_components", "bbox_mask_heatmaps"]:
            if not ("segmentation" in data):
                raise ValueError(f"{self.img_load} selected but no segmentation found.")
            
            if type(data["segmentation"]) == str:
                segmentation = eval(data["segmentation"])
                # try:
                #     segmentation = json.loads(data["segmentation"])
                # except json.JSONDecodeError:
                #     raise ValueError("Segmentation string could not be decoded. Ensure it is a valid JSON format.")
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

        if self.transform:
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
        if self.load_label: # default is True
            return img, self.labels[idx]
        else:
            return img

    def get_df(self) -> pd.DataFrame:
        return self.metadata
    
    # def get_query_df(self) -> pd.DataFrame:
    #     if 'query' in self.metadata.columns:
    #         # return self.metadata[self.metadata['query'] == True]
    #         return self.metadata[self.metadata['query']]
    #     else:
    #         print("No query column found.")
    #         return self.metadata
    
    # def get_gallery_df(self) -> pd.DataFrame:
    #     if 'query' in self.metadata.columns:
    #         return self.metadata[~self.metadata['query']]
    #     else:
    #         print("No query column found.")
    #         return self.metadata
    
    # def get_query_labels(self) -> np.ndarray:
    #     # return self.labels[self.metadata['query']]
    #     df = self.get_query_df()
    #     return df[self.col_label].values
    
    # def get_gallery_labels(self) -> np.ndarray:
    #     # return self.labels[~self.metadata['query']]
    #     df = self.get_gallery_df()
    #     return df[self.col_label].values
    

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
                 classic_transform=False
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

        # Handle transforms
        if self.preprocess_lvl == 3:
            resize_and_pad = t.ResizeAndPadBoth(self.size, skeleton=True)
            sync_transform = t.SynchTransforms(mean=self.mean, std=self.std, degrees=15, color_and_gaussian=True)
            sync_val_transform = t.SynchTransforms(mean=self.mean, std=self.std, degrees=0, color_and_gaussian=False)
            self.train_transforms =  [resize_and_pad, sync_transform]
            self.val_transforms = [resize_and_pad, sync_val_transform]  # everything except for color / gaussian transforms aka no someOf transforms
        elif self.preprocess_lvl == 4:
            resize_and_pad = t.ResizeAndPadRGB(self.size)
            sync_transform = t.ComponentGenerationTransforms(mean=self.mean, std=self.std, degrees=15, color_and_gaussian=True)
            sync_val_transform = t.ComponentGenerationTransforms(mean=self.mean, std=self.std, degrees=0, color_and_gaussian=False)
            self.train_transforms =  [resize_and_pad, sync_transform]
            self.val_transforms = [resize_and_pad, sync_val_transform]  # everything except for color / gaussian transforms
        elif self.preprocess_lvl == 5:
            resize_and_pad = t.ResizeAndPadBoth(self.size, skeleton=False)
            sync_transform = t.SynchMultiChannelTransforms(mean=self.mean, std=self.std, degrees=15, color_and_gaussian=True)
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
            if self.only_cache[1]:
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


        # Remove only one image per individual
        counts = df_all['identity'].value_counts()
        valid_identities = counts[df_all['identity'].value_counts() > 1].index
        # Filter the dataframe so each identity has 2 or more images
        df_all = df_all[df_all['identity'].isin(valid_identities)].reset_index(drop=True)

        # Split the dataset according to closed or open or default via metadata values
        if self.splitter == 'original_split':
            df_train = df_all[df_all['original_split'] == 'train']
            df_test = df_all[df_all['original_split'] == 'test']

            analyze_split(df_all, df_train.index, df_test.index)
        else:
            if self.splitter == 'closed':
                # Closed-set split (same individuals in train/test)
                splitter = splits.ClosedSetSplit(self.split_ratio)
            elif self.splitter == 'open':
                # Open-set split (some individuals only in test)
                splitter = splits.OpenSetSplit(self.split_ratio, 0.1)
                # TODO: Add an "unknown" class to the model’s output layer if new identities are expected
            for idx_train, idx_test in splitter.split(df_all):
                analyze_split(df_all, idx_train, idx_test)
            df_train, df_test = df_all.loc[idx_train], df_all.loc[idx_test]
        
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


        # Original identity label is 'identity' and the new factorized one for remaining identities is 'tmp_idx'
        # tmp_index is only for model training and cannot be used for evaluation
        # df_train_tmp_idx, train_labels_map = pd.factorize(df_train['identity'].values)
        # df_train['tmp_idx'] = df_train_tmp_idx
        # df_test_tmp_idx, test_labels_map = pd.factorize(df_test['identity'].values)
        # df_test['tmp_idx'] = df_test_tmp_idx

        # Split query set and gallery set for evaluation via SplitQueryDatabase
        self.val_split = SplitQueryDatabase()
        df_test = self.val_split(df_test)
        df_test['query'] = df_test['query'].astype(bool)
        df_query = df_test[df_test['query']]
        df_gallery = df_test[~df_test['query']]

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

        self.train_dataset = Wildlife(metadata=df_train, root=self.data_dir, transform=self.train_transforms, col_label = 'identity', img_load=self.method)

        self.val_query_dataset = Wildlife(metadata=df_query, root=self.data_dir, transform=self.val_transforms, col_label = 'identity', img_load=self.method)
        self.val_gallery_dataset = Wildlife(metadata=df_gallery, root=self.data_dir, transform=self.val_transforms, col_label = 'identity', img_load=self.method)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # If using Triplet loss need an anchor and a positive in same identity class: // handled by triplet loss function
        # sampler = RandomIdentitySampler(dataset=self.train_dataset, batch_size=self.batch_size)
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
    
    def val_dataloader(self):
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

