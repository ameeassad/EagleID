import pandas as pd
from wildlife_datasets import datasets, splits
from torch.utils.data import Dataset
from wildlife_tools.data.dataset import WildlifeDataset
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Callable

import pycocotools.mask as mask_coco
import numpy as np
import json
import math

import cv2
import re, os
import data. transforms as t
from data.transforms import RGBTransforms, ResizeAndPadRGB, ValTransforms, SynchTransforms, resize_and_pad, rotate_image

import ast
from preprocess.preprocess_utils import create_mask, create_skeleton_channel, create_multichannel_heatmaps
from preprocess.component_gen import component_generation_module
from preprocess.mmpose_fill import get_keypoints_info, get_skeleton_info


"""
datasets.DatasetFactory:
https://wildlifedatasets.github.io/wildlife-datasets/adding/

Column 	        Type 	    Description
bbox 	        List[float]             Bounding box in the form [x, y, w, h]. Therefore, the topleft corner has coordinates [x, y], while the bottomright corner has coordinates [x+w, y+h].
date 	        special 	            Timestamp of the photo. The preferred format is %Y-%m-%d %H:%M:%S from the datetime package but it is sufficient to be amenable to pd.to_datetime(x).
keypoints 	    List[float]             Keypoints coordinates in the image such as eyes or joints.
position 	    str 	                Position from each the photo was taken. The usual values are left and right.
segmentation 	List[float] or special 	Segmentation mask in the form [x1, y1, x2, y2, ...]. Additional format are possible such as file path to a mask image, or pytorch RLE.
species 	    str or List[str] 	    The depicted species for datasets with multiple species.
video 	        int 	                The index of a video.


Besides the dataframe, each dataset also contains some metadata. The metadata are saved in a separate csv file, which currently contains the following information. All entries are optional.
Column 	Description
name 	        Name of the dataset.
licenses 	    License file for the dataset.
licenses_url 	URL for the license file.
url 	        URL for the dataset.
cite 	        Citation in Google Scholar type of the paper.
animals 	    List of all animal species in the dataset.
real_animals 	Determines whether the dataset contains real animals.
reported_n_total 	    The reported number of total animals.
reported_n_identified 	The reported number of identified animals.
reported_n_photos 	    The reported number of photos.
wild 	        Determines whether the photos were taken in the wild.
clear_photos 	Determines whether the photos are clear.
pose 	        Determines whether the photos have one orientation (single), two orientation such as left and right flanks (double) or more (multiple).
unique_pattern 	Determines whether the animals have unique features (fur patern, fin shape) for recognition.
from_video 	    Determines whether the dataset was created from photos or videos.
cropped 	    Determines whether the photos are cropped.
span 	        The span of the dataset (the time difference between the last and first photos).
"""

class Raptors(datasets.DatasetFactory):

    def __init__(self, root: str, df = None, include_video = True):
        self.root = root
        if df is None:
            self.df = self.create_catalogue(include_video)
        else:
            self.df = df.copy()

        super().__init__(root=root, df=self.df)

    def create_catalogue(self, include_video=True) -> pd.DataFrame:
        """
        Create a DataFrame catalog of the dataset, including species, animal ID, image paths, and metadata.
        
        Returns:
            pd.DataFrame: A DataFrame with the catalog.
        """
        data = []
        identity_map = {}  # Dictionary to map identity to a unique identity_id
        identity_counter = 0  # Counter to assign new identity_id
        image_counter = 0  # Counter to assign image_id

        
        for species_name in os.listdir(self.root):
            species_path = os.path.join(self.root, species_name)
            
            if not os.path.isdir(species_path):
                continue  # Skip non-folder entries
            
            for animal_id in os.listdir(species_path):
                animal_path = os.path.join(species_path, animal_id)
                directory = os.path.join(species_name, animal_id)

                # Assign a unique identity_id if it's not already assigned
                if animal_id not in identity_map:
                    identity_map[animal_id] = identity_counter
                    identity_counter += 1
                
                if not os.path.isdir(animal_path):
                    continue  # Skip non-folder entries
                
                # Check for either images directly or video-based screenshots
                for sub_entry in os.listdir(animal_path):
                    # sub_entry is either a video folder or an image file
                    sub_path = os.path.join(animal_path, sub_entry)
                    
                    if os.path.isdir(sub_path) and include_video:
                        # It's a video folder; get screenshots inside the folder
                        for screenshot in os.listdir(sub_path):
                            if screenshot.endswith(('.jpg', '.jpeg', '.png')):
                                date = self.extract_date_from_filename(screenshot)
                                data.append({
                                    'image_id': int(image_counter),
                                    'species': species_name,
                                    'identity_id': identity_map[animal_id],
                                    'identity': animal_id,
                                    'path': os.path.join(directory, sub_entry, screenshot),
                                    'from_video': True,
                                    'video': sub_entry,
                                    'date': date
                                })
                                image_counter += 1
                    elif sub_entry.endswith(('.jpg', '.jpeg', '.png')):
                        # It's a direct image (not from video)
                        date = self.extract_date_from_filename(sub_entry)
                        data.append({
                            'image_id': int(image_counter),
                            'species': species_name,
                            'identity_id': identity_map[animal_id],
                            'identity': animal_id,
                            'path': os.path.join(directory, sub_entry),
                            'from_video': False,
                            'video': None,
                            'date': date
                        })
                        image_counter += 1
        
        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)
        df.reset_index(drop=True, inplace=True)
        # df = df.set_index('id') # Set 'id' as the new index
        return df

    def create_metadata(self) -> pd.DataFrame:
        """
        Create metadata DataFrame with additional details about the dataset.
        
        Returns:
            pd.DataFrame: A DataFrame with metadata information.
        """
        metadata = {
            'name': 'Raptor Celebs Dataset',
            'cite': "Amee Assad (2024). Master Thesis",
            'animals': os.listdir(self.root),
            'real_animals': True,
            'reported_n_total': len(os.listdir(self.root)),
            'wild': True,
            'from_video': True,
        }
        
        return pd.DataFrame([metadata])
    
    def extract_date_from_filename(self, filename: str) -> str:
        """
        Extract the year from the filename if it matches the pattern 20XX.
        
        Args:
            filename (str): The image filename.
        
        Returns:
            str: The extracted date in 'YYYY' format or None if no date is found.
        """
        match = re.search(r'20[0-2][0-9]', filename)
        if match:
            return int(match.group(0))  # Return the year as a string
        return int(2000)

    
class RaptorsWildlife(WildlifeDataset):
    """
    Custom Dataset for Raptors, inheriting from WildlifeDataset.
    Can be used to also call on Raptors class without already having the dataframe ready.
    """

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
    ):
        
        if metadata is None:
            raptor_dataset = Raptors(root)
            metadata = raptor_dataset.df

        super().__init__(
            metadata=metadata,
            root=root,
            transform=transform,
            img_load=img_load,
            col_path=col_path,
            col_label=col_label,
            load_label=load_label
        )
        self.raptor_dataset = Raptors(root, metadata)
        self.split = split
        if self.split:
            metadata = self.split(metadata)
        self.metadata = metadata.reset_index(drop=True)
        
    # def get_image(self, path):
    #     """
    #     Custom image loader, customized based on preprocessing level for raptor images.
    #     """
    #     img = cv2.imread(path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = Image.fromarray(img)
    #     return img

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

            if isinstance(segmentation, list):
                # Convert polygon to RLE
                print("segmentation: ", segmentation)
                height, width = int(data['height']), int(data['width'])
                print(height, width)
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
                img = self.transform[0](img)
                img = self.transform[1](img)

        if self.load_label:
            return img, self.labels[idx]
        else:
            return img
        
    
    def wildlife_dataset(self) -> Raptors:
        return self.raptor_dataset
    
    def get_df(self) -> pd.DataFrame:
        return self.metadata

# class GoldensWildlife(RaptorsWildlife):
#     """
#     Calls on Raptors directory and creates the dataset but only for golden eagles.
#     """
#     def __init__(
#         self,
#         metadata: pd.DataFrame | None = None,
#         root: str | None = None,
#         transform: callable = None,
#         **kwargs
#     ):
#         if metadata is None:
#             raptor_dataset = Raptors(root)
#             metadata = raptor_dataset.df[raptor_dataset.df['species'] == 'goleag']
#         else:
#             metadata = metadata[metadata['species'] == 'goleag']

#         super().__init__(metadata=metadata, root=root, transform=transform, **kwargs)
#         # if metadata is None:
#         #     raptor_dataset = Raptors(root)
#         #     metadata = raptor_dataset.df['species'] == 'goleag'
    
class WildlifeReidDataModule(pl.LightningDataModule):
    def __init__(self, metadata, config = None, data_dir="", preprocess_lvl=0, batch_size=8, size=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], num_workers=2, cache_path="../dataset/dataframe/cache.csv", animal_cat='bird', splitter ='closed', only_cache=False):
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
            resize_and_pad = t.ResizeAndPadRGB(self.size)
            rgb_transform = t.RGBTransforms(mean=self.mean, std=self.std, degrees=15, color_and_gaussian=True)
            rgb_val_tranform = t.RGBTransforms(mean=self.mean, std=self.std, degrees=0, color_and_gaussian=False)
            self.train_transforms =  [resize_and_pad, rgb_transform]
            self.val_transforms = [resize_and_pad, rgb_val_tranform]

        # Need to fix splits
        #  will i also offer open set split? aka some individuals in the query might not be present in the gallery. 
        # This implies that a query can return "unknown" results if the individual is not part of the gallery.
        if 'date' in self.metadata.columns and self.metadata['date'].isna().any():
            self.metadata.loc[pd.isna(self.metadata['date']), 'date'] = "unknown" 
        
        if self.splitter == 'closed':
            splitter = splits.ClosedSetSplit(self.split_ratio) # All individuals are both in the training and testing set.
        elif self.splitter == 'open':
            splitter = splits.OpenSetSplit(self.split_ratio, 0.1) # Some individuals are in the testing but not in the training set
        for idx_train, idx_test in splitter.split(metadata):
            splits.analyze_split(self.metadata, idx_train, idx_test)

        df_train, df_test = self.metadata.loc[idx_train], metadata.loc[idx_test]
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        print(f"Train set size: {len(df_train)}")
        print(f"Test set size: {len(df_test)}")

        # df_test = self.filter_identities(df_test)
        df_query, df_gallery = self.split_query_database(df_test)

        # preprocessing
        if self.preprocess_lvl > 0: # 1: bounding box cropped image or 2: masked image
            from preprocess.segmenter import add_segmentations

            df_train = add_segmentations(df_train, self.data_dir, cache_path=self.cache_path, only_cache=self.only_cache)
            df_query = add_segmentations(df_query, self.data_dir, cache_path=self.cache_path, only_cache=self.only_cache)
            df_gallery = add_segmentations(df_gallery, self.data_dir, cache_path=self.cache_path, only_cache=self.only_cache)
        
        if self.preprocess_lvl >= 3: # 3: masked + pose (skeleton) image in 1 channel or 4: masked + body part clusters in channels
            from preprocess.mmpose_fill import fill_keypoints, get_skeleton_info, get_keypoints_info

            self.keypoints = get_keypoints_info()
            self.skeleton = get_skeleton_info()

            df_train = fill_keypoints(df_train, self.data_dir, cache_path=self.cache_path, only_cache=self.only_cache, animal_cat=self.animal_cat)
            df_query = fill_keypoints(df_query, self.data_dir, cache_path=self.cache_path, only_cache=self.only_cache, animal_cat=self.animal_cat)
            df_gallery = fill_keypoints(df_gallery, self.data_dir, cache_path=self.cache_path, only_cache=self.only_cache, animal_cat=self.animal_cat)

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

        print(f"length of training dataset: {len(df_train)}")
        self.train_dataset = RaptorsWildlife(metadata=df_train, root=self.data_dir, transform=self.train_transforms, img_load=self.method)
        print(f"length of query dataset: {len(df_query)}")
        if not (len(df_query) == 0 or len(df_gallery) == 0):
            self.val_query_dataset = RaptorsWildlife(metadata=df_query, root=self.data_dir, transform=self.val_transforms, img_load=self.method)
            print(f"length of gallery dataset: {len(df_gallery)}")
            self.val_gallery_dataset = RaptorsWildlife(metadata=df_gallery, root=self.data_dir, transform=self.val_transforms, img_load=self.method)
            self.val_query_dataset.metadata.reset_index(drop=True, inplace=True)
            self.val_gallery_dataset.metadata.reset_index(drop=True, inplace=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        query_loader = DataLoader(self.val_query_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        gallery_loader = DataLoader(self.val_gallery_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return [query_loader, gallery_loader]
    
    def split_query_database(self, test_metadata):
        """
        Splits the test metadata into query and database sets ensuring each identity
        is present in both query and database datasets, without data leakage (i.e.,
        the same image cannot be in both query and database).

        Args:
            test_metadata (pd.DataFrame): Dataframe containing test image metadata.

        Returns:
            query_metadata (pd.DataFrame): Metadata for the query set.
            database_metadata (pd.DataFrame): Metadata for the database set.
        """
        identities = test_metadata['identity'].unique()
        query_indices = []
        database_indices = []

        for identity in identities:
            identity_indices = test_metadata[test_metadata['identity'] == identity].index.tolist()
            if len(identity_indices) > 1:
                query_indices.append(identity_indices[0])
                database_indices.extend(identity_indices[1:])
            else:
                # If only one image, add to gallery but not query
                database_indices.append(identity_indices[0])

        query_metadata = test_metadata.loc[query_indices].reset_index(drop=True)
        database_metadata = test_metadata.loc[database_indices].reset_index(drop=True)

        return query_metadata, database_metadata


    def filter_identities(self, metadata, min_images=2):
        """
        Filters out identities with fewer than a minimum number of images.

        Args:
            metadata (pd.DataFrame): Dataframe containing image metadata.
            min_images (int): Minimum number of images required per identity.

        Returns:
            filtered_metadata (pd.DataFrame): Metadata filtered to only include identities with the required number of images.
        """
        identity_counts = metadata['identity'].value_counts()
        valid_identities = identity_counts[identity_counts >= min_images].index
        return metadata[metadata['identity'].isin(valid_identities)].reset_index(drop=True)
    
    def get_img_channels(self):
        return self.img_channels

