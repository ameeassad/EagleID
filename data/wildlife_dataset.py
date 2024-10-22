"""
Modified class from wildlife-tools to support preprocessing levels. 
"""

import json
import os
import pickle
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_coco
from PIL import Image

from wildlife_tools.tools import realize

import pandas as pd
from typing import Callable

import pycocotools.mask as mask_coco
import numpy as np
import json
import math

import cv2
import os

import ast
from preprocess.preprocess_utils import create_mask, create_skeleton_channel, create_multichannel_heatmaps
from preprocess.component_gen import component_generation_module
from preprocess.mmpose_fill import get_keypoints_info, get_skeleton_info


class WildlifeDataset:
    """
    PyTorch-style dataset for a wildlife datasets

    Args:
        metadata: A pandas dataframe containing image metadata.
        root: Root directory if paths in metadata are relative. If None, paths in metadata are used.
        split: A function that splits metadata, e.g., instance of data.Split.
        transform: A function that takes in an image and returns its transformed version.
        img_load: Method to load images.
            Options: 'full', 'full_mask', 'full_hide', 'bbox', 'bbox_mask', 'bbox_hide',
                      and 'crop_black'.
        col_path: Column name in the metadata containing image file paths.
        col_label: Column name in the metadata containing class labels.
        load_label: If False, \_\_getitem\_\_ returns only image instead of (image, label) tuple.

    Attributes:
        labels np.array : An integers array of ordinal encoding of labels.
        labels_string np.array: A strings array of original labels.
        labels_map dict: A mapping between labels and their ordinal encoding.
        num_classes int: Return the number of unique classes in the dataset.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        root: str | None = None,
        split: Callable | None = None,
        transform: Callable | None = None,
        img_load: str = "full",
        col_path: str = "path",
        col_label: str = "identity",
        load_label: bool = True,
    ):
        self.split = split
        if self.split:
            metadata = self.split(metadata)

        self.metadata = metadata.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.img_load = img_load
        self.col_path = col_path
        self.col_label = col_label
        self.load_label = load_label
        self.labels, self.labels_map = pd.factorize(
            self.metadata[self.col_label].values
        )

    @property
    def labels_string(self):
        return self.metadata[self.col_label].astype(str).values

    @property
    def num_classes(self):
        return len(self.labels_map)

    def __len__(self):
        return len(self.metadata)

    def get_image(self, path):
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        except: 
            print(f"Error loading image: {path}")
            img = np.full((224, 224, 3), (0, 0, 0), dtype=np.uint8)
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
                if type(segmentation[0]) == int:
                    segmentation = [segmentation]
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
        
    def get_df(self) -> pd.DataFrame:
        return self.metadata
        
    @classmethod
    def from_config(cls, config):
        config["split"] = realize(config.get("split"))
        config["transform"] = realize(config.get("transform"))
        config["metadata"] = pd.read_csv(config["metadata"], index_col=False)
        return cls(**config)


class FeatureDataset:
    def __init__(
        self,
        features,
        metadata,
        col_label="identity",
        load_label=True,
    ):

        if len(features) != len(metadata):
            raise ValueError("Features and metadata (lables) have different length ! ")

        self.load_label = load_label
        self.features = features
        self.metadata = metadata.reset_index(drop=True)
        self.col_label = col_label
        self.labels, self.labels_map = pd.factorize(
            self.metadata[self.col_label].values
        )

    @property
    def labels_string(self):
        return self.metadata[self.col_label].astype(str).values

    def __getitem__(self, idx):
        if self.load_label:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

    def __len__(self):
        return len(self.metadata)

    @property
    def num_classes(self):
        return len(self.labels_map)

    def save(self, path):
        data = {
            "features": self.features,
            "metadata": self.metadata,
            "col_label": self.col_label,
            "load_label": self.load_label,
        }
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, path, **config):
        with open(path, "rb") as file:
            data = pickle.load(file)
        return cls(**data, **config)

    @classmethod
    def from_config(cls, config):
        path = config.pop("path")
        return cls.load(path, **config)


class FeatureDatabase(FeatureDataset):
    """Alias for FeatureDataset"""

    pass
