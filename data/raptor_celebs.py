from pycocotools.coco import COCO
import numpy as np
import os
import ast
import math
import pandas as pd
import json
from PIL import Image
import cv2

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)
from torchvision.transforms.functional import resize, pad
import pytorch_lightning as pl

from .data_utils import create_skeleton_channel, unnormalize
from .skeleton_category import AKSkeletonCategory
from preprocess.segmenterCOCO import COCOBuilder
from data.transforms import SynchTransforms, RGBTransforms, ValTransforms


class RaptorCelebs(pl.LightningDataModule):
    """
    Lightning DataModule for the raptor individuals dataset, 
    handles the loading, preprocessing, and transformation of the dataset 
    for training, validation, and testing.

    Args:
        data_dir (str): Path to the dataset directory.
        preprocess_lvl (int): Level of preprocessing to apply to the images.
            0: original image --> not used
            1: bounding box cropped image
            2: masked image
            3: masked + pose (skeleton) image in 1 channel
            4: masked + body parts in channels

        batch_size (int): Number of samples per batch.
        size (int): Size of the image for resizing.
        mean (float or tuple): Mean for normalization.
        std (float or tuple): Standard deviation for normalization.
        test (bool): Flag to indicate if in test mode.


    Attributes:
        train_transforms (callable): Transformations applied to the training dataset.
        val_transforms (callable): Transformations applied to the validation dataset.
    """
    def __init__(self, data_dir, preprocess_lvl=0, batch_size=8, size=256, mean=0.5, std=0.5, test=False, is_demo=False):
        super().__init__()
        self.data_dir = data_dir
        self.preprocess_lvl = preprocess_lvl
        self.batch_size = batch_size
        self.size = size
        self.mean = (mean, mean, mean) if isinstance(mean, float) else tuple(mean)
        self.std = (std, std, std) if isinstance(std, float) else tuple(std)
        self.test = test

    def setup(self, stage=None):
        self.train_dataset = TripletDataset(self.train_imgs, transform=self.train_transforms)

        if self.test:
            self.val_query_dataset = TripletDataset(self.val_query_imgs, mode='query', transform=self.val_transforms)
            self.val_gallery_dataset = TripletDataset(self.val_gallery_imgs, mode='gallery', transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        query_loader = DataLoader(self.val_query_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        gallery_loader = DataLoader(self.val_gallery_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return {'query': query_loader, 'gallery': gallery_loader}


class TripletDataset(Dataset):
    """
    Dataset for re-identification tasks using triplet loss.
    Provides (anchor, positive, negative) triplets.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory containing the dataset images.
            transform (callable, optional): Transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = glob.glob(osp.join(data_dir, '*.jpg'))
        self.pattern = re.compile(r'([-\d]+)_c(\d)')  # Regex for ID and camera parsing
        self.pid2img = self._group_images_by_pid()

    def _group_images_by_pid(self):
        """
        Groups image paths by person ID (pid).
        Returns:
            dict: Dictionary mapping each pid to a list of image paths.
        """
        pid2img = {}
        for img_path in self.img_paths:
            pid, _ = map(int, self.pattern.search(img_path).groups())
            if pid == -1:  # Exclude junk images
                continue
            if pid not in pid2img:
                pid2img[pid] = []
            pid2img[pid].append(img_path)
        return pid2img

    def _get_triplet(self, anchor_path):
        """
        Generates a triplet (anchor, positive, negative).
        Args:
            anchor_path (str): File path to the anchor image.
        
        Returns:
            tuple: Anchor, positive, and negative images.
        """
        anchor_pid, _ = map(int, self.pattern.search(anchor_path).groups())
        positive_path = random.choice(self.pid2img[anchor_pid])

        # Ensure we get a negative from a different class
        negative_pid = random.choice([pid for pid in self.pid2img.keys() if pid != anchor_pid])
        negative_path = random.choice(self.pid2img[negative_pid])

        return anchor_path, positive_path, negative_path

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        anchor_path = self.img_paths[idx]
        anchor_path, positive_path, negative_path = self._get_triplet(anchor_path)

        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img