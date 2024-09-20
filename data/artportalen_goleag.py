from pycocotools.coco import COCO
import numpy as np
import os
import ast
import math
import pickle
import pandas as pd
import json
from IPython.display import display
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    Pad,
    RandomRotation,
    ColorJitter,
)
import torchvision.transforms.functional as F
from torchvision.transforms.functional import resize, pad
import pytorch_lightning as pl

from segmentation import COCOBuilder
from transforms import SynchTransforms, RGBTransforms, ValTransforms, RGBSkelTransforms, SkelTransforms


class ArtportalenDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for the Artportalen dataset, handles the loading, 
    preprocessing, and transformation of the dataset 
    for training, validation, and testing.

    Args:
        data_dir (str): Path to the dataset directory.
        preprocess_lvl (int): Level of preprocessing to apply to the images.
            0: original image
            1: bounding box cropped image
            2: masked image
            3: masked + pose (skeleton) image in 1 channel
            4: masked + body parts in channels

        batch_size (int): Number of samples per batch.
        size (int): Size of the image for resizing.
        mean (float or tuple): Mean for normalization.
        std (float or tuple): Standard deviation for normalization.
        test (bool): Flag to indicate if in test mode.
        skeleton (bool): Whether to include the skeleton channel in the data.

    Attributes:
        train_transforms (callable): Transformations applied to the training dataset.
        val_transforms (callable): Transformations applied to the validation dataset.
    """
    def __init__(self, data_dir, preprocess_lvl=0, batch_size=8, size=256, mean=0.5, std=0.5, test=False, cache_dir=None, skeleton=False):
        super().__init__()
        self.data_dir = data_dir
        self.preprocess_lvl = preprocess_lvl
        self.batch_size = batch_size
        self.size = size
        self.mean = (mean, mean, mean) if isinstance(mean, float) else tuple(mean)
        self.std = (std, std, std) if isinstance(std, float) else tuple(std)
        self.test = test
        self.cache_dir = cache_dir

        if preprocess_lvl == 3:
            self.skeleton = True
        else:
            self.skeleton = False
        

        # transformations
        if self.skeleton:         
            self.train_transforms = SynchTransforms(mean=self.mean, std=self.std)
            self.val_transforms = ValTransforms(mean=self.mean, std=self.std, skeleton=True)
        else:
            self.train_transforms = RGBTransforms(mean=self.mean, std=self.std)
            self.val_transforms = Compose([
                # Resize(self.size),
                # Pad((self.size - 1, self.size - 1), padding_mode='constant'),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std)
            ])

    def prepare_data(self):
        """
        Prepares the dataset, downloading or splitting it if needed. 
        In test mode, it prepares the testing data.
        """
        # download, split, etc.
        if self.test:
            self.prepare_testing_data(self.data_dir)

    def prepare_testing_data(self, image_dir):
        """
        Prepares the testing data by creating a COCO-like JSON annotation from the images.
        Inside COCOBuilder, it uses YOLOv8 to detect the bounding boxes and segmentations.
        Finally calls setup_testing().

        Args:
            image_dir (str): Path to the directory containing the test images.
        """
        coco = COCOBuilder("./testing/images", testing=True)
        coco.setup_testing()
        coco.fill_coco()
        coco.create_coco_format_json("testing/coco_training.json")

        self.setup_testing("testing/coco_training.json")

    def setup_testing(self, test_annot):
        """
        Set up the testing dataset using COCO annotations and convert it to a DataFrame.
        Is called in prepare_testing_data().

        Args:
            test_annot (str): Path to the COCO annotations file for testing.
        """
        # Load COCO annotations
        with open(test_annot, 'r') as f:
            test_data = json.load(f)
        # Initialize COCO objects
        test_coco = COCO(test_annot)
        # Convert annotations to DataFrame
        test_df = self.coco_to_dataframe(test_coco)
        print(f"Test: {len(test_df)}")
        self.num_classes = 5
        print(f"Number of classes: {self.num_classes}")
        self.train_dataset = EagleDataset(test_df, self.data_dir, self.train_transforms)
        self.val_dataset = EagleDataset(test_df, self.data_dir, self.val_transforms, test=self.test)

    def setup_from_csv(self, train_csv, val_csv, stage=None):
        """
        Set up the dataset using CSV files containing the training and validation data.

        Args:
            train_csv (str): Path to the CSV file for training data.
            val_csv (str): Path to the CSV file for validation data.
            stage (str, optional): The stage for which the setup is being done (e.g., 'fit', 'test').
        """
         # Load the CSV files
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        # Columns renaming
        column_names = {
            'annot_id': 'id',
            'image_filename': 'file_name',
            'Reporter': 'photographer',
        }

        def jpg_extension(filename):
            filename = str(filename)
            if not filename.lower().endswith('.jpg') or not filename.lower().endswith('.jpeg') or not filename.lower().endswith('.png'):
                return f"{filename}.jpg"
            return filename

        train_df = train_df.rename(columns=column_names)
        train_df['file_name'] = train_df['file_name'].apply(jpg_extension)
        val_df = val_df.rename(columns=column_names)
        val_df['file_name'] = val_df['file_name'].apply(jpg_extension)

        # Print the number of samples in train and validation sets
        print(f"Train: {len(train_df)} Val: {len(val_df)}")

        # Initialize the datasets
        self.train_dataset = EagleDataset(train_df, self.data_dir, self.train_transforms, skeleton=self.skeleton)
        self.val_dataset = EagleDataset(val_df, self.data_dir, self.val_transforms, skeleton=self.skeleton)

        # Check the number of unique classes
        unique_classes = train_df['category_id'].unique()
        print(f"Unique classes in dataset: {unique_classes}")
        self.num_classes = len(unique_classes)
        print(f"Number of classes: {self.num_classes}")

    def setup_from_coco(self, train_annot, val_annot, stage=None):
        """
        Set up the dataset using COCO-style annotations for training and validation.

        Args:
            train_annot (str): Path to the COCO annotations file for training.
            val_annot (str): Path to the COCO annotations file for validation.
            stage (str, optional): The stage for which the setup is being done (e.g., 'fit', 'test').
        """
        # Load COCO annotations
        with open(train_annot, 'r') as f:
            train_data = json.load(f)
        with open(val_annot, 'r') as f:
            val_data = json.load(f)

        # Initialize COCO objects
        train_coco = COCO(train_annot)
        val_coco = COCO(val_annot)

        # Convert annotations to DataFrame
        train_df = self.coco_to_dataframe(train_coco)
        val_df = self.coco_to_dataframe(val_coco)

        print(f"Train: {len(train_df)} Val: {len(val_df)}")

        self.train_dataset = EagleDataset(train_df, self.data_dir, self.train_transforms, skeleton=self.skeleton)
        self.val_dataset = EagleDataset(val_df, self.data_dir, self.val_transforms, skeleton=self.skeleton)

        # Check number of classes
        unique_classes = train_df['category_id'].unique()
        print(f"Unique classes in dataset: {unique_classes}")
        self.num_classes = len(unique_classes)
        print(f"Number of classes: {self.num_classes}")

    def coco_to_dataframe(self, coco):
        """
        Converts COCO annotations to a pandas DataFrame.

        Args:
            coco (COCO): COCO object containing annotations.

        Returns:
            pd.DataFrame: DataFrame with image and annotation information.
        """
        data = []
        for ann in coco.anns.values():
            img_info = coco.loadImgs(ann['image_id'])[0]

            file_name = img_info['file_name']
            if '.' not in file_name:
                file_name += '.jpg'

            data.append({
                # 'id': ann['id'], # Annotation ID
                'image_id': ann['image_id'],
                'file_name': file_name,
                'height': img_info['height'],
                'width': img_info['width'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann['iscrowd'],
                'segmentation': ann['segmentation'],

            })
        return pd.DataFrame(data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


class EagleDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing eagle image data with optional skeleton.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image file paths and annotations.
        data_dir (str): Directory containing the image files.
        transform (callable, optional): Transformations to be applied to the images.
        size (int): Size to which the images should be resized.
        test (bool): Whether this is test data.
        skeleton (bool): Whether to include skeleton channel.
    """
    def __init__(self, dataframe, data_dir, transform=None, size=256, test=False, preprocess_lvl=0, skeleton=False):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.size = size
        self.test = test
        self.preprocess_lvl = preprocess_lvl
        self.skeleton = skeleton

        # Load cache from disk if available
        # self.mask_cache = self.load_cache('mask_cache.pkl') if cache_dir else {}
        self.mask_cache = {}
        self.mask_dir = f'data_cache/masks_{size}'
        os.makedirs(self.mask_dir, exist_ok=True)

        if skeleton:
            # self.skeleton_transform = skeleton
            self.skeleton_category = AKSkeletonCategory()
            self.skeleton_cache = {}
            # self.skeleton_cache = self.load_cache('skeleton_cache.pkl') if cache_dir and skeleton else {}
            self.skeleton_dir = f'data_cache/skeletons_{size}'
            os.makedirs(self.skeleton_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Returns an item (image and label) from the dataset at a given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: Tuple containing the image and its corresponding label.
        """
        annot_info = self.dataframe.iloc[idx]
        img_path = os.path.join(self.data_dir, str(annot_info['file_name']))
        annot_id = annot_info['id']
        label = annot_info['category_id'] - 1 
        if self.test:
            label = annot_info['category_id']

        # Check cache for precomputed mask and skeleton
        mask_filename = os.path.join(self.mask_dir, f"{annot_id}.png")
        if self.skeleton:
            skeleton_filename = os.path.join(self.skeleton_dir, f"{annot_id}.npy")
        if os.path.exists(mask_filename):
            masked_image = Image.open(mask_filename)
        # if idx in self.mask_cache:
        #     masked_image = self.mask_cache[idx]
            if self.skeleton:
                # skeleton_channel = self.skeleton_cache[idx]
                if os.path.exists(skeleton_filename):
                    skeleton_channel = np.load(skeleton_filename)
        else:
            image = Image.open(img_path).convert("RGB")

            if self.skeleton:
                keypoints = annot_info['keypoints']
                # Convert keypoints from string to list if necessary
                if isinstance(keypoints, str):
                    keypoints = ast.literal_eval(keypoints)
                connections = self.skeleton_category.get_connections()
                # Convert connections from string to list if necessary
                if isinstance(connections, str):
                    connections = ast.literal_eval(connections)
                skeleton_channel = create_skeleton_channel(keypoints, connections, height=image.size[0], width=image.size[1])

            # Extract bounding box and crop the image
            bbox = ast.literal_eval(annot_info['bbox'])
            x_min = math.floor(bbox[0])
            y_min = math.floor(bbox[1])
            w = math.ceil(bbox[2])
            h = math.ceil(bbox[3])
            bbox = [x_min, y_min, w, h]

            segmentation = ast.literal_eval(annot_info['segmentation'])
            mask = self.create_mask(image.size, segmentation)

            # masked_image = np.array(cropped_image) * np.expand_dims(cropped_mask, axis=2)
            masked_image = np.array(image) * np.expand_dims(mask, axis=2)
            masked_image = Image.fromarray(masked_image.astype('uint8'))

            # Crop the image and the mask to the bounding box
            masked_image = masked_image.crop((x_min, y_min, x_min + w, y_min + h))

            self.mask_cache[idx] = masked_image
            # Save mask
            masked_image.save(mask_filename)  # Save the cropped image as it is



            if self.skeleton:
                skeleton_channel = skeleton_channel[y_min:y_min + h, x_min:x_min + w]

                self.skeleton_cache[idx] = skeleton_channel
                # Save skeleton channel as a numpy file
                np.save(skeleton_filename, skeleton_channel)

        # resize, pad, transform (cached or newly computed images)
        if self.skeleton:
            # skeleton_channel = skeleton_channel[y_min:y_min + h, x_min:x_min + w]
            # print(skeleton_channel.shape)
            # print(skeleton_channel)
            masked_image, skeleton_channel = self.resize_and_pad(masked_image, self.size, skeleton_channel=skeleton_channel)
            masked_image = self.transform(masked_image, skeleton_channel)
        elif self.transform:
            masked_image, _ = self.resize_and_pad(masked_image, self.size)
            masked_image = self.transform(masked_image)
        
        return masked_image, label

    def create_mask(self, image_size, segmentation):
        """
        Creates a binary mask based on the segmentation of the object.

        Args:
            image_size (tuple): Size of the original image.
            segmentation (list): COCO-style segmentation of the object.

        Returns:
            np.array: Binary mask of the object.
        """
        mask = np.zeros(image_size[::-1], dtype=np.uint8)
        for seg in segmentation:
            poly = np.array(seg).reshape((len(seg) // 2, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        return mask
    
    def resize_and_pad(self, image, size, skeleton_channel=None):
        """
        Resizes and pads both the RGB image and the skeleton channel based on the RGB image dimensions.

        Args:
            image (PIL.Image): RGB image to be resized and padded.
            size (int): Target size for resizing.
            skeleton_channel (np.array): Skeleton channel to be resized and padded.

        Returns:
            tuple: Resized and padded RGB image and skeleton channel.
        """


        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = size
            new_height = math.ceil(new_width / aspect_ratio)
        else:
            new_height = size
            new_width = math.ceil(new_height * aspect_ratio)

        resized_image = resize(image, (new_height, new_width))

        # Calculate padding
        pad_width = size - new_width
        pad_height = size - new_height

        padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
        
        # Pad the image to make it square
        padded_image = pad(resized_image, padding, fill=0, padding_mode='constant')

        # Padding for skeleton channel (using np.pad)
        if skeleton_channel is not None and skeleton_channel.size > 0:
            skeleton_channel_resized = cv2.resize(skeleton_channel, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            padding_skeleton = ((pad_height // 2, pad_height - (pad_height // 2)),
                                (pad_width // 2, pad_width - (pad_width // 2)))
            padded_skeleton_channel = np.pad(skeleton_channel_resized, padding_skeleton, mode='constant', constant_values=0)

        else:
            padded_skeleton_channel = np.zeros((size, size), dtype=np.float32)


        return padded_image, padded_skeleton_channel



def unnormalize(x, mean, std):
    """
    Unnormalizes a tensor by applying the inverse of the normalization transform.

    Args:
        x (torch.Tensor): Tensor to be unnormalized.
        mean (tuple): Mean used for normalization.
        std (tuple): Standard deviation used for normalization.

    Returns:
        torch.Tensor: Unnormalized tensor.
    """
    x = x.clone().detach()[:3]

    mean = (mean, mean, mean) if isinstance(mean, float) else tuple(mean)
    std = (std, std, std) if isinstance(std, float) else tuple(std)

    mean = torch.tensor(mean)
    std = torch.tensor(std)
    
    if x.dim() == 3:  # Ensure the tensor has 3 dimensions
        unnormalized_x = x.clone()
        for t, m, s in zip(unnormalized_x, mean, std):
            t.mul_(s).add_(m)
        return unnormalized_x
    else:
        raise ValueError(f"Expected input tensor to have 3 dimensions, but got {x.dim()} dimensions.")


class AKSkeletonCategory:
    """
    Handles the extraction and organization of skeleton keypoints and connections from the COCO annotations.

    Args:
        coco_data (dict): COCO-style annotations.

    Attributes:
        connections (list): List of connections (limbs) between keypoints.
        joint_names (list): Names of the keypoints (joints).
    """
    def __init__(self, coco_data=None):
        if coco_data is None:
            coco_data = { 
                "categories" : [{
                    "supercategory" : "bird",
                    "id" : 1,
                    "name" : "eagle",
                }]
            }
        for cat in coco_data['categories']:
            if not cat.get('keypoints'):
                cat['keypoints'] = [
                    "Head_Mid_Top",
                    "Eye_Left",
                    "Eye_Right",
                    "Mouth_Front_Top",
                    "Mouth_Back_Left",
                    "Mouth_Back_Right",
                    "Mouth_Front_Bottom",
                    "Shoulder_Left",
                    "Shoulder_Right",
                    "Elbow_Left",
                    "Elbow_Right",
                    "Wrist_Left",
                    "Wrist_Right",
                    "Torso_Mid_Back",
                    "Hip_Left",
                    "Hip_Right",
                    "Knee_Left",
                    "Knee_Right",
                    "Ankle_Left",
                    "Ankle_Right",
                    "Tail_Top_Back",
                    "Tail_Mid_Back",
                    "Tail_End_Back"
                ]
                cat['skeleton'] = [
                    [2,1],
                    [3,1],
                    [4,5],
                    [4,6],
                    [7,5],
                    [7,6],
                    [1,14],
                    [14,21],
                    [21,22],
                    [22,23],
                    [1,8],
                    [1,9],
                    [8,10],
                    [9,11],
                    [10,12],
                    [11,13],
                    [21,15],
                    [21,16],
                    [15,17],
                    [16,18],
                    [17,19],
                    [18,20]
                ]
                self.connections = cat['skeleton']
                self.joint_names = cat['keypoints']
        self.coco_data = coco_data

    def __call__(self):
        return self.coco_data

    def get_updated_categories(self):
        return self.coco_data['categories']
    
    def get_connections(self):
        return self.connections
    
    def get_joint_names(self):  
        return self.joint_names


def create_skeleton_channel(keypoints, connections, height, width, sigma=2, thickness=2):
    """
    Create a 4th channel for the model input representing the skeleton.
    
    Args:
        keypoints (list): List of flattened COCO-style keypoints (x, y, visibility).
        connections (list): List of (start_idx, end_idx) for limbs, based on keypoint indices.
        height (int): Height of the image.
        width (int): Width of the image.
        sigma (int): Gaussian blur for keypoints.
        thickness (int): Thickness of the drawn limbs.
    
    Returns:
        skeleton_channel (np.array): The skeleton channel.
    """
    # Initialize heatmap and skeleton channel
    heatmap = np.zeros((height, width), dtype=np.float32)
    skeleton_channel = np.zeros((height, width), dtype=np.float32)
    
    # Create heatmaps for keypoints
    for i in range(0, len(keypoints), 3):
        x, y, visibility = float(keypoints[i]), float(keypoints[i+1]), int(keypoints[i+2])
        
        # Skip keypoints that are not visible or invalid
        if visibility == 0 or x < 0 or y < 0:
            continue

        # Create a Gaussian blob centered at (x, y)
        for h in range(height):
            for w in range(width):
                heatmap[h, w] += np.exp(-((w - x) ** 2 + (h - y) ** 2) / (2 * sigma ** 2))

    # Draw limbs on the skeleton channel
    for (start_idx, end_idx) in connections:
        start_x, start_y, start_vis = keypoints[(start_idx - 1) * 3:(start_idx - 1) * 3 + 3]
        end_x, end_y, end_vis = keypoints[(end_idx - 1) * 3:(end_idx - 1) * 3 + 3]

        # Draw line only if both keypoints are visible
        if start_vis > 0 and end_vis > 0:
            start_point = (int(start_x), int(start_y))
            end_point = (int(end_x), int(end_y))
            cv2.line(skeleton_channel, start_point, end_point, 1, thickness)

    # Combine keypoints heatmap and skeleton lines
    skeleton_channel += heatmap
    skeleton_channel = np.clip(skeleton_channel, 0, 1)  # Normalize to [0, 1] range

    return skeleton_channel
