import torch
from preprocess.mmpose_fill import get_keypoints_info
import pandas as pd
from wildlife_tools.data.split import Split

from torch.utils.data import Sampler
import numpy as np

from torch.utils.data.sampler import Sampler
import numpy as np
import random

class RandomIdentitySampler(Sampler):
    """
    Needed for Triplet Mining
    """
    def __init__(self, dataset, batch_size, num_instances=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        # Get all unique labels (animal IDs)
        self.labels = self.dataset.labels  # Assume dataset.labels exists
        
        # Create dictionary: {label: [indices]}
        self.label_to_indices = dict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        
        # Filter out classes with <2 instances
        self.valid_labels = [
            lbl for lbl in self.label_to_indices 
            if len(self.label_to_indices[lbl]) >= self.num_instances
        ]

    def __iter__(self):
        batch_indices = []
        
        # Shuffle valid labels
        shuffled_labels = np.random.permutation(self.valid_labels)
        
        for label in shuffled_labels:
            # Randomly sample 2 instances for this class
            instances = random.sample(self.label_to_indices[label], self.num_instances)
            batch_indices.extend(instances)
            
            # Yield a batch when we have enough samples
            if len(batch_indices) >= self.batch_size:
                yield batch_indices[:self.batch_size]
                batch_indices = batch_indices[self.batch_size:]

    def __len__(self):
        return (len(self.valid_labels) * self.num_instances) // self.batch_size

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
    

def calculate_num_channels(preprocess_lvl):
    img_channels = 3 
    if preprocess_lvl == 0:
        method = 'full'
    if preprocess_lvl == 1:
        method = 'bbox'
    elif preprocess_lvl == 2: 
        method = "bbox_mask"
    elif preprocess_lvl == 3: 
        method = "bbox_mask_skeleton"
        img_channels = 4
    elif preprocess_lvl == 4:
        method = "bbox_mask_components"
        img_channels = 3 + 3 * 5
    elif preprocess_lvl == 5:
        method = "bbox_mask_heatmaps"
        img_channels = 3 + len(get_keypoints_info())
    
    return img_channels


def create_df_from_coco(coco_obj):
    # Extract image info and create a dictionary to map image IDs to filenames, height, and width
    images_info = coco_obj.dataset['images']
    image_id_to_info = {image['id']: {'file_name': image['file_name'], 'height': image['height'], 'width': image['width']} for image in images_info}

    # Extract category info and create a dictionary to map category IDs to category names
    categories_info = coco_obj.dataset['categories']
    category_id_to_name = {category['id']: category['name'] for category in categories_info}

    # Extract annotations info and prepare data for the DataFrame
    annotations_info = coco_obj.dataset['annotations']
    annotations_data = []

    for annotation in annotations_info:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        image_info = image_id_to_info[image_id]
        
        annotation_data = {
            'image_id': annotation['image_id'],
            'image_filename': image_info['file_name'],
            'height': image_info['height'],        # Save height
            'width': image_info['width'],          # Save width
            'category_id': annotation['category_id'],
            'category_name': category_id_to_name[category_id],
            'bbox': annotation['bbox'],
            'area': annotation['area'],
            'iscrowd': annotation['iscrowd'],
            'segmentation': annotation['segmentation']
        }
        annotations_data.append(annotation_data)

    # Create the DataFrame
    df = pd.DataFrame(annotations_data)

    return df

class SplitQueryDatabase(Split):
    """
    Splits metadata into query and database sets by adding a 'query' column.
    Each identity is present in both query and database sets without data leakage.
    The 'query' column is set to 1 if the image is part of the query set; otherwise, 0.
    """

    def __call__(self, metadata):
        # Initialize 'query' column to 0 (default: part of database)
        metadata = metadata.copy()
        metadata['query'] = 0

        identities = metadata['identity'].unique()

        for identity in identities:
            identity_indices = metadata[metadata['identity'] == identity].index.tolist()
            if len(identity_indices) > 1:
                # Set the first image as query (value 1)
                metadata.at[identity_indices[0], 'query'] = 1
                # Remaining images stay as database (value 0)
            else:
                # If only one image, it remains in the database
                continue

        return metadata
    
def recognize_id_split(ids_train, ids_test):
    ids_train = set(ids_train)
    ids_test = set(ids_test)
    
    ids_test_only = ids_test - ids_train
    ids_joint = ids_train.intersection(ids_test)
    id_split = 'closed-set'
    if len(ids_joint) == 0:
        id_split = 'disjoint-set'
    elif len(ids_test_only) > 0:
        id_split = 'open-set'
    return id_split


def analyze_split(df, idx_train, idx_test):
    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test]
    
    ids = set(df['identity'])
    ids_train = set(df_train['identity'])
    ids_test = set(df_test['identity'])
    ids_train_only = ids_train - ids_test
    ids_test_only = ids_test - ids_train
    ids_joint = ids_train.intersection(ids_test)
    
    n = len(idx_train)+len(idx_test)
    n_train = len(idx_train)
    n_test_only = sum([sum(df_test['identity'] == ids) for ids in ids_test_only])    
    
    ratio_train = n_train / n    
    ratio_test_only = n_test_only / n
    
    id_split = recognize_id_split(ids_train, ids_test)
            
    print('Split: %s' % (id_split))
    print('Samples: train/test/unassigned/total = %d/%d/%d/%d' % (len(df_train), len(df_test), len(df)-len(df_train)-len(df_test), len(df)))
    print('Classes: train/test/unassigned/total = %d/%d/%d/%d' % (len(ids_train), len(ids_test), len(ids)-len(ids_train)-len(ids_test)+len(ids_train.intersection(ids_test)), len(ids)))
    print('Classes: train only/test only/joint  = %d/%d/%d' % (len(ids_train_only), len(ids_test_only),  len(ids_joint)))
    print('')    
    print('Fraction of train set     = %1.2f%%' % (100*ratio_train))
    print('Fraction of test set only = %1.2f%%' % (100*ratio_test_only))