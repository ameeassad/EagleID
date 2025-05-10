import torch
from preprocess.mmpose_fill import get_keypoints_info
import pandas as pd
from wildlife_tools.data.split import Split
from wildlife_datasets import splits
from torch.utils.data import Sampler
import numpy as np
import random

class PaddedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data_source)

    def __iter__(self):
        # Create list of indices (same as default DataLoader)
        indices = list(range(self.n_samples))
        if self.shuffle:
            random.shuffle(indices)  # Controlled by seed_everything
        
        # Generate batches (same as default DataLoader)
        batches = []
        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batches.append(batch_indices)
        
        # Check the last batch
        if batches and len(batches[-1]) == 1:
            # Pad last batch with 1 random sample to reach 2
            batches[-1].append(random.choice(indices))
            print(f"Padded last batch from 1 to 2 samples")
        
        # Yield batches
        for batch_indices in batches:
            yield batch_indices

    def __len__(self):
        # Number of batches, rounding up
        return (self.n_samples + self.batch_size - 1) // self.batch_size

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
            # identity_indices = metadata[metadata['identity'] == identity].index.tolist()
            identity_df = metadata[metadata['identity'] == identity].sort_values(by=['image_id'])  # Sort by image_id
            identity_indices = identity_df.index.tolist()
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


class CustomClosedSetSplit(splits.ClosedSetSplit):
    """Closed-set split with strict ratio adherence via adjustable samples."""

    def __init__(
        self,
        ratio_train: float,
        seed: int = 666,
        identity_skip: str = 'unknown',
        tolerance: float = 0.05
    ) -> None:
        super().__init__(
            ratio_train=ratio_train,
            seed=seed,
            identity_skip=identity_skip
        )
        self.tolerance = tolerance

    def general_split(
            self,
            df: pd.DataFrame,
            individual_train: list[str],
            individual_test: list[str],
        ) -> tuple[np.ndarray, np.ndarray]:
        
        lcg = self.initialize_lcg()
        idx_train, idx_test = [], []
        adjustable_train, adjustable_test = [], []

        # First pass: Initial split per identity
        for individual, df_individual in df.groupby('identity'):
            n = len(df_individual)
            shuffled_indices = df_individual.index[lcg.random_permutation(n)].tolist()

            if n == 1:
                idx_test.extend(shuffled_indices) # cannot be used for triplet mining
            elif n == 2:
                idx_train.extend(shuffled_indices)
            elif n == 3:
                idx_train.extend(shuffled_indices[:2])
                test_idx = shuffled_indices[2:]
                idx_test.extend(test_idx)
                adjustable_test.extend(test_idx)
            elif n == 4:
                train_part = shuffled_indices[:2]
                test_part = shuffled_indices[2:]
                idx_train.extend(train_part)
                idx_test.extend(test_part)
            else:  # n > 4
                train_fixed = shuffled_indices[:2]
                test_fixed = shuffled_indices[2:4]
                remaining = shuffled_indices[4:]
                n_remaining_train = int(np.round(self.ratio_train * len(remaining)))
                train_remaining = remaining[:n_remaining_train]
                test_remaining = remaining[n_remaining_train:]
                idx_train.extend(train_fixed + train_remaining)
                idx_test.extend(test_fixed + test_remaining)
                adjustable_train.extend(train_remaining)
                adjustable_test.extend(test_remaining)

        # Second pass: Adjust to meet ratio within tolerance
        total = len(idx_train) + len(idx_test)
        desired = self.ratio_train
        current_ratio = len(idx_train) / total if total > 0 else 0.0

        while abs(current_ratio - desired) > self.tolerance:
            if current_ratio < desired - self.tolerance:
                if not adjustable_test:
                    break
                permuted_indices = lcg.random_permutation(len(adjustable_test))
                move_idx = permuted_indices[0]

                sample = adjustable_test.pop(move_idx)
                idx_test.remove(sample)
                idx_train.append(sample)
                adjustable_train.append(sample)
            elif current_ratio > desired + self.tolerance:
                if not adjustable_train:
                    break
                move_idx = lcg.choice(len(adjustable_train))
                sample = adjustable_train.pop(move_idx)
                idx_train.remove(sample)
                idx_test.append(sample)
                adjustable_test.append(sample)
            else:
                break
            current_ratio = len(idx_train) / (len(idx_train) + len(idx_test))

        # Feedback on final ratio
        if abs(current_ratio - desired) > self.tolerance:
            print(f"Closest Achieved Ratio: {current_ratio:.2%} (Target: {desired:.0%} ±{self.tolerance:.0%})")

        return np.array(idx_train), np.array(idx_test)
    
class StratifiedSpeciesSplit(CustomClosedSetSplit):
    """Stratified split that maintains the train-test ratio per species using custom closed-set splitting.
    
    For each species group, the group's index is reset so that the custom split returns local indices.
    Then these local indices are mapped back to global indices.
    """

    def split(self, df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        # Group data by species (wildlife_name)
        grouped = df.groupby('wildlife_name', group_keys=False)
        
        idx_train_list, idx_test_list = [], []
        for species, group in grouped:
            # Reset index to get a local index column, but save the original global index.
            group_reset = group.reset_index()  # The original global index is in the 'index' column.
            
            # Call custom closed-set split (from CustomClosedSetSplit) on the group_reset.
            group_splits = super().split(group_reset)
            if not group_splits:
                continue
            group_idx_train, group_idx_test = group_splits[0]
            
            # Map local indices (from group_reset) to global indices using the 'index' column.
            global_train_indices = group_reset.loc[group_idx_train, "index"].to_numpy()
            global_test_indices = group_reset.loc[group_idx_test, "index"].to_numpy()
            
            idx_train_list.extend(global_train_indices)
            idx_test_list.extend(global_test_indices)
        
        return [(np.array(idx_train_list), np.array(idx_test_list))]


# class StratifiedSpeciesSplit(splits.ClosedSetSplit):
#     """Stratified split that maintains the train-test ratio per species."""

#     def split(self, df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
#         """Split the dataset such that each species maintains the specified train-test ratio.
        
#         For each species group, the group's index is reset so that the split method
#         returns local indices. Then, these local indices are mapped back to global indices.
#         """
#         # Group data by species (wildlife_name)
#         grouped = df.groupby('wildlife_name', group_keys=False)
        
#         idx_train_list, idx_test_list = [], []
#         for species, group in grouped:
#             # Reset index to get a local index column, but save the original global index.
#             group_reset = group.reset_index()  # The original global index is in the 'index' column.
            
#             # Apply the closed-set split on the group_reset DataFrame.
#             # This should now return indices relative to the group_reset.
#             group_splits = super().split(group_reset)
#             if not group_splits:
#                 continue
#             group_idx_train, group_idx_test = group_splits[0]
            
#             # Map local indices (from group_reset) to global indices using the 'index' column.
#             global_train_indices = group_reset.loc[group_idx_train, "index"].to_numpy()
#             global_test_indices = group_reset.loc[group_idx_test, "index"].to_numpy()
            
#             idx_train_list.extend(global_train_indices)
#             idx_test_list.extend(global_test_indices)
        
#         return [(np.array(idx_train_list), np.array(idx_test_list))]