import pandas as pd
from wildlife_datasets import datasets, splits
from torch.utils.data import Dataset
from wildlife_tools.data.dataset import WildlifeDataset
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


import cv2
import re, os

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

    def __init__(self, root: str, df = None):
        self.root = root
        if df is None:
            self.df = self.create_catalogue()
        else:
            self.df = df.copy()

        super().__init__(root=root, df=self.df)

    def create_catalogue(self) -> pd.DataFrame:
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

                # Assign a unique identity_id if it's not already assigned
                if animal_id not in identity_map:
                    identity_map[animal_id] = identity_counter
                    identity_counter += 1
                
                if not os.path.isdir(animal_path):
                    continue  # Skip non-folder entries
                
                # Check for either images directly or video-based screenshots
                for sub_entry in os.listdir(animal_path):
                    sub_path = os.path.join(animal_path, sub_entry)
                    
                    if os.path.isdir(sub_path):
                        # It's a video folder; get screenshots inside the folder
                        for screenshot in os.listdir(sub_path):
                            if screenshot.endswith(('.jpg', '.jpeg', '.png')):
                                date = self.extract_date_from_filename(screenshot)
                                data.append({
                                    'image_id': int(image_counter),
                                    'species': species_name,
                                    'identity_id': identity_map[animal_id],
                                    'identity': animal_id,
                                    'path': os.path.join(sub_path, screenshot),
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
                            'path': sub_path,
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

    Args:
        metadata (pd.DataFrame): Dataframe containing image metadata.
        root (str): Root directory for images.
        transform (callable, optional): Transform to be applied to the images.
        img_load (str, optional): How to load images ('full', 'bbox', etc.).
        col_path (str, optional): Column in metadata with image paths.
        col_label (str, optional): Column in metadata with class labels.
        load_label (bool, optional): Whether to load labels or not.
    """

    def __init__(
        self,
        metadata: pd.DataFrame | None = None,
        root: str | None = None,
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
        self.metadata = metadata
        
    # def get_image(self, path):
    #     """
    #     Custom image loader, customized based on preprocessing level for raptor images.
    #     """
    #     img = cv2.imread(path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = Image.fromarray(img)
    #     return img
    
    def wildlife_dataset(self) -> Raptors:
        return self.raptor_dataset
    
    def get_df(self) -> pd.DataFrame:
        return self.metadata
    
class WildlifeReidDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, metadata, preprocess_lvl=0, batch_size=8, size=256, mean=0.5, std=0.5, num_workers=2, config = None):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.preprocess_lvl = preprocess_lvl
        self.batch_size = batch_size
        self.size = size
        self.mean = (mean, mean, mean) if isinstance(mean, float) else tuple(mean)
        self.std = (std, std, std) if isinstance(std, float) else tuple(std)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        splitter = splits.ClosedSetSplit(0.8)
        for idx_train, idx_test in splitter.split(metadata):
            splits.analyze_split(metadata, idx_train, idx_test)

        df_train, df_test = metadata.loc[idx_train], metadata.loc[idx_test]
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)


        df_query, df_gallery = self.split_query_database(df_test)

        self.train_dataset = RaptorsWildlife(metadata=df_train, root=data_dir, transform=transform)
        self.val_query_dataset = RaptorsWildlife(metadata=df_query, root=data_dir, transform=transform)
        self.val_gallery_dataset = RaptorsWildlife(metadata=df_gallery, root=data_dir, transform=transform)
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
                # Skip identities with less than 2 images, avoid data leakage
                continue

        query_metadata = test_metadata.loc[query_indices].reset_index(drop=True)
        database_metadata = test_metadata.loc[database_indices].reset_index(drop=True)

        return query_metadata, database_metadata


class GoldensWildlife(RaptorsWildlife):
    def __init__(
        self,
        metadata: pd.DataFrame | None = None,
        root: str | None = None,
        transform: callable = None,
        **kwargs
    ):
        if metadata is None:
            raptor_dataset = Raptors(root)
            metadata = raptor_dataset.df[raptor_dataset.df['species'] == 'goleag']
        else:
            metadata = metadata[metadata['species'] == 'goleag']

        super().__init__(metadata=metadata, root=root, transform=transform, **kwargs)
        # if metadata is None:
        #     raptor_dataset = Raptors(root)
        #     metadata = raptor_dataset.df['species'] == 'goleag'
