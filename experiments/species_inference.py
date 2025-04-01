import sys
import torchvision.transforms as T

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.inference import KnnClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import itertools
from torch.optim import SGD
from utils.trainer_pl import basic_trainer_pl
from models.template_model import TemplateModel
from utils.triplet_loss_utils import TripletLoss_wildlife

from utils.triplet_loss_utils import KnnClassifier
from wildlife_tools.similarity import CosineSimilarity
from sklearn.metrics import precision_score, recall_score, f1_score

import timm
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning import Trainer
import numpy as np
from PIL import Image
import wandb
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from wildlife_datasets import analysis, datasets, loader

root = '/Users/amee/Documents/code/master-thesis/datasets/ATRW/'

# Load dataset metadata
metadata = datasets.ATRW(root)
transform = T.Compose([T.Resize([224, 224]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
dataset = WildlifeDataset(metadata.df, metadata.root, transform=transform)



# Download MegaDescriptor-T backbone from HuggingFace Hub
backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)

# Arcface loss - needs backbone output size and number of classes.
objective = TripletLoss_wildlife()

# Optimize parameters in backbone and in objective using single optimizer.
params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)

def print_epoch_loss(trainer, epoch_data):
    # This function will print the average loss at the end of each epoch
    print(f"Epoch {trainer.epoch}: Average Loss = {epoch_data['train_loss_epoch_avg']}")


trainer = basic_trainer_pl(
    dataset=dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=0,
    device='cpu',
    epoch_callback=print_epoch_loss
)

from wildlife_tools.features import DeepFeatures


dataset_database_P = WildlifeDataset(metadata.df.iloc[100:,:], metadata.root, transform=transform)
dataset_query_P = WildlifeDataset(metadata.df.iloc[:100,:], metadata.root, transform=transform)

# name = 'hf-hub:BVRA/MegaDescriptor-T-224'
extractor_P = DeepFeatures(backbone , device = 'cpu')

query_P, database_P = extractor_P(dataset_query_P), extractor_P(dataset_database_P)

similarity_function = CosineSimilarity()
similarity_P = similarity_function(query_P, database_P)
print(similarity_P)
classifier_P = KnnClassifier(k=1, database_labels=dataset_database_P.labels_string)
predictions_P = classifier_P(similarity_P['cosine'])
print("Predictions for 100 test Images:-\n",predictions_P)
accuracy_P = np.mean(dataset_query_P.labels_string == predictions_P)
print("Accuracy on ATRW data: {:.2f}%".format(accuracy_P * 100))

precision_P = precision_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
recall_P = recall_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
f1_P = f1_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
print("Precision:", precision_P)
print("Recall:", recall_P)
print("F1 Score:", f1_P)







def validate_species(model, species_name, query_metadata, db_metadata, root, transform):
    # Filter metadata for target species
    query_filtered = query_metadata[query_metadata['species'] == species_name].copy()
    db_filtered = db_metadata[db_metadata['species'] == species_name].copy()
    
    # Create datasets
    query_dataset = WildlifeDataset(query_filtered, root, transform=transform)
    db_dataset = WildlifeDataset(db_filtered, root, transform=transform)
    
    # Create extractor (assuming DeepFeatures works with your model)
    extractor = DeepFeatures(model.backbone, device='cpu')  # Use model's backbone
    
    # Extract features
    query_features = extractor(query_dataset)
    db_features = extractor(db_dataset)
    
    # Compute similarity
    similarity = CosineSimilarity()(query_features, db_features)
    
    # Get predictions
    classifier = KnnClassifier(k=1, database_labels=db_dataset.labels_string)
    predictions = classifier(similarity['cosine'])
    
    # Calculate metrics
    accuracy = np.mean(query_dataset.labels_string == predictions)
    precision = precision_score(query_dataset.labels_string, predictions, average='weighted', zero_division=1)
    recall = recall_score(query_dataset.labels_string, predictions, average='weighted', zero_division=1)
    f1 = f1_score(query_dataset.labels_string, predictions, average='weighted', zero_division=1)
    
    # Visualization
    query_prediction_results(
        root=root,
        query_metadata=query_filtered.reset_index(drop=True),
        db_metadata=db_filtered.reset_index(drop=True),
        query_start=0,
        predictions=predictions,
        num_images=min(10, len(query_filtered))
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Usage
species_results = validate_species(
    model=model,
    species_name="goleag",
    query_metadata=metadata.df.iloc[:100],  # Your query metadata
    db_metadata=metadata.df.iloc[100:],     # Your database metadata
    root=metadata.root,
    transform=transform
)
