from pytorch_lightning.callbacks import Callback
import os
import wandb
from utils.visualization import query_prediction_results_similarity_preprocessed, query_prediction_results_similarity
import pandas as pd


import os
import random
import wandb
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import pandas as pd
from utils.visualization import query_prediction_results_similarity


# class SimilarityVizCallback(Callback):
#     def __init__(self, config, outdir='results', log_every_n_epochs=10):
#         self.config = config
#         self.outdir = outdir
#         self.log_every_n_epochs = log_every_n_epochs

#     def on_validation_epoch_end(self, trainer, pl_module):
#         epoch = trainer.current_epoch
#         if epoch % self.log_every_n_epochs != 0:
#             return
#         if pl_module.distmat is None:
#             return  # or log a warning
        
#         root = self.config['dataset']

#         distmat = pl_module.distmat
#         query_metadata = pd.DataFrame([
#             {'path': path, 'identity': id_}
#             for paths, ids in zip(pl_module.query_path_epoch, pl_module.query_identity_epoch)
#             for path, id_ in zip(paths, ids)
#         ])

#         gallery_metadata = pd.DataFrame([
#             {'path': path, 'identity': id_}
#             for paths, ids in zip(pl_module.gallery_path_epoch, pl_module.gallery_identity_epoch)
#             for path, id_ in zip(paths, ids)
#         ])

#         # Generate and log the visualization
#         fig = query_prediction_results_similarity(
#             root=root,
#             query_metadata=query_metadata,
#             db_metadata=gallery_metadata,
#             query_start=0,
#             similarity_scores=-distmat,  # negate distance for similarity
#             num_images=10,
#             to_save=True)

#         if self.config.get('use_wandb', False):
#             wandb_img = wandb.Image(fig, caption=f"Val retrieval epoch {trainer.current_epoch + 1}")
#             pl_module.logger.experiment.log({"Val Retrieval": wandb_img})
#         else:
#             os.makedirs(self.outdir, exist_ok=True)
#             fig.savefig(os.path.join(self.outdir, f'val_retrieval_epoch{trainer.current_epoch + 1}_fig.png'))



# Replace
class SimilarityVizCallback(Callback):
    def __init__(self, config, outdir='results', log_every_n_epochs=10):
        self.config = config
        self.outdir = outdir
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return
        if pl_module.distmat is None:
            return  # or log a warning
        
        root = self.config['dataset']

        distmat = pl_module.distmat
        query_metadata = pd.DataFrame([
            {'path': path, 'identity': id_}
            for paths, ids in zip(pl_module.query_path_epoch, pl_module.query_identity_epoch)
            for path, id_ in zip(paths, ids)
        ])

        gallery_metadata = pd.DataFrame([
            {'path': path, 'identity': id_}
            for paths, ids in zip(pl_module.gallery_path_epoch, pl_module.gallery_identity_epoch)
            for path, id_ in zip(paths, ids)
        ])

        # Generate and log the visualization
        fig = query_prediction_results_similarity(
            root=root,
            query_metadata=query_metadata,
            db_metadata=gallery_metadata,
            query_start=0,
            similarity_scores=-distmat,  # negate distance for similarity
            num_images=10,
            to_save=True)

        if self.config.get('use_wandb', False):
            wandb_img = wandb.Image(fig, caption=f"Val retrieval epoch {trainer.current_epoch + 1}")
            pl_module.logger.experiment.log({"Val Retrieval": wandb_img})
        else:
            os.makedirs(self.outdir, exist_ok=True)
            fig.savefig(os.path.join(self.outdir, f'val_retrieval_epoch{trainer.current_epoch + 1}_fig.png'))
import os
import random
import wandb
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import pandas as pd
from utils.visualization import new_query_prediction_results_similarity

def load_image(path):
    return Image.open(path).convert('RGB')

class SimilarityVizCallback(Callback):
    def __init__(self, config, outdir='results', log_every_n_epochs=1):
        self.config = config
        self.outdir = outdir
        self.log_every_n_epochs = log_every_n_epochs
        self.batch_samples = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.batch_samples = []

    def validation_step(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0 and pl_module.incl_metadata:  # Query dataset
            img, path, identity = batch['img'], batch['path'], batch['identity']
            # Select random image from batch
            rand_idx = random.randint(0, len(img) - 1)
            self.batch_samples.append({
                'raw_img': img[rand_idx].clone(),  # Store raw tensor
                'preprocessed_img': img[rand_idx].clone(),  # Already preprocessed by dataset
                'path': path[rand_idx],
                'identity': identity[rand_idx]
            })

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0 or not self.batch_samples or pl_module.distmat is None:
            return

        root = self.config['dataset']
        distmat = pl_module.distmat

        # Create gallery metadata
        gallery_metadata = pd.DataFrame([
            {'path': path, 'identity': id_}
            for paths, ids in zip(pl_module.gallery_path_epoch, pl_module.gallery_identity_epoch)
            for path, id_ in zip(paths, ids)
        ])

        # Select random sample from batch_samples
        sample = random.choice(self.batch_samples)
        query_path = sample['path']
        query_identity = sample['identity']
        query_raw_img = sample['raw_img']
        query_preprocessed_img = sample['preprocessed_img']

        # Find query index in distmat
        query_idx = next(i for i, path in enumerate([p for paths in pl_module.query_path_epoch for p in paths]) if path == query_path)

        # Get predicted image
        closest_db_idx = np.argmax(-distmat[query_idx])  # Negate for similarity
        predicted_path = gallery_metadata.iloc[closest_db_idx]['path']
        predicted_identity = gallery_metadata.iloc[closest_db_idx]['identity']

        # Get preprocessed predicted image from dataset
        dataset_idx = next(i for i, path in enumerate(pl_module.trainer.datamodule.val_gallery_dataset.metadata['path']) if path == predicted_path)
        predicted_preprocessed_img = pl_module.trainer.datamodule.val_gallery_dataset[dataset_idx]['img']
        predicted_raw_img = load_image(os.path.join(root, predicted_path))

        # Denormalize for visualization
        denorm = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        query_preprocessed_img = denorm(query_preprocessed_img.cpu())
        query_preprocessed_img = torch.clamp(query_preprocessed_img, 0, 1)
        predicted_preprocessed_img = denorm(predicted_preprocessed_img.cpu())
        predicted_preprocessed_img = torch.clamp(predicted_preprocessed_img, 0, 1)

        # Convert to PIL for logging
        query_preprocessed_pil = T.ToPILImage()(query_preprocessed_img)
        predicted_preprocessed_pil = T.ToPILImage()(predicted_preprocessed_img)

        # Visualize
        fig = new_query_prediction_results_similarity(
            query_raw_img=load_image(os.path.join(root, query_path)),
            query_preprocessed_img=query_preprocessed_pil,
            predicted_raw_img=predicted_raw_img,
            predicted_preprocessed_img=predicted_preprocessed_pil,
            query_identity=query_identity,
            predicted_identity=predicted_identity,
            epoch=epoch + 1,
            to_save=True
        )

        # Log to WandB
        if self.config.get('use_wandb', False):
            wandb_img = wandb.Image(fig, caption=f"Val Image Comparison Epoch {epoch + 1}")
            pl_module.logger.experiment.log({"Val Image Comparison": wandb_img})
        else:
            os.makedirs(self.outdir, exist_ok=True)
            fig.savefig(os.path.join(self.outdir, f'val_image_comparison_epoch{epoch + 1}.png'))