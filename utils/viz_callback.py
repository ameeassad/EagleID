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

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = trainer.val_dataloaders[0]
        for batch in val_loader:
            img, path, identity = batch['img'], batch['path'], batch['identity']
            rand_idx = random.randint(0, len(img) - 1)
            self.batch_samples.append({
                'preprocessed_img': img[rand_idx].clone(),
                'path': path[rand_idx],
                'identity': identity[rand_idx]
            })
            break  # just one batch is enough

        epoch = trainer.current_epoch
        print(f"Epoch {epoch}: on_validation_epoch_end called, batch_samples: {len(self.batch_samples)}")
        if epoch % self.log_every_n_epochs != 0 or not self.batch_samples or pl_module.distmat is None:
            print(f"Epoch {epoch}: Skipping visualization (log_every_n_epochs={self.log_every_n_epochs}, batch_samples={len(self.batch_samples)}, distmat_exists={pl_module.distmat is not None})")
            return

        root = self.config['dataset']
        distmat = pl_module.distmat
        preprocess_lvl = self.config.get('preprocess_lvl', 2)  # Default to 2 (mask)
        print(f"Epoch {epoch}: distmat shape: {distmat.shape}, preprocess_lvl: {preprocess_lvl}")

        gallery_metadata = pd.DataFrame([
            {'path': path, 'identity': id_}
            for paths, ids in zip(pl_module.gallery_path_epoch, pl_module.gallery_identity_epoch)
            for path, id_ in zip(paths, ids)
        ])
        print(f"Epoch {epoch}: gallery_metadata size: {len(gallery_metadata)}")

        try:
            sample = random.choice(self.batch_samples)
            query_path = sample['path']
            query_identity = sample['identity']
            query_preprocessed_img = sample['preprocessed_img']
            print(f"Epoch {epoch}: Selected sample, query_path: {query_path}, query_identity: {query_identity}, channels: {query_preprocessed_img.shape[0]}")

            query_idx = next(i for i, path in enumerate([p for paths in pl_module.query_path_epoch for p in paths]) if path == query_path)
            closest_db_idx = np.argmax(-distmat[query_idx])
            predicted_path = gallery_metadata.iloc[closest_db_idx]['path']
            predicted_identity = gallery_metadata.iloc[closest_db_idx]['identity']
            print(f"Epoch {epoch}: Predicted path: {predicted_path}, identity: {predicted_identity}")

            dataset_idx = next(i for i, path in enumerate(pl_module.trainer.datamodule.val_gallery_dataset.metadata['path']) if path == predicted_path)
            predicted_preprocessed_img = pl_module.trainer.datamodule.val_gallery_dataset[dataset_idx]['img']
            predicted_raw_img = load_image(os.path.join(root, predicted_path))
            print(f"Epoch {epoch}: Predicted image channels: {predicted_preprocessed_img.shape[0]}")

            # Handle multi-channel images
            query_preprocessed_img = query_preprocessed_img.cpu()
            predicted_preprocessed_img = predicted_preprocessed_img.cpu()
            num_channels = query_preprocessed_img.shape[0]
            if num_channels != predicted_preprocessed_img.shape[0]:
                print(f"Epoch {epoch}: Channel mismatch, query: {query_preprocessed_img.shape[0]}, predicted: {predicted_preprocessed_img.shape[0]}")
                return

            # Denormalize only RGB channels (first 3 channels)
            rgb_mean = self.config['transforms']['mean']
            rgb_std = self.config['transforms']['std']
            denorm_rgb = T.Normalize(mean=[-m/s for m, s in zip(rgb_mean, rgb_std)], std=[1/s for s in rgb_std])

            query_rgb = query_preprocessed_img[:3]
            query_rgb = denorm_rgb(query_rgb)
            query_rgb = torch.clamp(query_rgb, 0, 1)
            predicted_rgb = predicted_preprocessed_img[:3]
            predicted_rgb = denorm_rgb(predicted_rgb)
            predicted_rgb = torch.clamp(predicted_rgb, 0, 1)

            # Prepare task-specific channels based on preprocess_lvl
            query_task = None
            predicted_task = None
            if preprocess_lvl >= 3 and num_channels > 3:
                if preprocess_lvl == 3:
                    # Level 3: Single skeleton channel
                    query_task = query_preprocessed_img[3:4]  # Keep as (1, H, W)
                    predicted_task = predicted_preprocessed_img[3:4]
                    print(f"Epoch {epoch}: Level 3 skeleton, query_task shape: {query_task.shape}, min: {query_task.min()}, max: {query_task.max()}")
                elif preprocess_lvl == 4:
                    # Level 4: All component channels (3 * num_components)
                    query_task = query_preprocessed_img[3:]  # All channels after RGB
                    predicted_task = predicted_preprocessed_img[3:]
                    print(f"Epoch {epoch}: Level 4 components, query_task shape: {query_task.shape}, min: {query_task.min()}, max: {query_task.max()}")
                elif preprocess_lvl == 5:
                    # Level 5: All heatmap channels, normalize to [0, 1]
                    query_task = query_preprocessed_img[3:]  # All channels after RGB
                    predicted_task = predicted_preprocessed_img[3:]
                    # Normalize heatmaps to ensure visibility
                    for i in range(query_task.shape[0]):
                        if query_task[i].max() > 0:
                            query_task[i] = query_task[i] / query_task[i].max()
                        if predicted_task[i].max() > 0:
                            predicted_task[i] = predicted_task[i] / predicted_task[i].max()
                    print(f"Epoch {epoch}: Level 5 heatmaps, query_task shape: {query_task.shape}, min: {query_task.min()}, max: {query_task.max()}")
                    print(f"Epoch {epoch}: Level 5 heatmaps, predicted_task shape: {predicted_task.shape}, min: {predicted_task.min()}, max: {predicted_task.max()}")

            # Convert RGB to PIL images
            query_rgb_pil = T.ToPILImage()(query_rgb)
            predicted_rgb_pil = T.ToPILImage()(predicted_rgb)

            # Visualize
            fig = new_query_prediction_results_similarity(
                query_raw_img=load_image(os.path.join(root, query_path)),
                query_rgb_img=query_rgb_pil,
                query_task_img=query_task,
                predicted_raw_img=predicted_raw_img,
                predicted_rgb_img=predicted_rgb_pil,
                predicted_task_img=predicted_task,
                query_identity=query_identity,
                predicted_identity=predicted_identity,
                epoch=epoch + 1,
                preprocess_lvl=preprocess_lvl,
                to_save=True
            )
            if fig is None:
                print(f"Epoch {epoch}: Visualization failed, figure is None")
                return
            print(f"Epoch {epoch}: Visualization figure created")

            # Log to WandB
            if self.config.get('use_wandb', False):
                wandb_img = wandb.Image(fig, caption=f"Val Image Comparison Epoch {epoch + 1}")
                pl_module.logger.experiment.log({"Val Image Comparison": wandb_img})
                print(f"Epoch {epoch}: Logged to WandB")
            else:
                os.makedirs(self.outdir, exist_ok=True)
                fig.savefig(os.path.join(self.outdir, f'val_image_comparison_epoch{epoch + 1}.png'))
                print(f"Epoch {epoch}: Saved to {self.outdir}/val_image_comparison_epoch{epoch + 1}.png")
        except Exception as e:
            print(f"Epoch {epoch}: Error in visualization: {e}")