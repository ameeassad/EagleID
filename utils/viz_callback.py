from pytorch_lightning.callbacks import Callback
import os
import wandb
from utils.visualization import query_prediction_results_similarity_preprocessed, query_prediction_results_similarity
import pandas as pd


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
            {'path': path, 'identity': id}
            for paths, ids in zip(pl_module.query_path_epoch, pl_module.query_identity_epoch)
            for path, id in zip(paths, ids)
        ])

        gallery_metadata = pd.DataFrame([
            {'path': path, 'identity': id}
            for paths, ids in zip(pl_module.gallery_path_epoch, pl_module.gallery_identity_epoch)
            for path, id in zip(paths, ids)
        ])

        # Generate and log the visualization
        fig = query_prediction_results_similarity(
            root=root,
            query_metadata=query_metadata,
            db_metadata=gallery_metadata,
            query_start=0,
            similarity_scores=-distmat,  # negate distance for similarity
            num_images=10,
            preprocess_option=self.config['preprocess_lvl']
        )

        if self.config.get('use_wandb', False):
            wandb_img = wandb.Image(fig, caption=f"Val retrieval epoch {trainer.current_epoch + 1}")
            pl_module.logger.experiment.log({"Val Retrieval": wandb_img})
        else:
            os.makedirs(self.outdir, exist_ok=True)
            fig.save(os.path.join(self.outdir, f'val_retrieval_epoch{trainer.current_epoch + 1}_fig.png'))  
