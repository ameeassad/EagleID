from pytorch_lightning.callbacks import Callback
import os
import wandb
from utils.visualization import query_prediction_results_similarity_preprocessed


class SimilarityVizCallback(Callback):
    def __init__(self, config, outdir='results', log_every_n_epochs=10):
        self.config = config
        self.outdir = outdir
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return

        distmat = pl_module.distmat.detach().cpu().numpy()
        query_metadata = pl_module.query_metadata_epoch
        gallery_metadata = pl_module.gallery_metadata_epoch
        root = self.config['root']

        # Generate and log the visualization
        fig = query_prediction_results_similarity_preprocessed(
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
