from pytorch_lightning.callbacks import Callback
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import ConfusionMatrix
import wandb


class ConfusionMatrixCallback(Callback):
    def __init__(self, config, outdir='results', log_every_n_epochs=1, num_classes=None):
        """
        Confusion Matrix callback for classification models.
        
        Args:
            config: Configuration dict, should include the `use_wandb` flag.
            outdir: Directory to save confusion matrix images locally.
            log_every_n_epochs: Interval at which confusion matrices are logged.
            num_classes: Number of classes for the confusion matrix.
        """
        self.config = config
        self.outdir = outdir
        self.log_every_n_epochs = log_every_n_epochs
        self.num_classes = num_classes
        self.confusion_matrix = None

    def on_validation_epoch_end(self, trainer, pl_module):
        # Check if the current epoch is a multiple of log_every_n_epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return  # Skip confusion matrix computation

        # Check if validation dataloader exists
        if not trainer.val_dataloaders:
            print("Validation dataloaders not found or empty.")
            return

        # Get number of classes from the model if not provided
        if self.num_classes is None:
            if hasattr(pl_module, 'num_classes'):
                self.num_classes = pl_module.num_classes
            else:
                print("Could not determine number of classes for confusion matrix.")
                return

        # Initialize confusion matrix metric
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes).to(pl_module.device)

        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        # Collect predictions and targets
        pl_module.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Handle both dict and tuple batches
                if isinstance(batch, dict):
                    x = batch['img']
                    target = batch['label']
                else:
                    x, target = batch

                x = x.to(pl_module.device)
                target = target.to(pl_module.device)

                # Get predictions
                logits = pl_module(x)
                
                # Handle different model types
                if hasattr(pl_module, 'logits_to_pred'):
                    # For AgeModel with CORAL
                    pred = pl_module.logits_to_pred(logits)
                else:
                    # For standard classification models
                    pred = torch.argmax(logits, dim=1)

                all_predictions.append(pred.cpu())
                all_targets.append(target.cpu())

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Compute confusion matrix
        self.confusion_matrix.update(all_predictions, all_targets)
        confusion_matrix = self.confusion_matrix.compute()

        # Create confusion matrix visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix.cpu().numpy(), 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=[str(i) for i in range(self.num_classes)],
                   yticklabels=[str(i) for i in range(self.num_classes)])
        plt.title(f'Confusion Matrix - Epoch {trainer.current_epoch + 1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        # Log confusion matrix to W&B if enabled
        if self.config.get('use_wandb', False) and pl_module.logger:
            wandb_img = wandb.Image(plt, caption=f"Confusion Matrix Epoch {trainer.current_epoch + 1}")
            pl_module.logger.experiment.log({"Confusion Matrix": wandb_img})

        # Save locally
        os.makedirs(self.outdir, exist_ok=True)
        plt.savefig(os.path.join(self.outdir, f'confusion_matrix_epoch{trainer.current_epoch + 1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Reset confusion matrix
        self.confusion_matrix.reset()

        pl_module.train()  # Set the model back to training mode 