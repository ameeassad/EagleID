from pytorch_lightning.callbacks import Callback
import torch
import os
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import wandb

from data.data_utils import unnormalize


class GradCAMCallback(Callback):
    def __init__(self, model, config, outdir='results', log_every_n_epochs=1):
        """
        Args:
            model: The model to use for GradCAM visualizations.
            config: Configuration dict, should include the `use_wandb` flag.
            outdir: Directory to save GradCAM images locally.
            log_every_n_epochs: Interval at which GradCAM images are logged.
        """
        self.model = model.backbone
        self.config = config
        self.outdir = outdir
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        # Check if validation dataloader exists
        if not trainer.val_dataloaders:
            print("Validation dataloaders not found or empty.")
            return  # Exit if no validation dataloader is available

        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:

            target_layer = self.get_resnet50_layer4(self.model)

            if target_layer is None:
                print("ResNet50 layer4 not found.")
                return
            # Proceed with GradCAM logic if dataloaders exist
            pl_module.eval()  # Ensure model is in evaluation mode
            for batch_idx, batch in enumerate(trainer.val_dataloaders):
                x, target = batch

                # Move inputs and targets to the device
                x = x.to(pl_module.device)
                target = target.to(pl_module.device)

                # GradCAM logic (same as before)
                with torch.enable_grad():
                    unnormalized_x = unnormalize(x[0].cpu(), self.config['transforms']['mean'], self.config['transforms']['std']).permute(1, 2, 0).numpy()
                    unnormalized_x = np.clip(unnormalized_x, 0, 1)

                    # cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
                    cam = GradCAM(model=self.model, target_layers=[target_layer])
                    targets = [ClassifierOutputTarget(class_idx) for class_idx in target]
                    grayscale_cam = cam(input_tensor=x, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(unnormalized_x, grayscale_cam, use_rgb=True)
                    img = Image.fromarray((visualization * 255).astype(np.uint8))

                    # Log GradCAM image to W&B if enabled
                    if self.config.get('use_wandb', False):
                        wandb_img = wandb.Image(visualization, caption=f"GradCAM Epoch {trainer.current_epoch + 1} Batch {batch_idx} Image 0")
                        pl_module.logger.experiment.log({"GradCAM Images": wandb_img})

                    # Save locally
                    os.makedirs(self.outdir, exist_ok=True)
                    img.save(os.path.join(self.outdir, f'cam_image_val_epoch{trainer.current_epoch + 1}_batch{batch_idx}_img0.png'))

                # Limit the number of batches for GradCAM to avoid excessive logs
                if batch_idx >= 2:
                    break

            pl_module.train()  # Set the model back to training mode

    def get_resnet50_layer4(self, model):
        """
        Retrieve the layer4 of a ResNet50 model.
        """
        if hasattr(model, 'layer4'):
            return model.layer4[-1]  # Use the last block of layer4 for GradCAM
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
            # In case the backbone is wrapped inside another model
            return model.backbone.layer4[-1]
        elif hasattr(model, 'model') and hasattr(model.model, 'layer4'):
            # If wrapped inside a SimpleModel or another object
            return model.model.layer4[-1]
        else:
            print("layer4 not found in the model.")
            return None