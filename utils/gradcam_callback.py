from pytorch_lightning.callbacks import Callback
import torch
import os
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import wandb

from data.transforms import denormalize


class GradCAMCallback(Callback):
    def __init__(self, model, config, outdir='results', log_every_n_epochs=10):
        """
        Args:
            model: The model to use for GradCAM visualizations.
            config: Configuration dict, should include the `use_wandb` flag.
            outdir: Directory to save GradCAM images locally.
            log_every_n_epochs: Interval at which GradCAM images are logged.
        """
        self.model = model
        self.config = config
        self.outdir = outdir
        self.log_every_n_epochs = log_every_n_epochs
        self.kp_included = config['preprocess_lvl'] >= 3 and config['model_architecture'] == 'FusionModel'

    def on_validation_epoch_end(self, trainer, pl_module):

        # Check if the current epoch is a multiple of log_every_n_epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return  # Skip GradCAM computation

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
            for batch_idx, batch in enumerate(trainer.val_dataloaders[0]): # first dataloader is the query images & labels
                x, target, *rest = batch

                # Move inputs and targets to the device
                x = x.to(pl_module.device)
                target = target.to(pl_module.device)

                if self.kp_included:
                    x = x[:, :3, :, :] # only take rgb channels
                    # TODO: Implement GradCAM for higher dimension inputs

                # GradCAM logic (same as before)
                with torch.enable_grad():
                    x_np = x[0].cpu().numpy()  # Convert from PyTorch tensor to numpy array
                    unnormalized_x = denormalize(x_np, self.config['transforms']['mean'], self.config['transforms']['std'])

                    # Reformat unnormalized_x back to (224, 224, 3) for visualization
                    unnormalized_x = np.transpose(unnormalized_x, (1, 2, 0))  # Shape is now (224, 224, 3)
                    # x_debug = unnormalized_x #debug
                    unnormalized_x = (unnormalized_x-np.min(unnormalized_x))/(np.max(unnormalized_x)-np.min(unnormalized_x))
                    unnormalized_x = np.clip(unnormalized_x, 0, 1)

                    # cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
                    cam = GradCAM(model=self.model, target_layers=[target_layer])
                    targets = [ClassifierOutputTarget(class_idx) for class_idx in target]
                    grayscale_cam = cam(input_tensor=x, targets=targets)[0, :]
                    grayscale_cam = np.repeat(grayscale_cam[:, :, np.newaxis], 3, axis=2) # make it compatible with x - 3 channels
                    visualization = show_cam_on_image(unnormalized_x, grayscale_cam, use_rgb=True)
                    img = Image.fromarray((visualization * 255).astype(np.uint8))

                    # Log GradCAM image to W&B if enabled
                    if self.config.get('use_wandb', False):
                        wandb_img = wandb.Image(visualization, caption=f"GradCAM Epoch {trainer.current_epoch + 1} Batch {batch_idx} Image 0")
                        pl_module.logger.experiment.log({"GradCAM Images": wandb_img})

                        # #below is for debugging purposes
                        # x_img = Image.fromarray(x_debug)
                        # x_img.save(os.path.join(self.outdir, f'input image{trainer.current_epoch + 1}_batch{batch_idx}_img0.jpg'))
                        # wandb_img = wandb.Image(x_img, caption=f"image input {trainer.current_epoch + 1} Batch {batch_idx} Image 0")
                        # pl_module.logger.experiment.log({"Images": wandb_img})

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
        try:
            backbone = model.backbone

            if hasattr(backbone, 'layer4'):
                return backbone.layer4[-1]  # Return the last block of layer4 for GradCAM
        except:
            print('model has no backbone')
            
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
        


        backbone = model.backbone
        if hasattr(backbone, 'layer4'):
            return backbone.layer4[-1]  # Return the last block of layer4 for GradCAM