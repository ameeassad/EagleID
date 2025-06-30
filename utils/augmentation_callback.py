from pytorch_lightning.callbacks import Callback
import torch


class AugmentationCallback(Callback):
    """
    Callback to enable advanced augmentations after a specified epoch.
    This helps transformer models learn clean label signals first before applying aggressive augmentations.
    """
    
    def __init__(self, enable_epoch=10):
        """
        Args:
            enable_epoch (int): Epoch at which to enable advanced augmentations
        """
        super().__init__()
        self.enable_epoch = enable_epoch
        self.augmentation_enabled = False
    
    def on_train_start(self, trainer, pl_module):
        """Log the augmentation strategy"""
        print(f"AugmentationCallback: Advanced augmentations will be enabled at epoch {self.enable_epoch}")
        if hasattr(trainer.datamodule, 'train_dataset') and hasattr(trainer.datamodule.train_dataset, 'transform'):
            if hasattr(trainer.datamodule.train_dataset.transform, 'use_advanced_aug'):
                current_state = trainer.datamodule.train_dataset.transform.use_advanced_aug
                print(f"AugmentationCallback: Current advanced augmentation state: {current_state}")
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Enable advanced augmentations at the specified epoch"""
        current_epoch = trainer.current_epoch
        
        if current_epoch == self.enable_epoch and not self.augmentation_enabled:
            # Enable advanced augmentations in the transform
            if hasattr(trainer.datamodule, 'train_dataset') and hasattr(trainer.datamodule.train_dataset, 'transform'):
                transform = trainer.datamodule.train_dataset.transform
                
                # Enable advanced augmentations
                if hasattr(transform, 'use_advanced_aug'):
                    transform.use_advanced_aug = True
                    self.augmentation_enabled = True
                    print(f"AugmentationCallback: Advanced augmentations ENABLED at epoch {current_epoch}")
                    
                    # Log to wandb if available
                    if hasattr(pl_module, 'log'):
                        pl_module.log('train/advanced_aug_enabled', 1.0, prog_bar=False)
                
                # Also enable MixUp/CutMix if the transform supports it
                if hasattr(transform, 'mixup_prob'):
                    transform.mixup_prob = 0.3
                    transform.cutmix_prob = 0.3
                    print(f"AugmentationCallback: MixUp/CutMix ENABLED at epoch {current_epoch}")
                
                # Enable advanced augmentations probability
                if hasattr(transform, 'advanced_aug_prob'):
                    transform.advanced_aug_prob = 0.8
                    print(f"AugmentationCallback: Advanced aug probability set to 0.8 at epoch {current_epoch}")
        
        # Log current augmentation state
        if hasattr(pl_module, 'log'):
            pl_module.log('train/advanced_aug_enabled', 1.0 if self.augmentation_enabled else 0.0, prog_bar=False) 