import os
import shutil
from pytorch_lightning.callbacks import Callback

class WandbCacheCleanupCallback(Callback):
    def __init__(self, every_n_epochs=1):
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch % self.every_n_epochs == 0:
            cache_path = os.path.expanduser("~/.cache/wandb/artifacts")
            if os.path.exists(cache_path):
                try:
                    shutil.rmtree(cache_path)
                    print(f"🧹 Cleared W&B cache at epoch {current_epoch}")
                except Exception as e:
                    print(f"⚠️ Failed to clear W&B cache: {e}") 