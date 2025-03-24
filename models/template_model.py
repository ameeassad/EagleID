import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback

class TemplateModel(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_module,
        optimizer,
        scheduler=None,
    ):
        super().__init__()
        self.model = model
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.loss_module(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log("train_loss_epoch_avg", avg_loss)
        self.train_losses.clear()

    def configure_optimizers(self):
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
            }
        return self.optimizer

    def save(self, folder, file_name="checkpoint.pth", save_rng=True):
        checkpoint = {
            "model": self.model.state_dict(),
            "loss_module": self.loss_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
        }
        if save_rng:
            checkpoint["rng_states"] = torch.get_rng_state()
        os.makedirs(folder, exist_ok=True)
        torch.save(checkpoint, os.path.join(folder, file_name))

    def load(self, path, load_rng=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.loss_module.load_state_dict(checkpoint["loss_module"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if load_rng and "rng_states" in checkpoint:
            torch.set_rng_state(checkpoint["rng_states"])

class EpochCallback(Callback):
    def __init__(self, epoch_callback):
        self.epoch_callback = epoch_callback

    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = trainer.callback_metrics.get("train_loss_epoch_avg")
        epoch_data = {"train_loss_epoch_avg": avg_loss.item() if avg_loss else None}
        if self.epoch_callback:
            self.epoch_callback(trainer=trainer, epoch_data=epoch_data)