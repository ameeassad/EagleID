import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.template_model import TemplateModel, EpochCallback
from pytorch_lightning.callbacks import Callback

def basic_trainer_pl(
    dataset,
    model,
    objective,
    optimizer,
    epochs,
    scheduler=None,
    device="cuda",
    batch_size=128,
    num_workers=1,
    accumulation_steps=1,
    epoch_callback=None,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    model = TemplateModel(
        model=model,
        loss_module=objective,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    callbacks = [EpochCallback(epoch_callback)] if epoch_callback else []

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        accumulate_grad_batches=accumulation_steps,
        callbacks=callbacks,
    )
    trainer.fit(model, loader)