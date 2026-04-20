import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from datamodule_oxpet import OxfordPetDataModule
from model_unetpp import LightningUnetPP

import os
import wandb
from dotenv import load_dotenv

load_dotenv()  # loads api_key from .env
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY not found! Make sure it's set in your .env file.")

wandb.init(
    project="oxpet_unetpp",
    config=dict(
        resize=512,
        batch_size=4,
        classes="trimap",
        optimizer="adamw",
        lr=1e-3,
        scheduler="plateau",
        loss="cross_entropy",
        epochs=100,
    ),
)
# src: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#starter-example
def main():
    cfg = wandb.config # accesses hyperparameters
    # prepare the data
    dm = OxfordPetDataModule(
        batch_size=cfg.batch_size, 
        class_choice="trimap",
        resize=cfg.resize
    )
    dm.prepare_data()
    dm.setup()

    # load the model
    num_classes = 3 if dm.class_choice == "trimap" else 2
    model = LightningUnetPP(num_classes=num_classes, lr=cfg.lr)

    # set callbacks and logger
    checkpoint = ModelCheckpoint(
        monitor="val_miou", # select the best model from THIS epoch that produces to best image segmentation prediction
        save_top_k=1, # save the best model
        mode="max", # maximize the mIoU
        filename="best_chkpt"
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10
    )
    wandb_logger = WandbLogger(
        project="oxpet_unetpp"
        )
    logger = TensorBoardLogger(
        "lightning_logs", 
        name="unetpp"
    )
    # create the PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint, early_stop],
        accumulate_grad_batches=4,
        logger=[logger, wandb_logger],
        log_every_n_steps=10,
    )
    # start training
    trainer.fit(model, datamodule=dm)

    print("Best checkpoint:", checkpoint.best_model_path)
    print("Best mIoU:", checkpoint.best_model_score.item())

if __name__ == "__main__":
    main()
    wandb.finish()

