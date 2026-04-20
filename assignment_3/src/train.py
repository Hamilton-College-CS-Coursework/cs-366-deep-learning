import lightning as L
from lightning.pytorch.callbacks import \
ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import \
    WandbLogger, TensorBoardLogger
from transformers import BertTokenizer
import config
from datamodule import IMDBDataModule
from light_model import LightningBi_LSTM

import os
import wandb
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
api_key = os.getenv("API_KEY")

if not api_key:
    print("API_KEY not found in .env file.")
    print("Continuing without API key...")
    api_key = None

# turns the config file into a python dictionary
config_dict = {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("__")
    }

wandb.init(
    project="IMDB",
    config=dict(
        batch_size=config.BATCH_SIZE,
        optimizer=config.OPTIMIZER,
        lr=config.LEARNING_RATE,
        scheduler=config.SCHEDULER,
        loss=config.LOSS,
        epochs=config.NUM_EPOCHS,
    ),
)

def main():
    # random seed to reproduce this model experiment
    L.seed_everything(config.RANDOM_SEED, workers=True)

    # uses Google's pre-trained bert-base-uncased's tokenizer
    print("loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    # get vocab size to pass to the model
    vocab_size = tokenizer.vocab_size

    print("preparing and initializing the IMDB data module...")
    data_module = IMDBDataModule(
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_SEQ_LENGTH
    )
    # passes the bert-base-uncased vocab_size to your LightningBi_LSTM
    model_module = LightningBi_LSTM(vocab_size=vocab_size)

    # saves best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # save the chkpts here
        filename="best_model", # Add -{epoch:02d}-{val_loss:.2f} to end during testing
        save_top_k=1,         # save the best model
        monitor="val_loss",   # monitor the val_loss
        mode="min"            # choose lowest val_loss
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5
    )
    wandb_logger = WandbLogger(
        project="IMDB",  # project name on wandb
        log_model=True,
        config=config_dict # from the config.py file
    )
    logger = TensorBoardLogger(
        "lightning-log",
        name="imdb"
    )
    trainer = L.Trainer(
            precision="16-mixed",
            max_epochs=config.NUM_EPOCHS,
            callbacks=[checkpoint_callback, early_stop],
            logger=wandb_logger,
            devices=1,
            accelerator="gpu"
        )

    print("training in process...")
    trainer.fit(model_module, datamodule=data_module)
    print("training done!")

if __name__ == "__main__":
    main()
    wandb.finish()
