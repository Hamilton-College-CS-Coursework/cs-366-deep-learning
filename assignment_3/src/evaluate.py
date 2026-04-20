import lightning as L
import torch
from transformers import BertTokenizer

import config
from datamodule import IMDBDataModule
from light_model import LightningBi_LSTM


def main():
    """Loads the checkpoint, evaluates it, and prints metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/best_model.ckpt" # update this with the location of the chkpoint

    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_NAME)

    print(f"Loading model from {checkpoint_path}...")
    model_module = LightningBi_LSTM.load_from_checkpoint(
        checkpoint_path, vocab_size=tokenizer.vocab_size
    )
    model_module.to(device)
    model_module.eval()

    print("Preparing the IMDB data module...")
    data_module = IMDBDataModule(
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_SEQ_LENGTH
    )
    # run the test dataset
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    all_preds = []
    all_labels = []
    misclassed = []

    #Loop to collect the misclassified examples 
    print("running evaluation loop...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["ratings"].to(device)
            
            logits = model_module(input_ids)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().tolist()) # shift to cpu to save resources
            all_preds.extend(preds.cpu().tolist())

            # finds all the indices where the predictions differ from the labels,
            # returning a tensor of misclassified positions then flattens it into
            # Python list
            mis_idxs = torch.nonzero(preds != labels,as_tuple=False).flatten().tolist()
            # the for loop walks through each observation in the list and decodes it
            # back into text via tokenizer.decode(...)
            for idx in mis_idxs:
                decoded_txt = tokenizer.decode(
                    batch["input_ids"][idx].tolist(), 
                    skip_special_tokens=True).strip() # strip removes leading and trailing spaces, newlines, and tabs
                # and then builds a dictionary capturing the predicted class, the true class,
                # and the decoded review, later appending it to misclassed (i.e a Python list)
                misclassed.append(
                    {"predicted": preds[idx].item(), 
                    "true": labels[idx].item(),
                     "text": decoded_txt
                    }
                )

    # The metrics
    # Technically a second evaluation but the first non lightning loop was to collect the misclassifed
    trainer = L.Trainer()
    trainer.test(model_module, datamodule=data_module)

    print("\n3 misclassified examples...")
    for example in misclassed[:3]:
        predicted = example["predicted"]
        true = example["true"]
        text = example["text"]
        print(f'Predicted: {predicted}\nTrue: {true}\nText: {text}\n')

if __name__ == "__main__":
    L.seed_everything(config.RANDOM_SEED, workers=True)
    main()
