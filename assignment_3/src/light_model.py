import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config
from model import Bi_LSTM

class LightningBi_LSTM(L.LightningModule):
  """ LightningModule wrapper for Bi_LSTM """
  def __init__(self, vocab_size: int):
    super().__init__()
    # saves all the __init__ args to the checkpoint Lightning logs
    self.save_hyperparameters() 
    self.learning_rate = config.LEARNING_RATE
    

    # vocab_size come from the bert-based-case tokenizer in 
    # the main training script
    self.model = Bi_LSTM(input_size=vocab_size)

    self.criterion = nn.NLLLoss()

    # logging metrics
    self.train_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=config.NUM_CLASSES
    )
    self.val_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=config.NUM_CLASSES
    )
    self.test_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=config.NUM_CLASSES
    )
    self.test_conf_matrix = torchmetrics.ConfusionMatrix(
        task="multiclass", num_classes=config.NUM_CLASSES
    )

  def forward(self, x):
    """Calls forward pass in the model"""
    return self.model(x)
  
  def _helper_step(self, batch, batch_idx):
    """ Training_step, validation_step, and test_step use 
    this function to get the pred, labels, and loss
    """
    # batch from the IMDBDataModule's collate_fn
    input_ids = batch['input_ids']
    labels = batch['ratings']
    
    # get the logits
    logits = self.forward(input_ids)
    
    # loss calc
    loss = self.criterion(logits, labels)
    
    # get preditions (class index with the highest logit)
    preds = torch.argmax(logits, dim=1)
    
    return loss, preds, labels
  

  def training_step(self, batch, batch_idx):
    """ Does one training step """
    loss, preds, labels = self._helper_step(batch, batch_idx)
    
    # Update and log metrics
    self.train_acc(preds, labels)
    self.log('train_loss', 
             loss, on_step=True, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)
    self.log('train_acc', 
             self.train_acc, 
             on_step=False, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)
    
    return loss # needed for backpropagation

  def validation_step(self, batch, batch_idx):
    """ Does one validation step """
    loss, preds, labels = self._helper_step(batch, batch_idx)
    
    # metrics logs and updates
    self.val_acc(preds, labels)
    self.log('val_loss', 
             loss, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)
    self.log('val_acc', 
             self.val_acc, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)

  def test_step(self, batch, batch_idx):
    """ Does one test step """
    loss, preds, labels = self._helper_step(batch, batch_idx)
    self.test_acc(preds, labels)
    self.test_conf_matrix(preds, labels)
    self.log('test_loss', 
             loss, on_epoch=True, 
             prog_bar=True, 
             logger=True)
    self.log('test_acc', 
             self.test_acc, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)
    
  def on_test_epoch_end(self):
    """ Runs after the test dataloader is done """
    # computes the last confusion matrix
    cm = self.test_conf_matrix.compute()
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1] 

    print("\nTest Confusion Matrix:")
    print(f'   0    1\n0 {tn} {fp}\n1 {fn} {tp}\n')
    #src: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.float.html
    self.log("test_TN", tn.to(torch.float32))
    self.log("test_FP", fp.to(torch.float32))
    self.log("test_FN", fn.to(torch.float32))
    self.log("test_TP", tp.to(torch.float32))
    self.test_conf_matrix.reset()

  def configure_optimizers(self):
    """ Prepares optimizer """
    optimizer = optim.AdamW(
          self.parameters(), 
          lr=self.learning_rate,
          weight_decay=0.01
    )
    # src: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    scheduler = {
        'scheduler': ReduceLROnPlateau(
            optimizer,
            mode='min',         # Monitor a loss metric ('min')
            factor=0.1,         # New LR = Current LR * 0.1
            patience=5,         # Number of epochs with no improvement before reduction
            min_lr=1e-7         # Minimum learning rate threshold
        ),
        'interval': 'epoch',    # Check the metric after every epoch
        'frequency': 1,         # Every 1 epoch
        'monitor': 'val_loss',  # The metric to monitor (must be logged in validation_step)
        'strict': True,
    }
    # Return the optimizer and scheduler dictionary
    return [optimizer], [scheduler]