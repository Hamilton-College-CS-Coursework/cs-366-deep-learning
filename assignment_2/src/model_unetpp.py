import torch
import torch.nn as nn
import lightning as L
from unetplusplus import UNetPP
from torchmetrics.segmentation import DiceScore, MeanIoU

#src: https://lightning.ai/pages/community/tutorial/step-by-step-walk-through-of-pytorch-lightning/
class LightningUnetPP(L.LightningModule):
    def __init__(self, num_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNetPP(num_classes=num_classes) # our model given a class type
        self.loss_fn = nn.CrossEntropyLoss() # the loss function we're using is cross entropy
        self.lr = lr # the learning rate
        self.class_names = ["pet", "background", "border"] if num_classes == 3 else ["background", "pet"]

        self.miou = MeanIoU(num_classes=num_classes, 
                            per_class=False,
                            input_format='index')# overall meanIoU
        self.miou_per_class = MeanIoU(num_classes=num_classes,
                                      per_class=True,
                                      input_format='index')# per class meanIoU

        self.dice = DiceScore(num_classes=num_classes,
                              input_format='index',
                              average='micro') # overall dice
        self.dice_per_class = DiceScore(num_classes=num_classes,
                                        input_format='index',
                                        average='none') # per class dice
    def forward(self, x):
        """returns the Unet++ model"""
        return self.model(x) # returns the model

    def training_step(self, batch, batch_idx):
        """computes loss for one training batch and computes
        logs for training loss before returning the training loss
        for backpropagation"""
        imgs, masks = batch
        if masks.ndim == 4: # dimension check
            masks = masks.squeeze(1)
        masks = masks.long()

        logits = self(imgs) # [B, num_classes, H, W]
        loss = self.loss_fn(logits, masks)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """computes loss for one training batch and computes
        logs for the validation loss before retruning the validation
        loss for backpropagation"""
        imgs, masks = batch
        if masks.ndim == 4: # dimension check
            masks = masks.squeeze(1)
        masks = masks.long()

        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.miou.update(preds, masks)
        self.miou_per_class.update(preds, masks)
        self.dice.update(preds, masks)
        self.dice_per_class.update(preds, masks)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """resets the dice and miou metrics for the next epoch"""
        miou_overall = self.miou.compute()
        dice_overall = self.dice.compute()
        miou_classes = self.miou_per_class.compute()
        dice_classes = self.dice_per_class.compute()

        # metric logs visable during runs
        self.log('val_miou', miou_overall, prog_bar=True)
        self.log('val_dice', dice_overall, prog_bar=True)

        for i, name in enumerate(self.class_names):
            self.log(f'val_iou/{name}', miou_classes[i], on_epoch=True)
            self.log(f'val_dice/{name}', dice_classes[i], on_epoch=True)
    
        self.dice.reset()
        self.dice_per_class.reset()
        self.miou_per_class.reset()
        self.miou.reset()
    
    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.long()

        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.miou.update(preds, masks)
        self.miou_per_class.update(preds, masks)
        self.dice.update(preds, masks)
        self.dice_per_class.update(preds, masks)

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        miou_overall = self.miou.compute()
        dice_overall = self.dice.compute()
        miou_classes = self.miou_per_class.compute()
        dice_classes = self.dice_per_class.compute()

        self.log("test_miou", miou_overall)
        self.log("test_dice", dice_overall)

        for i, name in enumerate(self.class_names):
            self.log(f"test_iou/{name}", miou_classes[i])
            self.log(f"test_dice/{name}", dice_classes[i])

        self.dice.reset()
        self.dice_per_class.reset()
        self.miou.reset()
        self.miou_per_class.reset()

    def configure_optimizers(self):
        """assigns the optimizer with the learning rate
        and weight decay values and assignes the scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=10
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
