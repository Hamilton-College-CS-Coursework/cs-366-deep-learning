from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T # needed to transform images into a tensor
import torchvision.transforms.functional as F # necessary for augmentation def
import torch
import lightning as L
from PIL import Image # handles image resizing
import numpy as np
import random

# src: Xiao Chen's oxpet_download_and_viz.py (OxfordIIITPet)
def mask_to_classes(mask_pil, mode="trimap"):
    """ Change a PIL mask from the Oxford-IIIT Pet dataset into:
        • binary | pet = 1, background = 0 
        • trimap | border = 2, background = 1, pet = 0 """
    m = np.array(mask_pil, dtype=np.int64)
    if mode == "trimap":
        m = m - 1
    else: # binary mode
        pet = (m == 1) | (m == 3)
        m = pet.astype(np.int64)
    # change it into a int type Pytorch sensor
    return torch.from_numpy(m).long()

#Data Augmentation for Improvement #1
#Use Functional: https://www.geeksforgeeks.org/computer-vision/pytorch-functional-transforms-for-computer-vision/
#More control with functional since augments to the img need to match augments to the respective mask
#Work on PIL images

def augment_geometric(img, mask):
    """Applies geometric augmentations to PIL Image and its mask"""
    # 75% chance of any augmentation
    if random.random() < 0.75:
        # 50% chance of flipping
        if random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)
        
        # 50% chance of rotation
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle, interpolation=Image.NEAREST) # av
            
    return img, mask

def augment_photometric(img_tensor):
    """Applies photometric augmentations to an image tensor"""
    # 75% chance of any photometric augmentation
    if random.random() < 0.75:
        # 50% chance of invert color
        if random.random() < 0.5:
            img_tensor = F.invert(img_tensor)

        # 50% chance of change in brightness
        if random.random() < 0.5:
            brightness = random.uniform(0.5,1.5)
            img_tensor = F.adjust_brightness(img_tensor, brightness)

        # 50% chance of blur
        if random.random() < 0.5:
            # for tensor input, kernel_size can be a single int or sequence
            img_tensor = F.gaussian_blur(img_tensor, kernel_size = [3, 3]) 
    return img_tensor

class OxfordIIITPetMapper(Dataset):
    """
    Wraps the OxfordIIITPet dataset to apply resizing, augmentations, and
    maps pixel values to class ID's for the trimap and binary
    """
    def __init__(self, base_dataset, resize, mode="trimap", aug=False):
        self.base = base_dataset
        self.mode = mode
        self.aug = aug
        self.resize = resize

        # for resizing and transforming an image and mask into tensors
        self.resize_img = T.Resize((self.resize, self.resize))
        self.resize_mask = T.Resize((self.resize, self.resize),
                                    interpolation=Image.NEAREST)
        self.to_tensor = T.ToTensor()
        self.pil_to_tensor = T.PILToTensor()
        # src: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]) # Improvement #2
    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx] # mask is a PIL image here

        if self.aug:
            # first, do a geometric augmentation on the PIL images
            img, mask = augment_geometric(img, mask)
        
        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        # resize and transform into a tensor
        img = T.Resize((self.resize, self.resize))(img) # Resize must be done explicitly here
        img = T.ToTensor()(img)

        # NOW do the photometric augmentation on the image tensor
        if self.aug:
            img = augment_photometric(img)

        # normalize the image tensor
        img = self.normalize(img)

        # resize and transform into tensor
        mask = self.pil_to_tensor(mask)
       
        mask = mask.squeeze(0) # remove channel dimension
        mask = torch.clamp(mask, min=1) # prevents raw pixels from becoming -1
        mask = mask_to_classes(mask, self.mode) # map class to ID's {0, 1, 2} and {0, 1}
        return img, mask

# src: https://lightning.ai/docs/pytorch/stable/data/datamodule.html#why-do-i-need-a-datamodule
class OxfordPetDataModule(L.LightningDataModule):
    """
    Downloads and prepares the Oxford IIIT Pet dataset for model
    training and loads the training, validation, and tests sets in
    select batch sizes
    """
    def __init__(
        self, 
        location="~/hamilton/cpsci366/data/oxford-iiit-pet",
        batch_size=64,
        num_workers=8,
        class_choice="trimap",
        resize = 512 # the default is 512, if not provided
    ):
        """set all parameters and resize the images"""
        super().__init__()
        self.location = location # file path to the dataset
        self.batch_size = batch_size # minibatch stochastic descent batch size
        self.num_workers = num_workers # how many GPUs we'll used; pytorch calls them num_workers
        self.class_choice = class_choice # binary (i.e. ["pet", "border"])) or trimap (i.e. ["pet", "background", "border"])
        self.resize = resize

    # fetch the oxfordiitpet dataset
    def prepare_data(self):
        """retreive the data from the specified location; otherwise
        create a folder at the specified location and save the data
        there."""
        OxfordIIITPet(root=self.location, download=True)

    def setup(self, stage=None):
        """splits the data into training, validation if trainer.fit(...) is run;
         otherwise, splits the data into a test set"""
        # creates the training and validation sets
        if stage in (None, "fit"):
            base_ds = OxfordIIITPet(
                root=self.location, # the assumption is now I already have the data at the location I've specified
                split="trainval",
                target_types="segmentation",
                download=False # we don't need to download the dataset again
            )

            train_set_size = int(0.85 * len(base_ds)) # size for the training set 85/15
            val_set_size = len(base_ds) - train_set_size # size of dataset - size of training set == size of validation set

            # random_split(<your_dataset>, [size of training set, size of validation set])
            train_subset, val_subset = random_split(
                base_ds,
                [train_set_size, val_set_size],
                generator=torch.Generator().manual_seed(101) # seed is useful to reproduce results
            )
        
            self.train_set = OxfordIIITPetMapper(train_subset, self.resize,  mode=self.class_choice, aug=True) #Use augment for train
            self.val_set   = OxfordIIITPetMapper(val_subset, self.resize,  mode=self.class_choice, aug=False) #No augment for val

        # creates the test set
        base_test = OxfordIIITPet(
            root=self.location,
            split="test",
            target_types="segmentation",
            download=False # we don't need to download the dataset again
        )
        self.test_set = OxfordIIITPetMapper(base_test, self.resize, mode=self.class_choice, aug=False) #No augment for test

    def train_dataloader(self):
        """loads the data for training dataset"""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    def val_dataloader(self):
        """loads validation dataset"""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    def test_dataloader(self):
        """loads test dataset"""
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )