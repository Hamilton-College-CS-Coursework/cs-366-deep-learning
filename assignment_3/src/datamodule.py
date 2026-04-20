from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from transformers import BertTokenizer
import torch
import lightning as L
import config 

class IMDBDataModule(L.LightningDataModule):
  """Downloads and prepares the IMDB Dataset for model
  training and loads the training, validation, and tests sets"""
  def __init__(self, tokenizer: BertTokenizer, batch_size: int, max_length: int, root: str = ".data"):
    super().__init__()
    self.root = root # directory where the IMDB dataset will be saved
    self.tokenizer = tokenizer # BertTokenizer
    self.max_length = max_length # fixed uniform length for every review
    self.batch_size = batch_size # size of the batch for training
    self.num_workers = config.NUM_WORKERS
    
    # collate_fn expects a function that translate our reviews and ratings into a format
    # the model can understand and use for training
    self.collate_fn = self.translate

  def prepare_data(self):
    """Downloads the IMDB dataset
    Important: Hugging Face datasets automatically downloads the dataset"""
    load_dataset("imdb", cache_dir=self.root)
  
  def setup(self, stage: str = None):
    """Seteup the training, validation, and test sets
    Reqs: 70 (train) /15 (validation)/ 15 (test)"""

    # Load the IMDB dataset from Hugging Face
    dataset = load_dataset("imdb", cache_dir=self.root)
    
    # Combine train and test splits
    train_list = [(item['label'] + 1, item['text']) for item in dataset['train']]
    test_list = [(item['label'] + 1, item['text']) for item in dataset['test']]

    # combine the sets
    IMDB_dataset = train_list + test_list

    # the dataset's size; ned this to compute for 70/15/15
    size = len(IMDB_dataset)
    train_set_size = int(size * config.TRAIN_SPLIT)
    val_set_size = int(size * config.VAL_SPLIT)
    
    # Make test_set_size the remainder to ensure they add up perfectly
    test_set_size = size - train_set_size - val_set_size

    # The order of sizes in the list MUST match the order of variables
    self.train_dataset, self.val_dataset, self.test_dataset = random_split(
      IMDB_dataset, 
      [train_set_size, val_set_size, test_set_size],
      generator = torch.Generator().manual_seed(config.RANDOM_SEED)
  )
    
  def translate(self, batch):
    """Translates the batch of reviews and ratings into numerical tensors for the model"""
    # The IMDB dataset returns (rating, review).
    # src: https://www.kaggle.com/code/tusonggao/get-imdb-data-from-torchtext
    # src: https://www.w3schools.com/python/ref_func_zip.asp. how to use zip function

    ratings, reviews = zip(*batch) # unzips list of (rating, review) tuples
    
    # tokenize the batch of reviews
    encoding = self.tokenizer(
          reviews, # chagng tuple of reviews into a list
          padding='max_length', # padding to standardize review tensors to size 256
          truncation=True, # cut the review tensor if it's above 256
          max_length=self.max_length, # size 256 for padding
          return_tensors='pt' # "pt" is for pytorch, return pytorch tensors
    )
    # ratings: 1 (neg) and 2 (pos). Subtract 1 so that neg -> 0 and pos -> 1
    ratings_tensor = torch.tensor(
       [rate - 1 for rate in ratings],
       dtype = torch.long
    )
    # single dictionary for the model
    encoding["ratings"] = ratings_tensor
    return encoding
  
  def train_dataloader(self):
    """Load em (training set, validation set, and test set) up!"""
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      collate_fn=self.collate_fn, # Use the assigned function
      num_workers = self.num_workers
    )
    
  def val_dataloader(self):
    return DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      collate_fn=self.collate_fn,
      num_workers = self.num_workers
    )
      
  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      collate_fn=self.collate_fn,
      num_workers = self.num_workers
    )