"""__seed__"""
RANDOM_SEED = 42

"""__tokenizer__"""
TOKENIZER_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 256

"""__datamodule__"""
VAL_SPLIT = 0.15
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.15
BATCH_SIZE = 32
NUM_WORKERS = 8

"""__model__"""
LSTM_LAYERS = 2
DROPOUT_RATE = 0.3
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
NUM_CLASSES = 2 # positive/negative ratings

"""__training__"""
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
OPTIMIZER = "AdamW" # or some other optimizer algorithm
SCHEDULER="ReduceLROnPlateau"
LOSS="NLLoss"