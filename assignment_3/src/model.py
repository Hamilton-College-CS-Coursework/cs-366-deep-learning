import torch
import torch.nn as nn
import config

'''
Notes from Assignment directions:
    Implement a Bidirectional LSTM (Bi-LSTM) model (no attention)
    Embedding layer: trainable, no pre-trained weights
    Bidirectional LSTM: at least one Bi-LSTM layer (dropout optional)
    Fully connected layer: output = 2 classes (positive/negative)
    Activation: Sigmoid or Softmax
    No attention or transformer components allowed
'''
#LSTM citation: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#Embedding citation: https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
class Bi_LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int = config.EMBEDDING_DIM,
        hidden_size: int = config.HIDDEN_DIM,
        output_size: int = config.NUM_CLASSES,
        num_layers: int = config.LSTM_LAYERS,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        #Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        #Bi-LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, #Can change in testing
            batch_first=True, #check the order of embedding for this
            dropout=dropout,  #Leave this here to change in testing
            bidirectional=True, 
        )
        #FC layer with 2 class output
        self.fc = nn.Linear(hidden_size * 2, output_size)
        #Activation: Sigmoid or Softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.fc(x)
        x = self.softmax(x)
        return x