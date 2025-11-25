import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import LSTM, Linear, Dropout

class LTSM(nn.Module):
    def __init__(self):
        super(LTSM, self).__init__()
        """Basic Structure of a Long Short-Term Memory (LSTM) Network for MNIST classification"""
        # MNIST images are grayscale with 1 channel and size 28x28
        self.lstm = LSTM(input_size=28, hidden_size=128, num_layers=2, batch_first=True)
        self.dropout = Dropout(p=0.2) # Dropout layer is here to prevent overfitting after the LSTM Layers
        self.fc1 = Linear(128, 64) 
        self.fc2 = Linear(64, 10) # 10 outputs for MNIST

    def forward(self, x):
        """Defines the activation of any Neurons in the network"""
         # Reshape input to (batch_size, sequence_length, input_size), conceptually different from CNNs and MLPs
        x = x.view(x.size(0), 28, 28)
        lstm_out, (h_n, c_n) = self.lstm(x) # Forward pass on the LTSM Layer
        x = self.dropout(h_n[-1])  # Use the last layer's hidden state
        x = F.relu(self.fc1(x)) # On Fully conncted layers we continue to relu
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    """
    LSTM is fundamentally different from both Convultional Neural Networks and Multi-Layer Perceptrons as it 
    designed to handle sequential data. In this instance of the MNIST dataset, we treat each image as a sequence
    of 28 rows, which in turn have 28 pixels that are fed into the LTSM layer one at a time.
    """