import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        """Basic Structure of a Multi-Layer Perceptron with 3 Fully Connected Layers for images 28x28 pixels."""
        # input is 28*28 = 784
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        """Defines the activation of any neurons in the network.
            Rectified Linear Unit (ReLU) is defined as a binary function which equates the binary nature of a real neuron to our artificial one.
            By not using a gradient saturation function like sigmoid we speed up training due to the simple nature of 0 or 1.
        """
        # flatten the input image (batch_size, C, H, W) -> (batch_size, 28*28)
        x = x.view(x.size(0), -1)
        # fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
    """
    MLP was developed in 1943 and thus is one of the first neural network architectures ever created.
    Its simple nature makes it a good starting point for image recognition. However upon looking at what the
    hidden layers are actual learning we find that they are not learning anything related to the image itself.
    The asumption would be the hidden layers would identify edges and shapes which correspond to a particular number,
    however visualising the hidden layers shows their processing is seemingly random and could not be parsed
    even by a human as recognition. This is the main shortcoming of Multi-Layer Perceptrons.
    """