import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        """Basic Structure of a Multi-Layer Perceptron with 3 Fully Connected Layers for images 28x28 pixels."""
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        """Defines the activation of any neurons in the network.
            Rectified Linear Unit (ReLU) is defined as a binary function which equates the binary nature of a real neuron to our artificial one.
            By not using a gradient saturation function like sigmoid we speed up training due to the simple nature of 0 or 1.
        """
        # after every layer we apply ReLU activation function
        x = F.relu(F.max_pool2d(self.fc1(x), 2))
        x = F.relu(F.max_pool2d(self.fc2(x), 2))
        x = F.relu(F.max_pool2d(self.fc3(x), 2))
        x = x.view(-1, 28*28)
        return F.log_softmax(x)
        
    