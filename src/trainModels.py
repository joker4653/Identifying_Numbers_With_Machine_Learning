from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data = datasets.MNIST(
    root = "data",
    train = True,
    transform = ToTensor(),
    download = True
)

test_data = datasets.MNIST(
    root = "data",
    train = False,
    transform = ToTensor(),
    download = True
)

loaders = {
    "train": DataLoader(train_data, 
                        batch_size=100, 
                        shuffle=True,
                        num_workers=1),
                        
    "test": DataLoader(test_data, 
                       batch_size=100, 
                       shuffle=True,
                       num_workers=1),
}


def trainMLP():
    """Function to train a Multi-Layer Perceptron (MLP) model."""
    pass

def trainCNN():
    """Function to train a Convolutional Neural Network (CNN) model."""
    pass

def trainLTSM():
    """Function to train a Long Short-Term Memory (LSTM) model."""
    pass


if __name__ == "__main__":
    choice = input("Which model would you like to train? (MLP/CNN/LTSM): ").strip().upper()

    match choice:
        case "MLP":
            trainMLP()
        case "CNN":
            trainCNN()
        case "LTSM":
            trainLTSM()
        case _:
            print("Invalid choice. Please select MLP, CNN, or LTSM. Exiting...")