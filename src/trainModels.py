import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn as nn
from models.MLP import MLP
from models.CNN import CNN
from models.LTSM import LTSM

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

global device

def trainMLP(epoch):
    """Function to train a Multi-Layer Perceptron (MLP) model."""
    model = MLP().to(device)

    lossFunct = nn.NLLLoss()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

    """Defining the training process"""
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Prior to back propagation, we need to zero the gradients
        output = model(data) # intial correct class prediction
        loss = lossFunct(output, target) # Calculate the loss (how close it is to the correct solution)
        loss.backward() # Propogate loss backwards (our incentive for the model to change weights)
        optimizer.step()

        # Logging
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)}' # type: ignore
                  f'({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')\
                  
    model.eval()
    testLoss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += lossFunct(output, target).item() # we calculate but do not propogate the loss
            pred = output.argmax(dim=1, keepdim=True) # predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(loaders["test"].dataset) # type: ignore
    print(f'\nTest set: Average loss: {testLoss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)}' # type: ignore
          f'({100. * correct / len(loaders["test"].dataset):.0f}%)\n') # type: ignore


def trainCNN(epoch):
    """Function to train a Convolutional Neural Network (CNN) model."""
    model = CNN().to(device)

    lossFunct = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.03)

    """Defining the training process"""
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Prior to back propagation, we need to zero the gradients
        output = model(data) # intial correct class prediction
        loss = lossFunct(output, target) # Calculate the loss (how close it is to the correct solution)
        loss.backward() # Propogate loss backwards (our incentive for the model to change weights)
        optimizer.step()

        # Logging
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)}' # type: ignore
                  f'({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')\
                  
    model.eval()
    testLoss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += lossFunct(output, target).item() # we calculate but do not propogate the loss
            pred = output.argmax(dim=1, keepdim=True) # predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(loaders["test"].dataset) # type: ignore
    print(f'\nTest set: Average loss: {testLoss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)}' # type: ignore
          f'({100. * correct / len(loaders["test"].dataset):.0f}%)\n') # type: ignore

def trainLTSM(epoch):
    """Function to train a Long Short-Term Memory (LSTM) model."""
    model = LTSM().to(device)

    lossFunct = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    """Defining the training process"""
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Prior to back propagation, we need to zero the gradients
        output = model(data) # intial correct class prediction
        loss = lossFunct(output, target) # Calculate the loss (how close it is to the correct solution)
        loss.backward() # Propogate loss backwards (our incentive for the model to change weights)
        optimizer.step()

        # Logging
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)}' # type: ignore
                  f'({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')\
                  
    model.eval()
    testLoss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += lossFunct(output, target).item() # we calculate but do not propogate the loss
            pred = output.argmax(dim=1, keepdim=True) # predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(loaders["test"].dataset) # type: ignore
    print(f'\nTest set: Average loss: {testLoss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)}' # type: ignore
          f'({100. * correct / len(loaders["test"].dataset):.0f}%)\n') # type: ignore


if __name__ == "__main__":
    choice = input("Which model would you like to train? (MLP/CNN/LTSM): ").strip().upper()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    match choice:
        case "MLP":
            for epoch in range(1, 11):
                trainMLP(epoch)
        case "CNN":
            for epoch in range(1, 11):
                trainCNN(epoch)
        case "LTSM":
            for epoch in range(1, 11):
                trainLTSM(epoch)
        case _:
            print("Invalid choice. Please select MLP, CNN, or LTSM. Exiting...")