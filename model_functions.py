import torch
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from helper_functions import create_model_directory


class MNIST_DF_Dataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing MNIST data stored in a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing MNIST data with labels in the first column and pixel values in the remaining columns.

    Attributes:
        data (np.ndarray): Numpy array of the dataframe values.
        labels (torch.Tensor): Tensor containing the labels.
        images (torch.Tensor): Tensor containing the normalized pixel values.
    """

    def __init__(self, dataframe):
        self.data = dataframe.values
        self.labels = torch.tensor(self.data[:, 0], dtype=torch.long)  # First column contains the labels
        self.images = torch.tensor(self.data[:, 1:], dtype=torch.float32)  # Remaining columns are the pixel values
        self.images = self.images / 255.0  # Normalize pixel values to [0, 1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding label by index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: (image, label), where image is a reshaped tensor of size (28, 28) and label is a long tensor.
        """
        image = self.images[idx].reshape(28, 28)  # Reshape to 28x28
        label = self.labels[idx]
        return image, label

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Von 2 oder 3 auf 1 Kanal ändern
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 Klassen für MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(model, dataloader, epochs, optimizer, criterion, model_directory, model_name):
    """
    Train the model using the provided DataLoader, optimizer, and loss criterion.

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader providing the training data.
        epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        criterion (torch.nn.Module): Loss function to minimize.
        model_directory (str): Directory where model checkpoints will be saved.
        model_name (str): Name prefix for saving model checkpoints.

    Returns:
        None
    """

    # Create a directory to save the model checkpoints
    model_run_directory = create_model_directory(model_directory, model_name + time.strftime("_time-%H-%M-%S"))

    for epoch in range(epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # Get the inputs and labels from the DataLoader
            inputs, labels = data

            # Reshape input tensors to (batch_size, channels, height, width)
            inputs = inputs.unsqueeze(1)  # Add a dimension for the channel (1 for grayscale)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        # Save model checkpoint
        checkpoint_path = f"{model_run_directory}/epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / 2000
        }, checkpoint_path)

    print('Finished Training')


def evaluate_model(net, test_dataloader):
    """
    Evaluate the accuracy of the model on the test dataset.

    Args:
        net (torch.nn.Module): The trained neural network model.
        test_dataloader (DataLoader): DataLoader providing the test data.

    Returns:
        None
    """
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for data in test_dataloader:
            images, labels = data
            images = images.unsqueeze(1)  # Add a channel dimension
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')


import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_v2(net, test_dataloader):
    """
    Evaluate the accuracy of the model on the test dataset and print a confusion matrix along with standard evaluation metrics.

    Args:
        net (torch.nn.Module): The trained neural network model.
        test_dataloader (DataLoader): DataLoader providing the test data.

    Returns:
        None
    """
    net.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for data in test_dataloader:
            images, labels = data
            images = images.unsqueeze(1)  # Add a channel dimension
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Precision, Recall, F1 Score
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1:.4f}')

    return accuracy, precision, recall, f1

