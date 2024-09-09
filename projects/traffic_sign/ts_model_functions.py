import os
from PIL import Image
from torch.utils.data import Dataset
from imageio.v2 import imread

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse the directory to find images and their corresponding labels
        image_folder_path = os.path.join(root_dir, "images")
        label_folder_path = os.path.join(root_dir, "labels")
        
        # Collect image paths and their corresponding label paths
        for img_file in os.listdir(image_folder_path):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(image_folder_path, img_file)
                label_path = os.path.join(label_folder_path, img_file.replace('.jpg', '.txt'))
                self.image_paths.append(img_path)
                self.labels.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        # image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = imread(img_path)

        # Load the label
        label_path = self.labels[idx]
        labels = []
        with open(label_path, 'r') as file:
            # Extract the first character from each line and convert to integer
            for line in file:
                label = int(line.split(' ')[0])
                labels.append(label)

        # Use the first label for this example, adjust if necessary
        if labels == []: 
            main_label = 15
        else: 
            main_label = labels[0]  # Example: Use only the first label found                

        # Apply the transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, main_label


from pathlib import Path
def create_model_directory(run_directory, model_name):
    """
    Create a directory for storing a specific model within a run directory.

    Args:
        run_directory (str): The path to the run directory where the model directory will be created.
        model_name (str): The name of the model for which the directory will be created.

    Returns:
        str: The path to the created model directory.
    """
    model_directory = os.path.join(run_directory, model_name)
    Path(model_directory).mkdir(parents=True, exist_ok=True)

    return model_directory

import torch
import time
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def train(model, train_loader, test_loader, epochs, optimizer, criterion, model_directory, model_name):
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
    model_run_directory = create_model_directory(model_directory, time.strftime("datetime_%y_%m_%d_%H_%M_%S-") + model_name)
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []


    for epoch in range(epochs):  # Loop over the dataset multiple times
        model.train()  # Set model to training mode
        train_running_loss = 0.0
        train_labels = []
        train_predictions = []


        # Initialize tqdm with custom description
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train Loss: 0.000 Train Acc: 0.000 Test Loss: 0.000 Test Acc: 0.000", leave=True)

        for i, data in enumerate(progress_bar):
            # Get the inputs and labels from the DataLoader
            inputs, labels = data
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # metrics
            # loss
            train_running_loss += loss.item()
            # calc pred label
            _, predicted = torch.max(outputs.data, 1)
            train_labels.extend(labels.numpy())
            train_predictions.extend(predicted.numpy())
            
            # Update tqdm description with current loss and accuracy
            train_acc = accuracy_score(train_labels, train_predictions)
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Train Loss: {train_running_loss / (i + 1):.6f} Train Acc: {train_acc:.6f} Test Loss: 0.000 Test Acc: 0.000")

        # Append training loss
        train_losses.append(train_running_loss / len(train_loader))
        
        # # Convert lists to numpy arrays
        # all_labels = np.array(all_labels)
        # all_predictions = np.array(all_predictions)
        # Precision, Recall, F1 Score
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_accuracies.append(train_accuracy)

        # Validation/Test Loop
        model.eval()  # Set model to evaluation mode
        test_labels = []
        test_predictions = []
        test_running_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_labels.extend(labels.numpy())
                test_predictions.extend(predicted.numpy())
        
        # Append training loss
        test_losses.append(test_running_loss / len(test_loader))
        # Calculate validation accuracy
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_accuracies.append(test_accuracy)

        # Update tqdm description with validation accuracy
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Train Loss: {train_running_loss / len(train_loader):.6f} Train Acc: {train_acc:.6f} Test Loss: {test_running_loss / len(test_loader):.6f} Test Acc: {test_accuracy:.6f}")
        progress_bar.close()
        
        metrics_dict = {'train_losses': train_losses,
                        'train_accuracies': train_accuracies,
                        'test_losses': test_losses,
                        'test_accuracies': test_accuracies}

        # Save model checkpoint
        checkpoint_path = f"{model_run_directory}/epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics':metrics_dict
        }, checkpoint_path)

    print('Finished Training')
    return metrics_dict


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
            # images = images.unsqueeze(1)  # Add a channel dimension
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

def evaluate_model_v2(net, test_dataloader,fig_size=(10,10)):
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
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=fig_size)
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


def imshow(img, figsize, title=None):
    """
    Display an image using Matplotlib.

    Args:
        img (torch.Tensor): The image tensor to display.
        figsize (tuple): Size of the figure for the plot.
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        None
    """
    img = img.numpy()  # Convert to numpy array
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')  # Display grayscale image
    if title is not None:
        plt.title(title)
    
    plt.show()

def visualize_predictions(net, test_dataloader, num_images=6, figsize=(4, 4)):
    """
    Visualize predictions made by the model on the test dataset.

    Args:
        net (torch.nn.Module): The trained neural network model.
        test_dataloader (DataLoader): DataLoader providing the test data.
        num_images (int, optional): Number of images to visualize. Defaults to 6.
        figsize (tuple, optional): Size of the figure for each plot. Defaults to (4, 4).
    
    Returns:
        None
    """
    dataiter = iter(test_dataloader)
    images, labels = next(dataiter)
    
    # Ensure not to display more images than available in the batch
    num_images = min(num_images, images.size(0))
    
    images = images[:num_images]
    labels = labels[:num_images]

    # Get model predictions
    outputs = images #net(images.unsqueeze(1))  # Add channel dimension
    _, predicted = torch.max(outputs, 1)

    # Display individual images
    for i in range(num_images):
        imshow(images[i].squeeze(), figsize=figsize,
               title=f"Label: {labels[i].item()}, Predicted: {predicted[i].item()}")  # Remove extra dimensions
