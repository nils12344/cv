import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def mnist_csv_plot_number(df, index=0, count=None, figsize=(3, 3)):
    """
    Plot a specified number of MNIST images from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing MNIST data with labels in the first column and pixel values in the remaining columns.
        index (int, optional): Starting index for plotting images. Defaults to 0.
        count (int, optional): Number of images to plot. Defaults to None (plots only one image).
        figsize (tuple, optional): Size of the figure for each plot. Defaults to (3, 3).
    
    Returns:
        None
    """
    for i in range(count):
        image_data = df.iloc[index, 1:].values
        label = df.iloc[index, 0]

        # Reshape picture to original 28x28 
        image_data = image_data.reshape(28, 28)

        plt.figure(figsize=figsize)
        plt.imshow(image_data, cmap='gray')
        plt.title(f'Row: "{index}", Label: "{label}"')
        
        index += 1

    plt.show()


def mnist_csv_plot_number_canvas(df, index=0, figsize=(11, 11)):
    """
    Plot a single MNIST image from a DataFrame with annotated pixel values.

    Args:
        df (pd.DataFrame): DataFrame containing MNIST data with labels in the first column and pixel values in the remaining columns.
        index (int, optional): Index of the image to plot. Defaults to 0.
        figsize (tuple, optional): Size of the figure for the plot. Recommended to be at least (11, 11) for visibility.
    
    Returns:
        None
    """
    images = df.iloc[index, 1:].values
    label = df.iloc[index, 0]

    # Reshape picture to original 28x28 
    img = images.reshape(28, 28)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')
    
    plt.title(f'Row: "{index}", Label: "{label}"')
    plt.show()


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


def create_run_directory(path, special_name=None):
    """
    Create a run directory for storing output data related to a specific run.

    Args:
        path (str): The path where the run directory will be created.
        special_name (str, optional): An additional name to append to the run directory for better identification. Defaults to None.

    Returns:
        str: The path to the created run directory.
    """
    current_time = time.strftime("%Y-%b-%d")
    run_directory_string = "Run_" + current_time

    if special_name is not None:
        run_directory_string = run_directory_string + "_" + special_name

    run_directory = os.path.join(path, run_directory_string)
    Path(run_directory).mkdir(parents=True, exist_ok=True)

    return run_directory


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
    outputs = net(images.unsqueeze(1))  # Add channel dimension
    _, predicted = torch.max(outputs, 1)

    # Display individual images
    for i in range(num_images):
        imshow(images[i].squeeze(), figsize=figsize,
               title=f"Label: {labels[i].item()}, Predicted: {predicted[i].item()}")  # Remove extra dimensions
