import torch
import torch.nn as nn
import torch.nn.functional as F

class ts_net(nn.Module):
    def __init__(self):
        super(ts_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # input (3), output (6), kernel size (5x5)
        self.pool = nn.MaxPool2d(2, 2)   # Max pooling layer with kernel size 2x2
        self.conv2 = nn.Conv2d(6, 16, 5) # input (6), output (16), kernel size (5x5)

        # Dynamische Berechnung der Größe nach Convolutional Layern
        self._to_linear = None
        self.convs = nn.Sequential(self.conv1, self.pool, self.conv2, self.pool)
        self._get_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 120)  # angepasst an die dynamische Größe
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 16)  # Ausgabeschicht (15 Klassen+1)

    def _get_conv_output_size(self):
        # Dummy forward pass to calculate output size
        with torch.no_grad():
            x = torch.zeros(1, 3, 416, 416)  # dummy input
            x = self.convs(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# more convultional layer, more filter, batch normalization
class ts_net_filter_norm(nn.Module):
    def __init__(self):
        super(ts_net_filter_norm, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # input (3), output (32), kernel size (3x3)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization after conv1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # Dynamische Berechnung der Größe nach Convolutional Layern
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.pool,
            self.conv2, self.bn2, nn.ReLU(), self.pool,
            self.conv3, self.bn3, nn.ReLU(), self.pool
        )
        self._get_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)  # angepasste Größe
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)

    def _get_conv_output_size(self):
        # Dummy forward pass to calculate output size
        with torch.no_grad():
            x = torch.zeros(1, 3, 416, 416)  # dummy input
            x = self.convs(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# dropout, more neurons in fc layer
class ts_net_dropout(nn.Module):
    def __init__(self):
        super(ts_net_dropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.dropout = nn.Dropout(0.5)  # Dropout rate 50%

        # Dynamische Berechnung der Größe nach Convolutional Layern
        self._to_linear = None
        self.convs = nn.Sequential(self.conv1, nn.ReLU(), self.pool, 
                                   self.conv2, nn.ReLU(), self.pool,
                                   self.conv3, nn.ReLU(), self.pool)
        self._get_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 16)

    def _get_conv_output_size(self):
        # Dummy forward pass to calculate output size
        with torch.no_grad():
            x = torch.zeros(1, 3, 416, 416)  # dummy input
            x = self.convs(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.dropout(F.relu(self.fc1(x)))  # Apply dropout after activation
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x



# resnet 18 pretrained model 
from torchvision import models

class ts_pretrained_rsnet18(nn.Module):
    def __init__(self):
        super(ts_pretrained_rsnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Load pre-trained ResNet18
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 16)  # Replace final layer to match our number of classes

    def forward(self, x):
        x = self.model(x)
        return x



# https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
