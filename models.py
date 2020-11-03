import torch.nn.functional as F
from torch import nn
import torchvision


class Net(nn.Module):
    """
    LeNet5 customized net model
    input size must be an even number!
    """

    def __init__(self, nb_class,input_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16*int(input_size/4-2)*int(input_size/4-2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nb_class)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
