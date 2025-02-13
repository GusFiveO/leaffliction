import sys
import os
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 6, 5
        )  # (3, 256, 256) -> new dim (256 - 5 + (2 * 0)) / 1 + 1 ) = 252 (6, 252, 252)
        self.pool = nn.MaxPool2d(2, 2)  # (6, 252, 252) -> (6, 126, 126)
        self.conv2 = nn.Conv2d(
            6, 16, 5
        )  # (6, 126, 126) -> (16, 122, 122) -> MaxPool -> (16, 61, 61)
        self.fc1 = nn.Linear(16 * 61 * 61, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_dataset(directory_path):
    dataset = torchvision.datasets.ImageFolder(root=directory_path)
    print(dataset)
    return dataset


def train_model(dataset):
    net = CNN()
    print(net)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # for epoch in range(10):


if __name__ == "__main__":
    dir_path = sys.argv[1]
    dataset = load_dataset(dir_path)
    train_model(dataset)
