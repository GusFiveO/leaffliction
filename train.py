import sys
import os
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_dataset(directory_path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = torchvision.datasets.ImageFolder(root=directory_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    # print(images, labels)
    print(dataset.classes)
    # imshow(torchvision.utils.make_grid(images))
    labels = np.array([dataset.targets[i] for i in range(len(dataset))])

    # Define split sizes
    train_size = 0.7  # 70% for training
    val_size = 0.15  # 15% for validation
    test_size = 0.15  # 15% for testing

    # First, split into train + temp (val + test)
    train_idx, temp_idx, _, temp_labels = train_test_split(
        np.arange(len(dataset)),
        labels,
        stratify=labels,
        test_size=(1 - train_size),
        random_state=42,
    )

    # Then, split temp into validation and test
    val_idx, test_idx = train_test_split(
        temp_idx,
        stratify=temp_labels,
        test_size=(test_size / (test_size + val_size)),
        random_state=42,
    )

    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
    # test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)
    test_loader = DataLoader(dataset, sampler=test_sampler)
    return train_loader, val_loader, test_loader


def train_model(train_loader):
    net = CNN()
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(i)
        print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
    return net


def test_model(net, test_loader):
    classes = test_loader.classes
    outputs = net(test_loader)
    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))


if __name__ == "__main__":
    dir_path = sys.argv[1]
    train_loader, val_loader, test_loader = load_dataset(dir_path)
    net = train_model(train_loader)
    test_model(net, test_loader)
