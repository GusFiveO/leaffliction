import argparse
import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassConfusionMatrix,
)


class LeafCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def create_dataloaders(dataset):
    data_loader = DataLoader(dataset, shuffle=True)
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    labels = np.array([dataset.targets[i] for i in range(len(dataset))])

    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    train_indices, temp_indices, _, temp_labels = train_test_split(
        np.arange(len(dataset)),
        labels,
        stratify=labels,
        test_size=(1 - train_size),
        random_state=42,
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        stratify=temp_labels,
        test_size=(test_size / (test_size + val_size)),
        random_state=42,
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
    test_loader = DataLoader(dataset, sampler=test_sampler)
    return train_loader, val_loader, test_loader


def load_dataset(directory_path):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=directory_path, transform=transform)
    return dataset


def compute_validation_metrics(model, validation_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    validation_metrics = {"loss": 0}
    with torch.no_grad():
        f1_score = MulticlassF1Score(num_classes=4, average="macro")
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            validation_metrics["loss"] += loss.item()
            f1_score.update(preds, labels)
    validation_metrics["loss"] /= len(validation_loader)
    validation_metrics["f1_score"] = f1_score.compute()
    model.train()
    return validation_metrics


def update_validation_metrics_history(
    model, validation_loader, validation_metrics_history
):
    new_validation_metrics = compute_validation_metrics(model, validation_loader)
    for name, value in new_validation_metrics.items():
        validation_metrics_history[name].append(value)


def early_stopping(
    state_dict, validation_loss, best_loss=None, counter=0, patience=5, min_delta=0
):
    if best_loss is None:
        best_loss = validation_loss
    elif validation_loss < best_loss - min_delta:
        best_loss = validation_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            torch.save(state_dict, "best_model.pth")
            return True, best_loss, counter
    return False, best_loss, counter


def train_model(train_loader, validation_loader, epochs, patience):
    model = LeafCNN()
    criterion = nn.CrossEntropyLoss()
    best_loss = None
    counter = 0
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    validation_metrics_history = {"f1_score": [], "loss": [], "accuracy": []}
    train_metrics_history = {"f1_score": [], "loss": [], "accuracy": []}
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(epoch, running_loss / len(train_loader), best_loss, i)
        train_metrics_history["loss"].append(running_loss / len(train_loader))
        update_validation_metrics_history(
            model, validation_loader, validation_metrics_history
        )
        state_dict = model.state_dict()
        stop, best_loss, counter = early_stopping(
            state_dict,
            validation_metrics_history["loss"][-1],
            best_loss=best_loss,
            counter=counter,
            patience=patience,
        )
        if stop:
            return validation_metrics_history, train_metrics_history
    torch.save(model.state_dict(), "best_model.pth")
    return validation_metrics_history, train_metrics_history


def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN on leaf images."
    )
    parser.add_argument("data_dir", type=str, help="Directory path to the dataset.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="Mode: train or test.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model file (required for test mode).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(dataset)

    if args.mode == "train":
        validation_history, train_history = train_model(
            train_loader, val_loader, args.epochs, args.patience
        )
    elif args.mode == "test":
        if not args.model_path:
            raise ValueError("Model path must be provided for test mode.")
        model = LeafCNN()
        model.load_state_dict(torch.load(args.model_path))
        evaluate_model(model, test_loader, dataset.classes)
