import argparse
from tqdm import tqdm
import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassF1Score,
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


def create_dataloader(dataset, batch_size=32, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# def create_dataloaders(train_dataset, valid_dataset):
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
#     return train_loader, val_loader, None


def load_dataset(dir):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=dir, transform=transform)
    return dataset


# def load_dataset(train_dir, valid_dir):
#     transform = transforms.Compose(
#         [
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     )
#     train_dataset = torchvision.datasets.ImageFolder(
#         root=train_dir, transform=transform
#     )
#     valid_dataset = torchvision.datasets.ImageFolder(
#         root=valid_dir, transform=transform
#     )
#     return train_dataset, valid_dataset


def compute_validation_metrics(model, validation_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    validation_metrics = {"loss": 0}
    with torch.no_grad():
        accuracy = MulticlassAccuracy(num_classes=4)
        f1_score = MulticlassF1Score(num_classes=4, average="macro")
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            validation_metrics["loss"] += loss.item()
            f1_score.update(preds, labels)
            accuracy.update(preds, labels)
    validation_metrics["loss"] /= len(validation_loader)
    validation_metrics["f1_score"] = f1_score.compute()
    validation_metrics["accuracy"] = accuracy.compute()
    model.train()
    return validation_metrics


def update_metrics_history(model, validation_loader, validation_metrics_history):
    new_validation_metrics = compute_validation_metrics(model, validation_loader)
    for name, value in new_validation_metrics.items():
        validation_metrics_history[name].append(value)


def early_stopping(
    model_path,
    state_dict,
    validation_accuracy,
    epoch,
    best_accuracy=None,
    best_epoch=None,
    counter=0,
    patience=5,
    min_delta=0,
):
    if best_accuracy is None:
        best_epoch = epoch
        best_accuracy = validation_accuracy
    elif validation_accuracy > best_accuracy - min_delta:
        best_epoch = epoch
        best_accuracy = validation_accuracy
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            torch.save(state_dict, model_path)
            return True, best_accuracy, best_epoch, counter
    return False, best_accuracy, best_epoch, counter


def log_metrics(validation_metrics_history, train_metrics_history):
    print(f"Training Loss: {train_metrics_history['loss'][-1]}")
    print(f"Validation Loss: {validation_metrics_history['loss'][-1]}")
    print(f"Validation F1 Score: {validation_metrics_history['f1_score'][-1]}")
    print(f"Validation Accuracy: {validation_metrics_history['accuracy'][-1]}")


def plot_metrics(validation_metrics_history, train_metrics_history, class_names):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(validation_history["loss"], label="Validation Loss", color="blue")
    axs[0].plot(train_history["loss"], label="Training Loss", color="green")
    axs[1].plot(
        validation_history["f1_score"], label="Validation F1 Score", color="orange"
    )
    axs[1].plot(train_history["f1_score"], label="Training F1 Score", color="red")
    axs[2].plot(
        validation_history["accuracy"], label="Validation Accuracy", color="purple"
    )
    axs[2].plot(train_history["accuracy"], label="Training Accuracy", color="brown")
    axs[0].set_title("Loss")
    axs[1].set_title("Validation F1 Score")
    axs[2].set_title("Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[2].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("F1 Score")
    axs[2].set_ylabel("Accuracy")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.tight_layout()
    plt.show()


def train_model(train_loader, validation_loader, epochs, patience, model_path):
    model = LeafCNN()
    criterion = nn.CrossEntropyLoss()
    best_accuracy = None
    best_epoch = None
    counter = 0
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    validation_metrics_history = {"f1_score": [], "loss": [], "accuracy": []}
    train_metrics_history = {"f1_score": [], "loss": [], "accuracy": []}
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"), 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(epoch, running_loss / len(train_loader), best_loss, i)

        update_metrics_history(model, train_loader, train_metrics_history)
        update_metrics_history(model, validation_loader, validation_metrics_history)
        log_metrics(validation_metrics_history, train_metrics_history)
        state_dict = model.state_dict()
        stop, best_accuracy, best_epoch, counter = early_stopping(
            model_path,
            state_dict,
            validation_metrics_history["accuracy"][-1],
            epoch,
            best_accuracy=best_accuracy,
            best_epoch=best_epoch,
            counter=counter,
            patience=patience,
        )
        if stop:
            print(
                f"Early stopping at epoch {best_epoch} with best accuracy {best_accuracy}"
            )
            return validation_metrics_history, train_metrics_history
    print(f"Early stopping at epoch {best_epoch} with best accuracy {best_accuracy}")
    torch.save(model.state_dict(), model_path)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN on leaf images."
    )
    parser.add_argument(
        "train_dir", type=str, help="Directory path to the train dataset."
    )
    parser.add_argument(
        "valid_dir", type=str, help="Directory path to the validation dataset."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Set to True to run the test mode.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth",
        help="Path to the trained model file (required for test mode).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.train_dir):
        print(f"Train directory does not exist: {args.train_dir}")
        sys.exit(1)
    if not os.path.isdir(args.train_dir):
        print(f"Train directory is not a valid directory: {args.train_dir}")
        sys.exit(1)
    if not os.path.exists(args.valid_dir):
        print(f"Validation directory does not exist: {args.valid_dir}")
        sys.exit(1)
    if not os.path.isdir(args.valid_dir):
        print(f"Validation directory is not a valid directory: {args.valid_dir}")
        sys.exit(1)
    if args.test_dir and not args.model_path:
        print("Model path must be provided for test mode.")
        sys.exit(1)

    train_dataset = load_dataset(args.train_dir)
    valid_dataset = load_dataset(args.valid_dir)
    train_loader = create_dataloader(train_dataset)
    val_loader = create_dataloader(valid_dataset, shuffle=False)
    # train_loader, val_loader, test_loader = create_dataloaders(
    #     train_dataset, valid_dataset
    # )

    # validation_history, train_history = train_model(
    #     train_loader, val_loader, args.epochs, args.patience, args.model_path
    # )
    # plot_metrics(validation_history, train_history, train_dataset.classes)

    if args.test_dir:
        test_dataset = load_dataset(args.test_dir)
        test_loader = create_dataloader(test_dataset, shuffle=False)
        model = LeafCNN()
        model.load_state_dict(torch.load(args.model_path))
        evaluate_model(model, test_loader, train_dataset.classes)
