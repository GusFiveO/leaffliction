import os
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import sys
from LeafCNN import LeafCNN
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


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


def create_dataloader(dataset, batch_size=32, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


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


def load_model(weights_path):
    model = LeafCNN()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def predict(model, image_path, class_names):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]

    return image, prediction


def plot_image(image, prediction):
    plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Leaf Disease Prediction")
    parser.add_argument("target", type=str, help="Path to the image file or directory")
    parser.add_argument("weights_path", type=str, help="Path to the model weights file")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["apple", "grape"],
        help="Type of leaf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target = args.target
    weights_path = args.weights_path
    if not os.path.exists(target):
        print(f"Invalid target: {target}")
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"Weights file does not exist: {weights_path}")
        sys.exit(1)

    apple_class_names = ["Apple_Black_rot", "Apple_healthy", "Apple_rust", "Apple_scab"]
    graple_class_names = [
        "Grape_black_rot",
        "Grape_Esca",
        "Grape_healthy",
        "Grape_spot",
    ]
    class_names = {
        "apple": apple_class_names,
        "grape": graple_class_names,
    }

    model = load_model(weights_path)
    if os.path.isdir(target):
        test_dataset = load_dataset(args.target)
        test_loader = create_dataloader(test_dataset, shuffle=False)
        model = LeafCNN()
        model.load_state_dict(torch.load(args.weights_path))
        evaluate_model(model, test_loader, class_names[args.type])
    else:
        image, prediction = predict(model, target, class_names[args.type])
        plot_image(image, prediction)
