import os
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
from train import LeafCNN


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
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("weights_path", type=str, help="Path to the model weights file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    weights_path = args.weights_path
    if not os.path.exists(image_path):
        print(f"Image file does not exist: {image_path}")
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"Weights file does not exist: {weights_path}")
        sys.exit(1)

    class_names = ["Apple_Black_rot", "Apple_healthy", "Apple_rust", "Apple_scab"]

    model = load_model(weights_path)
    image, prediction = predict(model, image_path, class_names)
    plot_image(image, prediction)
