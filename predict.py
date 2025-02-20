import torch
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <image_path> <weights_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    weights_path = sys.argv[2]

    class_names = ["Apple_Black_rot", "Apple_healthy", "Apple_rust", "Apple_scab"]

    model = load_model(weights_path)
    image, prediction = predict(model, image_path, class_names)
    plot_image(image, prediction)
