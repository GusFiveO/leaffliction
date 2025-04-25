import argparse

# matplotlib.use("GTK3Agg")

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from PIL import Image
import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def augmentation(file_path):
    parent_dir = os.path.dirname(file_path)
    basename = os.path.splitext(os.path.basename(file_path))[0]
    image = Image.open(file_path).convert("RGB")

    augmentations = {
        "Rotation": transforms.RandomRotation(45),
        "ResizedCrop": transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
        "Brightness": transforms.ColorJitter(brightness=0.7),
        "Contrast": transforms.ColorJitter(contrast=0.8),
        "Perspective": transforms.RandomPerspective(p=1.0),
        "Blur": transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5)),
    }

    fig, axes = plt.subplots(1, len(augmentations) + 1, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    for i, (name, transform) in enumerate(augmentations.items()):
        augmented_image = transform(image)
        axes[i + 1].imshow(augmented_image)
        axes[i + 1].set_title(name)
        output_path = os.path.join(parent_dir, f"{basename}_{name}.JPG")
        print(output_path)
        augmented_image.save(output_path)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Augmentation script")
    parser.add_argument("file_path", type=str, help="Path to the image file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        exit(1)
    augmentation(file_path)
