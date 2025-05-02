import argparse
import random

import matplotlib.pyplot as plt
from PIL import Image
import os
import torchvision.transforms as transforms

augmentations = {
    "Rotation": transforms.RandomRotation(45),
    "ResizedCrop": transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
    "Brightness": transforms.ColorJitter(brightness=0.7),
    "Contrast": transforms.ColorJitter(contrast=0.8),
    "Perspective": transforms.RandomPerspective(p=1.0),
    "Blur": transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5)),
}


def file_augmentation(file_path):
    dst_path = os.path.dirname(file_path)
    basename = os.path.splitext(os.path.basename(file_path))[0]
    image = Image.open(file_path).convert("RGB")

    fig, axes = plt.subplots(1, len(augmentations) + 1, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    for i, (transform_name, transform) in enumerate(augmentations.items()):
        augmented_image = transform(image)
        axes[i + 1].imshow(augmented_image)
        axes[i + 1].set_title(transform_name)
        output_path = os.path.join(dst_path, f"{basename}_{transform_name}.JPG")
        print(f"Augmented {output_path}")
        augmented_image.save(output_path)

    plt.show()


def list_subdirectories(directory):
    return [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]


def count_images_in_subdirectories(subdirectories):
    images_per_directory = []
    for directory in subdirectories:
        count = sum(1 for entry in os.scandir(directory) if entry.is_file())
        images_per_directory.append((directory, count))
    return images_per_directory


def random_augmentation(image):
    augmentation = random.choice(list(augmentations.values()))
    return augmentation(image)


def folder_augmentation(src):
    subdirectories = list_subdirectories(src)
    images_per_directory = count_images_in_subdirectories(subdirectories)
    max_images = max(count for _, count in images_per_directory)

    for subdirectory, count in images_per_directory:
        images = [
            os.path.join(subdirectory, f)
            for f in os.listdir(subdirectory)
            if os.path.isfile(os.path.join(subdirectory, f))
        ]
        while count < max_images:
            random_image_path = random.choice(images)
            image = Image.open(random_image_path).convert("RGB")
            augmented_image = random_augmentation(image)
            basename = os.path.splitext(os.path.basename(random_image_path))[0]
            augmented_image_path = os.path.join(
                subdirectory, f"{basename}_augmented_{count}.JPG"
            )
            augmented_image.save(augmented_image_path)
            count += 1
            print(f"Augmented {augmented_image_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Augmentation script")
    parser.add_argument("src", type=str, help="Path to the source directory or image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src = args.src
    if not os.path.exists(src):
        print(f"Source does not exist: {src}")
        exit(1)
    if os.path.isdir(src):
        folder_augmentation(src)
    elif os.path.isfile(src):
        file_augmentation(src)
