import os
import argparse

import random
import shutil
from PIL import Image
import torchvision.transforms as transforms


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
    augmentations = {
        "Rotation": transforms.RandomRotation(45),
        "ResizedCrop": transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
        "Brightness": transforms.ColorJitter(brightness=0.7),
        "Contrast": transforms.ColorJitter(contrast=0.8),
        "Perspective": transforms.RandomPerspective(p=1.0),
        "Blur": transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5)),
    }
    augmentation = random.choice(list(augmentations.values()))
    return augmentation(image)


def augment_dataset(original_dir, augmented_dir):
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
    shutil.copytree(original_dir, augmented_dir)

    subdirectories = list_subdirectories(augmented_dir)
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
            # images.append(augmented_image_path)
            count += 1
            print(f"Augmented {augmented_image_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Augmentation script")
    parser.add_argument("original_dir", type=str, help="Path to the original dataset")
    parser.add_argument("augmented_dir", type=str, help="Path to the augmented dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    original_dir = args.original_dir
    augmented_dir = args.augmented_dir
    if not os.path.exists(original_dir):
        print(f"Original directory does not exist: {original_dir}")
        exit(1)
    # if not os.path.exists(augmented_dir):
    #     print(f"Augmented directory does not exist: {augmented_dir}")
    #     exit(1)
    # original_dataset_dir = os.path.expanduser(original_dir)
    # augmented_dataset_dir = os.path.expanduser(augment_dataset)
    # augment_dataset(original_dataset_dir, augmented_dataset_dir)
    augment_dataset(original_dir, augmented_dir)
    # original_apple_dataset_dir = os.path.expanduser("~/sgoinfre/leaves/images/Apple/")
    # augmented_apple_dataset_dir = os.path.expanduser(
    #     "~/sgoinfre/leaves/augmented_dataset/Apple/"
    # )
    # augment_dataset(original_apple_dataset_dir, augmented_apple_dataset_dir)

    # original_grape_dataset_dir = os.path.expanduser("~/sgoinfre/leaves/images/Grape/")
    # augmented_grape_dataset_dir = os.path.expanduser(
    #     "~/sgoinfre/leaves/augmented_dataset/Grape/"
    # )
    # augment_dataset(original_grape_dataset_dir, augmented_grape_dataset_dir)
