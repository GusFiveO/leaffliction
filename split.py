import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import argparse
import os

from tqdm import tqdm


def load_dataset(directory_path):
    transform = transforms.Compose(
        [
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(
        root=directory_path, transform=transform
    )
    return dataset


def check_split(targets, indices):
    """
    Checks if the split is stratified
    """
    unique_labels = np.unique(targets)
    for label in unique_labels:
        label_indices = np.where(targets == label)[0]
        selected_indices = np.isin(label_indices, indices)
        print(
            f"Label {label}: {len(label_indices)} \
in total, {np.sum(selected_indices)} selected"
        )
        print(f"Ratio: {np.sum(selected_indices) / len(label_indices)}")
        if len(label_indices) != len(selected_indices):
            print(f"Label {label} is not stratified in the split.")
            return False
    return True


def split_dataset(
    original_dir, augmented_dir, train_size=0.7, val_size=0.15, test_size=0.15
):
    """
    Splits the original dataset into subdirectories
    """

    dataset = load_dataset(original_dir)
    print(f"Loaded dataset with {len(dataset)} images from {original_dir}")
    labels = dataset.targets
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

    save_dataset(dataset, train_indices, os.path.join(augmented_dir, "train"))
    save_dataset(dataset, val_indices, os.path.join(augmented_dir, "val"))
    save_dataset(dataset, test_indices, os.path.join(augmented_dir, "test"))


def save_dataset(dataset, indices, directory):
    """
    Saves the dataset to a directory
    """
    os.makedirs(directory, exist_ok=True)
    for index in tqdm(indices, desc=f"Saving {directory}"):
        image, label = dataset[index]
        label_dir = os.path.join(directory, dataset.classes[label])
        os.makedirs(label_dir, exist_ok=True)
        torchvision.utils.save_image(
            image, os.path.join(label_dir, f"{index}.png")
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Augmentation script")
    parser.add_argument("-src", type=str, help="Path to the original dataset")
    parser.add_argument("-dst", type=str, help="Path to the augmented dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(args.src, args.dst)


if __name__ == "__main__":
    main()
