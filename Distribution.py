import argparse

# matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt
import os
import sys


def compute_distribution(directory_path):
    distrib = []
    childs = [child for child in os.scandir(directory_path) if child.is_dir()]
    for child in childs:
        files = [child for child in os.scandir(child.path) if child.is_file()]
        nb_files = len(files)
        if nb_files == 0:
            print(f"Warning: {child.name} contains no files.")
            continue
        distrib.append((child.name, nb_files))
    return distrib


def plot_distribution(directory_path):
    basename = os.path.basename(directory_path)
    distrib = compute_distribution(directory_path)
    if not distrib:
        print(f"No valid subdirectories found in {directory_path}.")
        return
    labels, values = zip(*distrib)

    colors = [
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#ff7f0e",
    ]  # Blue, Green, Red, Orange

    fig, axs = plt.subplots(2, figsize=(8, 10))

    axs[0].pie(
        values, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90
    )

    axs[1].bar(labels, values, color=colors)

    fig.suptitle(f"{basename} distribution charts", fontsize=15)
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Distribution script")
    parser.add_argument(
        "directory_path", type=str, help="Path to the directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    directory_path = args.directory_path
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        sys.exit(1)
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        sys.exit(1)
    plot_distribution(directory_path)
