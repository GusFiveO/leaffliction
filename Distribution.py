import matplotlib

# matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys


def compute_distribution(directory_path):
    distrib = []
    childs = [child for child in os.scandir(directory_path) if child.is_dir()]
    for child in childs:
        files = [child for child in os.scandir(child.path) if child.is_file()]
        nb_files = len(files)
        distrib.append((child.name, nb_files))
    return distrib


def plot_distribution(directory_path):
    basename = os.path.basename(directory_path)
    distrib = compute_distribution(directory_path)
    labels, values = zip(*distrib)

    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]  # Blue, Green, Red, Orange

    fig, axs = plt.subplots(2, figsize=(8, 10))

    axs[0].pie(values, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90)

    axs[1].bar(labels, values, color=colors)

    fig.suptitle(f"{basename} distribution charts", fontsize=15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    directory_path = sys.argv[1]
    plot_distribution(directory_path)
