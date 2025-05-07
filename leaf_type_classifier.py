import joblib
import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_files

from plantcv import plantcv as pcv


def parse_args():
    parser = argparse.ArgumentParser(description="Leaf Type Classifier")
    parser.add_argument(
        "dataset", type=str, help="Path to the dataset directory"
    )
    return parser.parse_args()


def load_dataset(dataset_path):
    dataset = load_files(dataset_path, load_content=False, shuffle=True)
    return dataset


def convert_to_grayscale(img):
    return pcv.rgb2gray_lab(img, "a")


def apply_gaussian_blur(gray, ksize=11):
    return pcv.gaussian_blur(gray, (ksize, ksize), 0)


def create_mask(image):
    pcv.params.sample_label = "plant"
    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale)
    binary = pcv.threshold.triangle(gray_img=blurred, object_type="dark")
    binary = pcv.erode(binary, ksize=5, i=1)
    binary = pcv.fill(binary, size=200)
    binary = pcv.fill_holes(binary)
    return binary


def analyze(img, mask, label):
    pcv.params.sample_label = label
    shape_img = pcv.analyze.size(img=img, labeled_mask=mask)
    return shape_img


def create_dataframe(dataset):
    samples = []
    for directory, label in zip(dataset.filenames, dataset.target):
        images = [f.path for f in os.scandir(directory)]
        for image_path in images:
            img, imgpath, imgname = pcv.readimage(image_path.decode())
            mask = create_mask(img)
            obs_key = imgpath + "/" + imgname
            analyze(img, mask, obs_key)
            obs = pcv.outputs.observations[obs_key + "_1"]
            sample = {
                "image_path": image_path,
                "label": label,
                "area": obs["area"]["value"],
                "convex_hull_area": obs["convex_hull_area"]["value"],
                "solidity": obs["solidity"]["value"],
                "perimeter": obs["perimeter"]["value"],
                "width": obs["width"]["value"],
                "height": obs["height"]["value"],
                "longest_path": obs["longest_path"]["value"],
                "center_of_mass_x": obs["center_of_mass"]["value"][0],
                "center_of_mass_y": obs["center_of_mass"]["value"][1],
                "convex_hull_vertices": obs["convex_hull_vertices"]["value"],
            }
            samples.append(sample)
    return pd.DataFrame(samples)


def plot_correlation_matrix(features):
    corr = features.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    plt.show()


def evaluate_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(
        "Classification Report:\n", classification_report(y_test, y_test_pred)
    )


def train_pipeline(X, y):
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Build pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("\nValidation Results:")
    evaluate_model(pipeline, X_val, y_val)

    print("\nTest Results:")
    evaluate_model(pipeline, X_test, y_test)

    return pipeline


def save_model(model, path="leaf_classifier_pipeline.pkl"):
    joblib.dump(model, path)


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset
    dataset = load_dataset(dataset_path)

    if not os.path.exists("leaf_data.csv"):
        print("Extracting features, this may take a while...")
        df = create_dataframe(dataset)
        df.to_csv("leaf_data.csv", index=False)
    else:
        df = pd.read_csv("leaf_data.csv")

    print("Dataframe loaded:")
    # print(df.head())

    # Remove non-feature columns
    drop_cols = ["image_path"]
    X = df.drop(columns=drop_cols + ["label"])
    y = df["label"]

    # Optional: visualize correlation
    plot_correlation_matrix(X)

    pipeline = train_pipeline(X, y)
    save_model(pipeline)
    print("Model saved as 'leaf_classifier_pipeline.pkl'")
