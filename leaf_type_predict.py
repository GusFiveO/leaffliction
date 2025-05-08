import argparse
import pandas as pd
import joblib
from plantcv import plantcv as pcv
import os
from transformation_utils import create_mask, analyze


# def create_mask(img):
#     gray = pcv.rgb2gray_lab(img, "b")
#     # gray = pcv.rgb2gray_lab(img, "a")
#     blurred = pcv.gaussian_blur(gray, (11, 11), 0)
#     mask = pcv.threshold.triangle(gray_img=blurred, object_type="dark")
#     mask = pcv.fill_holes(pcv.fill(pcv.erode(mask, ksize=5, i=1), size=200))
#     return mask


# def analyze(img, mask, label):
#     pcv.params.sample_label = label
#     pcv.analyze.size(img=img, labeled_mask=mask)


def normalize_features(features):
    # features = df.drop(["image_path", "in_bounds", "object_in_frame"], axis=1)
    normalized_features = (features - features.mean()) / features.std()
    return normalized_features


def extract_features(image_path):
    img, imgpath, imgname = pcv.readimage(image_path)
    mask = create_mask(img)
    analyze(img, mask, imgpath + "/" + imgname)
    obs_key = imgpath + "/" + imgname + "_1"

    obs = pcv.outputs.observations[obs_key]
    features = pd.DataFrame(
        [
            {
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
        ]
    )
    return features


def main():
    parser = argparse.ArgumentParser(description="Predict leaf type from image")
    parser.add_argument("image", type=str, help="Path to the image")
    parser.add_argument("model", type=str, help="Path to the .pkl model file")
    args = parser.parse_args()

    features = extract_features(args.image)
    print(f"Extracted features: {features}")

    pipeline = joblib.load(args.model)
    print(pipeline)
    # predictions = pipeline.predict(features)
    scaled_features = pipeline.named_steps["scaler"].transform(features)
    print("Scaled features:\n", scaled_features)

    predictions = pipeline.named_steps["classifier"].predict(scaled_features)
    print(f"Predictions: {predictions}")
    prediction = predictions[0]
    label = "Apple" if prediction == 0 else "Grape"
    print(f"Predicted leaf type: {label}")


if __name__ == "__main__":
    main()
