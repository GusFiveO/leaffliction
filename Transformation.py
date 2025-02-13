import sys
from PIL import Image
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import cv2

color_channel_map = {
    # LAB Color Space
    "lightness": "gray",  # L -> Perceived brightness
    "green-magenta": "magenta",  # A -> Green (-) to Red (+)
    "blue-yellow": "yellow",  # B -> Blue (-) to Yellow (+)
    # HSV Color Space
    "hue": "purple",  # H -> Represents color spectrum
    "saturation": "cyan",  # S -> Intensity of color
    "value": "orange",  # V -> Brightness (0 = black, 255 = full bright)
    # RGB Color Space
    "red": "red",
    "green": "green",
    "blue": "blue",
}


def convert_to_grayscale(img):
    gray = pcv.rgb2gray_lab(img, "a")
    return gray


def apply_gaussian_blur(gray, ksize=11):
    blurred = pcv.gaussian_blur(gray, (ksize, ksize), 0)
    return blurred


def create_mask(gray, threshold=125):
    binary = pcv.threshold.gaussian(
        gray_img=gray, ksize=2500, offset=5, object_type="dark"
    )
    # binary = pcv.erode(binary, ksize=5, i=1)
    binary = pcv.fill(binary, size=200)
    binary_clean = pcv.fill_holes(binary)
    return binary_clean


def apply_mask(img, mask):
    masked = pcv.apply_mask(img, mask, mask_color="white")
    return masked


def analyze(img, mask):
    shape_img = pcv.analyze.size(img=img, labeled_mask=mask)
    return shape_img


def draw_landmarks(img, landmarks, color):
    for x, y in landmarks:
        cv2.circle(
            img,
            (int(x), int(y)),
            radius=5,
            color=color,
            thickness=-1,
        )


def apply_landmarks(img, mask):

    pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)

    bottom_landmarks = pcv.outputs.observations["plant"]["bottom_lmk"]["value"]
    top_landmarks = pcv.outputs.observations["plant"]["top_lmk"]["value"]
    center_landmarks = pcv.outputs.observations["plant"]["center_v_lmk"]["value"]

    img_with_landmarks = img.copy()

    colors = {
        "bottom": (0, 0, 255),  # Red
        "top": (255, 0, 0),  # Blue
        "center": (0, 255, 0),  # Green
    }

    draw_landmarks(img_with_landmarks, bottom_landmarks, colors["bottom"])
    draw_landmarks(img_with_landmarks, top_landmarks, colors["top"])
    draw_landmarks(img_with_landmarks, center_landmarks, colors["center"])
    return img_with_landmarks


def extract_channel_histograms(img, color_space, channels, bins=50, range=(0, 255)):
    histograms = {}
    for ch, name in channels:
        if color_space:
            channel_img = color_space(rgb_img=img, channel=ch)
        else:
            channel_img = img[:, :, channels.index((ch, name))]

        count, bin_edges = np.histogram(channel_img, bins=bins, range=range)
        proportion = (count / count.sum()) * 100

        histograms[name] = (bin_edges[:-1], proportion)

    return histograms


def extract_LAB_channels(img, bins=50, range=(0, 255)):
    return extract_channel_histograms(
        img,
        pcv.rgb2gray_lab,
        [("l", "lightness"), ("a", "green-magenta"), ("b", "blue-yellow")],
        bins,
        range,
    )


def extract_HSV_channels(img, bins=50, range=(0, 255)):
    return extract_channel_histograms(
        img,
        pcv.rgb2gray_hsv,
        [("h", "hue"), ("s", "saturation"), ("v", "value")],
        bins,
        range,
    )


def extract_RGB_channels(img, bins=50, range=(0, 255)):
    return extract_channel_histograms(
        img, None, [("r", "red"), ("g", "green"), ("b", "blue")], bins, range
    )


def plot_color_histogram(img):
    rgb_channels = extract_RGB_channels(img)
    lab_channels = extract_LAB_channels(img)
    hsv_channels = extract_HSV_channels(img)

    ax = plt.subplot()
    for name, hist_data in rgb_channels.items():
        ax.plot(
            hist_data[0], hist_data[1], label=name, c=color_channel_map[name], alpha=0.7
        )
    for name, hist_data in lab_channels.items():
        ax.plot(
            hist_data[0], hist_data[1], label=name, c=color_channel_map[name], alpha=0.7
        )
    for name, hist_data in hsv_channels.items():
        ax.plot(
            hist_data[0], hist_data[1], label=name, c=color_channel_map[name], alpha=0.7
        )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Color channels distribution")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Pixel proportion (%)")


def plot_colorspaces(img):
    cs = pcv.visualize.colorspaces(rgb_img=img, original_img=False)
    fig, ax = plt.subplots()
    ax.set_title("Colorspaces")
    ax.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )
    ax.imshow(cs)


def save_image(img_array, name, dst):
    new_image_path = os.path.join(dst, name)
    os.makedirs(dst, exist_ok=True)
    img = Image.fromarray(img_array)
    img.save(new_image_path)
    print(f"SAVED: {new_image_path}")


def save_transformation(img, transformation_name, original_name, dst_dir):
    if dst_dir is None:
        return
    basename, ext = original_name.split(".", 1)
    new_image_name = f"{basename}_{transformation_name}.{ext}"
    save_image(img, new_image_name, dst_dir)


def plot_image_transformations(img, imgname, dst_dir):
    grayscale_img = convert_to_grayscale(img)
    blurred_img = apply_gaussian_blur(grayscale_img)
    mask = create_mask(grayscale_img)
    # mask = create_mask(blurred_img)
    masked_img = apply_mask(img, mask)
    analyze_img = analyze(img, mask)
    landmarks_img = apply_landmarks(img, mask)

    fig, axs = plt.subplots(2, 3)
    axs[0][0].imshow(img)
    axs[0][0].set_title("Original image")

    axs[0][1].imshow(grayscale_img, cmap="gray")
    axs[0][1].set_title("Green magenta channel")
    save_transformation(grayscale_img, "Grayscale", imgname, dst_dir)

    axs[0][2].imshow(mask, cmap="gray")
    axs[0][2].set_title("Image mask")
    save_transformation(mask, "Mask", imgname, dst_dir)

    axs[1][0].imshow(masked_img)
    axs[1][0].set_title("Masked Image")
    save_transformation(masked_img, "Masked", imgname, dst_dir)

    axs[1][1].imshow(analyze_img)
    axs[1][1].set_title("Size and shape")
    save_transformation(analyze_img, "Size&Shape", imgname, dst_dir)

    axs[1][2].imshow(landmarks_img)
    axs[1][2].set_title("Pseudo-landmarks")
    save_transformation(landmarks_img, "PsLandmarks", imgname, dst_dir)

    fig.suptitle("Image Transformations")
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])


def transform(file_path, dst):
    img, imgpath, imgname = pcv.readimage(file_path)
    pcv.params.sample_label = "plant"
    plot_color_histogram(img)

    plot_colorspaces(img)

    plot_image_transformations(img, imgname, dst)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("-dst", type=str, required=False)
    args = parser.parse_args()

    transform(args.file_path, args.dst)
