import sys
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
    # gray = pcv.rgb2gray_lab(img, "l")
    # gray = pcv.rgb2gray_hsv(img, "s")
    # gray = pcv.rgb2gray_hsv(img, "")
    return gray


def apply_gaussian_blur(gray, ksize=11):
    blurred = pcv.gaussian_blur(gray, (ksize, ksize), 0)
    return blurred


def create_mask(gray, threshold=125):
    # binary = pcv.threshold.otsu(gray, "light")
    # binary = pcv.threshold.otsu(gray, "dark")
    binary = pcv.threshold.gaussian(
        gray_img=gray, ksize=2500, offset=5, object_type="dark"
    )
    # binary = pcv.erode(binary,ksize=5, i=1)
    binary = pcv.fill(binary, size=200)
    binary_clean = pcv.fill_holes(binary)
    return binary_clean


def apply_mask(img, mask):
    masked = pcv.apply_mask(img, mask, mask_color="white")
    return masked


def analyze(img, mask):
    shape_img = pcv.analyze.size(img=img, labeled_mask=mask)
    return shape_img


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


def plot_image_transformations(img):
    grayscale_img = convert_to_grayscale(img)
    blurred_img = apply_gaussian_blur(grayscale_img)
    mask = create_mask(grayscale_img)
    # mask = create_mask(blurred_img)
    masked_img = apply_mask(img, mask)
    analyze_img = analyze(img, mask)

    pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)

    # Access data stored out from x_axis_pseudolandmarks
    bottom_landmarks = pcv.outputs.observations["plant"]["bottom_lmk"]["value"]
    top_landmarks = pcv.outputs.observations["plant"]["top_lmk"]["value"]
    center_landmarks = pcv.outputs.observations["plant"]["center_v_lmk"]["value"]

    fig, axs = plt.subplots(2, 3)
    axs[0][0].imshow(img)
    axs[0][0].set_title("Original image")

    axs[0][1].imshow(grayscale_img, cmap="gray")
    axs[0][1].set_title("Green magenta channel")

    axs[0][2].imshow(mask, cmap="gray")
    axs[0][2].set_title("Image mask")

    axs[1][0].imshow(masked_img)
    axs[1][0].set_title("Masked Image")

    axs[1][1].imshow(analyze_img)
    axs[1][1].set_title("Size and shape")

    axs[1][2].imshow(img)
    bottom_landmarks_x, bottom_landmarks_y = zip(*bottom_landmarks)
    axs[1][2].scatter(
        bottom_landmarks_x,
        bottom_landmarks_y,
        color="red",
        marker="o",
        label="Bottom Landmark",
    )
    top_landmarks_x, top_landmarks_y = zip(*top_landmarks)
    axs[1][2].scatter(
        top_landmarks_x,
        top_landmarks_y,
        color="blue",
        marker="o",
        label="Top Landmark",
    )
    center_landmarks_x, center_landmarks_y = zip(*center_landmarks)
    axs[1][2].scatter(
        center_landmarks_x,
        center_landmarks_y,
        color="green",
        marker="o",
        label="Center Landmark",
    )
    axs[1][2].set_title("Pseudo-landmarks")

    fig.suptitle("Image Transformations")
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])


def transform(file_path):
    img, imgpath, imgname = pcv.readimage(file_path)
    pcv.params.sample_label = "plant"
    plot_color_histogram(img)

    plot_colorspaces(img)

    plot_image_transformations(img)

    plt.show()


if __name__ == "__main__":
    file_path = sys.argv[1]
    transform(file_path)
