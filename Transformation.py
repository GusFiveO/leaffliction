import sys
import numpy as np
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import cv2


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


def extract_LAB_channels(img):
    l_channel = pcv.rgb2gray_lab(rgb_img=img, channel="l")
    a_channel = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    b_channel = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    return {"l": l_channel, "a": a_channel, "b": b_channel}


def extract_HSV_channels(img):
    h_channel = pcv.rgb2gray_hsv(rgb_img=img, channel="h")
    s_channel = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    v_channel = pcv.rgb2gray_hsv(rgb_img=img, channel="v")
    return {"h": h_channel, "s": s_channel, "v": v_channel}


def extract_RGB_channels(img):
    # [x for xs in xss for x in xs]
    r_channel = [r for chan in img[:, :, 0] for r in chan]
    g_channel = [r for chan in img[:, :, 1] for r in chan]
    b_channel = [r for chan in img[:, :, 2] for r in chan]
    r_count, r_bins = np.histogram(r_channel, 50)
    r_proportion = (r_count / r_count.sum()) * 100
    ax = plt.subplot()
    ax.plot(r_bins[:-1], r_proportion)
    # ax.hist(g_channel)
    # ax.hist(b_channel)
    plt.plot()
    return {"r": r_channel, "g": g_channel, "b": b_channel}


def transform(file_path):
    img, imgpath, imgname = pcv.readimage(file_path)
    pcv.params.debug = "print"
    pcv.params.sample_label = "plant"
    cs = pcv.visualize.colorspaces(rgb_img=img, original_img=False)
    pcv.plot_image(cs)
    grayscale_img = convert_to_grayscale(img)
    blurred_img = apply_gaussian_blur(grayscale_img)
    mask = create_mask(grayscale_img)
    # mask = create_mask(blurred_img)
    masked_img = apply_mask(img, mask)
    analyze_img = analyze(img, mask)

    extract_RGB_channels(img)

    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)

    # Access data stored out from x_axis_pseudolandmarks
    bottom_landmarks = pcv.outputs.observations["plant"]["bottom_lmk"]["value"]
    top_landmarks = pcv.outputs.observations["plant"]["top_lmk"]["value"]
    center_landmarks = pcv.outputs.observations["plant"]["center_v_lmk"]["value"]

    fig, axs = plt.subplots(2, 3)
    axs[0][0].imshow(img)
    # axs[0][1].imshow(blurred_img, cmap="gray")
    axs[0][1].imshow(grayscale_img, cmap="gray")
    axs[0][2].imshow(mask, cmap="gray")
    axs[1][0].imshow(masked_img)
    axs[1][1].imshow(analyze_img)
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

    plt.show()


if __name__ == "__main__":
    file_path = sys.argv[1]
    transform(file_path)
