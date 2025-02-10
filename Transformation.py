import sys
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import cv2


def convert_to_grayscale(img):
    # gray = pcv.rgb2gray_lab(img, "l")
    gray = pcv.rgb2gray(img)
    return gray


def apply_gaussian_blur(gray, ksize=5):
    blurred = pcv.gaussian_blur(gray, (ksize, ksize), 0)
    return blurred


def transform(file_path):
    img, imgpath, imgname = pcv.readimage(file_path)
    pcv.params.debug = "print"

    gray_scale_img = convert_to_grayscale(img)
    blurred_img = apply_gaussian_blur(gray_scale_img)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[1].imshow(gray_scale_img, cmap="gray")
    axs[2].imshow(blurred_img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    file_path = sys.argv[1]
    transform(file_path)
