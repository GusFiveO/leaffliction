import cv2
import os
import numpy as np
from plantcv import plantcv as pcv
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Apply mask to an image.")
    parser.add_argument(
        "image_folder_path", type=str, help="Path to the input image folder."
    )
    return parser.parse_args()


def save_image(img_array, image_path):
    img = Image.fromarray(img_array)
    print(f"Image path: {image_path}")
    img.save(image_path)
    print(f"MASKED: {image_path}")


def convert_to_grayscale(img):
    gray = pcv.rgb2gray_lab(img, "a")
    # gray = pcv.rgb2gray_lab(img, "b")
    # gray = pcv.rgb2gray_cmyk(img, "c")
    return gray
    print(gray)


def apply_gaussian_blur(gray, ksize=11):
    blurred = pcv.gaussian_blur(gray, (ksize, ksize), 0)
    return blurred


def create_mask(image):
    pcv.params.sample_label = "plant"
    # fill the transormation dict with the transfornmations
    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale)
    # transformations["mask"] = create_mask(transformations["grayscale"])
    # binary = pcv.threshold.mean(
    #     # gray_img=gray, ksize=2500, offset=5, object_type="dark"
    #     gray_img=blurred,
    #     ksize=300,
    #     offset=5,
    #     object_type="light",
    # )
    binary = pcv.threshold.triangle(gray_img=blurred, object_type="dark")
    binary = pcv.erode(binary, ksize=5, i=1)
    binary = pcv.fill(binary, size=200)
    binary = pcv.fill_holes(binary)
    return binary


def apply_mask(image, mask):
    masked = pcv.apply_mask(image, mask, mask_color="white")
    return masked


def mask_images(image_folder_path):
    subfolders = [f.path for f in os.scandir(image_folder_path) if f.is_dir()]
    print(f"Subfolders found: {subfolders}")
    for subfolder in subfolders:
        images = [
            f.path
            for f in os.scandir(subfolder)
            if f.is_file() and f.name.endswith((".JPG", ".jpg", ".png"))
        ]
        for image_path in images:
            print(f"Processing image: {image_path}")
            img, imgpath, imgname = pcv.readimage(image_path)
            mask = create_mask(img)
            masked_image = apply_mask(img, mask)
            save_image(masked_image, os.path.join(imgpath, imgname))


if __name__ == "__main__":
    agrs = parse_args()
    image_folder_path = agrs.image_folder_path
    mask_images(image_folder_path)
