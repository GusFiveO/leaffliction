import sys
import matplotlib

from Augmentation_utils import rotate_img, shear_img

matplotlib.use("GTK3Agg")

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def augmentation(file_path):
    print(file_path)
    im = np.asarray(Image.open(file_path), dtype=np.uint8).copy()

    fig, axs = plt.subplots(3)

    rot_im = rotate_img(im, 50)
    shear_im = shear_img(im, 0.2, -0.01)

    axs[0].imshow(im)
    axs[1].imshow(rot_im)
    axs[2].imshow(shear_im)

    plt.show()


if __name__ == "__main__":
    file_path = sys.argv[1]
    augmentation(file_path)
