import cv2
import numpy as np
from skimage import transform


def rotate_img(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    image = cv2.warpAffine(image, M, (nW, nH))

    image = cv2.resize(image, (w, h))

    # rotated_image = cv2.warpAffine(image, M, (nW, nH))

    # x_start = (nW - w) // 2
    # y_start = (nH - h) // 2

    # cropped_image = rotated_image[y_start : y_start + h, x_start : x_start + w]

    # return cropped_image
    return image


def shear_img(image, shear_factor_x, shear_factor_y):
    matrix = np.array(
        [[1, shear_factor_x, 0], [shear_factor_y, 1, 0], [0.0015, 0.0015, 1]]
    )  # Projective transformation matrix

    tform = transform.ProjectiveTransform(matrix=matrix)  # Create the transformation

    # Apply the transformation
    transformed_image = transform.warp(
        image, tform.inverse
    )  # Inverse to apply the transformation

    return transformed_image
    # return sheared_image
