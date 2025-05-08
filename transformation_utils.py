from plantcv import plantcv as pcv


def convert_to_grayscale(img):
    return pcv.rgb2gray_lab(img, "a")
    # return pcv.rgb2gray_lab(img, "b")
    # return pcv.rgb2gray_lab(img, "l")


def apply_gaussian_blur(gray, ksize=11):
    return pcv.gaussian_blur(gray, (ksize, ksize), 0)


def create_mask(image):
    pcv.params.sample_label = "plant"
    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale)
    binary = pcv.threshold.triangle(gray_img=blurred, object_type="dark")

    # binary = pcv.threshold.triangle(gray_img=blurred)
    # egi = pcv.spectral_index.egi(rgb_img=image)
    # egi = pcv.spectral_index.gli(image)
    # binary = pcv.threshold.binary(
    #     gray_img=egi.array_data, threshold=0, object_type="light"
    # )
    binary = pcv.erode(binary, ksize=5, i=1)
    binary = pcv.fill(binary, size=200)
    binary = pcv.fill_holes(binary)
    return binary


def analyze(img, mask, label):
    pcv.params.sample_label = label
    shape_img = pcv.analyze.size(img=img, labeled_mask=mask)
    return shape_img
