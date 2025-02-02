from pathlib import Path

import numpy as np
from filetype import is_image
from natsort import natsorted
from skimage import img_as_float
from skimage.filters import gaussian, threshold_li
from skimage.io import imread
from skimage.restoration import estimate_sigma


def list_img_fps(dp):
    """List the filepaths for all images in the specified directory.

    Args:
        dp (str): absolute path of images directory

    Returns:
        img_files (list): absolute paths for each image file
    """
    all_fps = natsorted(Path(dp).glob("*"))
    img_fps = [fp for fp in all_fps if is_image(fp)]
    return img_fps


def subtract_bgd(img, n_thr=2, gau_scale=1, vert_scale=2, ct_cutoff=0.1):
    """Subract background noise from fluorescence image.

    1. Generate an approximation of the background by thresholding away the bright features.
    2. Apply a gaussian filter to smooth out the remainder.
    3. Subtract this approximation from the raw image.

    Args:
        img_fp (str): absolute filepath to image file
        n_thr (int): number of thresholds to perform for removing bright features
        gau_scale (int): degree of gauss smoothing of bkg approx
        vert_scale (int): shifts bkg approx up and down
        ct_cutoff (float): cutoff value for detecting if image is low contrast

    Returns:
        img (2D array): processed image with background subtracted
        bgd (2D array): approximate image of background fluorescence
    """
    bgd = img.copy()
    shift = estimate_sigma(bgd)
    tha = img.copy()
    thr = np.percentile(tha, 99)
    for _ in range(n_thr):
        thr = threshold_li(tha)
        tha = tha[tha < thr]
    fbg = bgd[bgd > thr]
    contrast = np.round(np.std(fbg) / np.mean(fbg), 2)
    # print(contrast)
    if contrast >= ct_cutoff:
        bgd[bgd > thr] = thr
    bgd = gaussian(bgd, 25 * gau_scale) + shift * vert_scale
    bgd[bgd < 0] = 0
    img = img - bgd
    img[img < 0] = 0
    return img, bgd


def calc_imgs_cmax(imgs_dp):
    """Calculate the maximum pixel intensity value to use across all images in the directory.

    Args:
        img_dp (str): absolute path of images directory

    Returns:
        cmax (float): colorscale max value across all images
    """
    img_fps = list_img_fps(imgs_dp)
    max_vals = [np.percentile(img_as_float(imread(img_fp)), 99.99) for img_fp in img_fps]
    max_img_fp = img_fps[np.argmax(max_vals)]
    max_img = subtract_bgd(img_as_float(imread(max_img_fp)), n_thr=2, gau_scale=2, vert_scale=1)[0]
    cmax = np.percentile(max_img, 99.99)
    return cmax
