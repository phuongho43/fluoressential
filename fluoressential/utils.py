import contextlib
import os

import filetype
from natsort import natsorted


def makedirs(dp):
    """Create directories as needed for the desired directory path.

    Args:
        dp (str): directory path desired
    """
    with contextlib.suppress(OSError):
        os.makedirs(dp)


def list_img_files(dp):
    """List the filepath for all images in the specified directory.

    Args:
        dp (str): absolute path of images directory

    Returns:
        img_files (list): absolute path for each image file
    """
    img_files = []
    for fn in natsorted(os.listdir(dp), key=lambda y: y.lower()):
        fp = os.path.join(dp, fn)
        if filetype.is_image(fp):
            img_files.append(fp)
    return img_files
