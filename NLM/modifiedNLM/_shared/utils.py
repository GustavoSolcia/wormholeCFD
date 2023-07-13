import inspect
import warnings
import functools
import sys
import numpy as np
import numbers
from .dtype import img_as_float

def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.
    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Notes:
    ------
    * Input images with `float32` data type are not upcast.
    Returns
    -------
    image : ndarray
        Transformed version of the input.
    """
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        image = img_as_float(image)
    return image
