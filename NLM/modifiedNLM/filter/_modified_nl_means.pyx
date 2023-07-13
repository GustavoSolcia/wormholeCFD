#cython: initializedcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
from _shared.fast_exp cimport _fast_exp
from libc.math cimport fabs
from _shared.fused_numerics cimport np_floats # import para programação genérica de floats

cnp.import_array()

cdef inline np_floats patch_distance_2d(np_floats [:, :, :] p1,
                                        np_floats [:, :, :] p2,
                                        np_floats [:, ::] w,
                                        Py_ssize_t s, np_floats var,
                                        Py_ssize_t n_channels) nogil:
    """
    Compute a Gaussian distance between two image patches.
    Parameters
    ----------
    p1 : 3-D array_like
        First patch, 2D image with last dimension corresponding to channels.
    p2 : 3-D array_like
        Second patch, 2D image with last dimension corresponding to channels.
    w : 2-D array_like
        Array of weights for the different pixels of the patches.
    s : Py_ssize_t
        Linear size of the patches.
    var_diff : np_floats
        The double of the expected noise variance.
    n_channels : Py_ssize_t
        The number of channels.
    Returns
    -------
    distance : np_floats
        Gaussian distance between the two patches
    Notes
    -----
    The returned distance is given by
    .. math::  \exp( -w ((p1 - p2)^2 - 2*var))
    """

    cdef Py_ssize_t i, j, channel
    cdef np_floats DISTANCE_CUTOFF = 5.0
    cdef np_floats tmp_diff = 0
    cdef np_floats distance = 0
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for channel in range(n_channels):
                tmp_diff = p1[i, j, channel] - p2[i, j, channel]
                distance += w[i, j] * (tmp_diff * tmp_diff)
    return _fast_exp(-max(0.0,distance))

cdef inline np_floats patch_distance_3d(np_floats [:, :, :] p1,
                                        np_floats [:, :, :] p2,
                                        np_floats [:, :, ::] w,
                                        Py_ssize_t s, np_floats var) nogil:
    """
    Compute a Gaussian distance between two image patches.
    Parameters
    ----------
    p1 : 3-D array_like
        First patch.
    p2 : 3-D array_like
        Second patch.
    w : 3-D array_like
        Array of weights for the different pixels of the patches.
    s : Py_ssize_t
        Linear size of the patches.
    var_diff : np_floats
        The double of the expected noise variance.
    Returns
    -------
    distance : np_floats
        Gaussian distance between the two patches
    Notes
    -----
    The returned distance is given by
    .. math::  \exp( -w ((p1 - p2)^2 - 2*var))
    """

    cdef Py_ssize_t i, j, k
    cdef np_floats DISTANCE_CUTOFF = 5.0
    cdef np_floats distance = 0
    cdef np_floats tmp_diff

    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for k in range(s):
                tmp_diff = p1[i, j, k] - p2[i, j, k]
                distance += w[i, j, k] * (tmp_diff * tmp_diff)
    return _fast_exp(-fabs(distance))

def _RICE_nl_means_denoising_2d(cnp.ndarray[np_floats, ndim=3] image, Py_ssize_t s,
                           Py_ssize_t d, double h, double var):
    """
    Perform non-local means denoising on 2-D RGB image
    Parameters
    ----------
    image : ndarray
        Input RGB image to be denoised
    s : Py_ssize_t, optional
        Size of patches used for denoising
    d : Py_ssize_t, optional
        Maximal distance in pixels where to search patches used for denoising
    h : np_floats, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.
    Notes
    -----
    This function operates on 2D grayscale and multichannel images.  For
    2D grayscale images, the input should be 3D with size 1 along the last
    axis.  The code is compatible with an arbitrary number of channels.
    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """

    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t n_row, n_col, n_channels
    n_row, n_col, n_channels = image.shape[0], image.shape[1], image.shape[2]
    cdef Py_ssize_t offset = s / 2
    cdef Py_ssize_t row, col, i, j, channel, i_start, i_end, j_start, j_end
    cdef np_floats[::1] new_values = np.zeros(n_channels, dtype=dtype)
    cdef np_floats[:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, ((offset, offset), (offset, offset), (0, 0)),
               mode='reflect'))
    cdef np_floats [:, :, ::1] result = np.empty_like(image)
    cdef np_floats new_value
    cdef np_floats weight_sum, weight

    cdef np_floats A = ((s - 1.) / 4.)
    cdef np_floats [::1] range_vals = np.arange(-offset, offset + 1,
                                                dtype=dtype)
    xg_row, xg_col = np.meshgrid(range_vals, range_vals, indexing='ij')
    cdef np_floats [:, ::1] w = np.ascontiguousarray(
        np.exp(-(xg_row * xg_row + xg_col * xg_col) / (2 * A * A)))
    w *= 1. / (n_channels * np.sum(w) * h *h )

    cdef np_floats [:, :, :] central_patch
    var *= 2

    # Iterate over rows, taking padding into account
    with nogil:
        for row in range(n_row):
            # Iterate over columns, taking padding into account
            i_start = row - min(d, row)
            i_end = row + min(d + 1, n_row - row)

            for col in range(n_col):
                # Initialize per-channel bins
                new_values[:] = 0
                # Reset weights for each local region
                weight_sum = 0

                central_patch = padded[row:row+s, col:col+s, :]
                j_start = col - min(d, col)
                j_end = col + min(d + 1, n_col - col)

                # Iterate over local 2d patch for each pixel
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        weight = patch_distance_2d[np_floats](
                            central_patch,
                            padded[i:i+s, j:j+s, :],
                            w, s, var, n_channels)

                        # Collect results in weight sum
                        weight_sum += weight
                        # Apply to each channel multiplicatively
                        for channel in range(n_channels):
                            new_values[channel] += weight * padded[i+offset,
                                                                    j+offset,
                                                                    channel] * padded[i+offset,
                                                                                        j+offset,
                                                                                        channel]

                # Normalize the result
                for channel in range(n_channels):
                  #  result[row, col, channel] = new_values[channel] / weight_sum
                    result[row, col, channel] = (new_values[channel] / weight_sum - var * var)**0.5

    return np.squeeze(np.asarray(result))

def _RICE_nl_means_denoising_3d(cnp.ndarray[np_floats, ndim=3] image,
                           Py_ssize_t s, Py_ssize_t d,
                           double h, double var):
    """
    Perform non-local means denoising on 3-D array
    Parameters
    ----------
    image : ndarray
        Input data to be denoised.
    s : int, optional
        Size of patches used for denoising.
    d : Py_ssize_t, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : np_floats, optional
        Cut-off distance (in gray levels).
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.
    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """

    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t n_pln, n_row, n_col
    n_pln, n_row, n_col = image.shape[0], image.shape[1], image.shape[2]
    cdef Py_ssize_t i_start, i_end, j_start, j_end, k_start, k_end
    cdef Py_ssize_t pln, row, col, i, j, k
    cdef Py_ssize_t offset = s / 2
    # padd the image so that boundaries are denoised as well
    cdef np_floats [:, :, :] padded = np.ascontiguousarray(
        np.pad(image, offset, mode='reflect'))
    cdef np_floats [:, :, :] result = np.empty_like(image)
    cdef np_floats new_value
    cdef np_floats weight_sum, weight

    cdef np_floats A = ((s - 1.) / 4.)
    cdef np_floats [::] range_vals = np.arange(-offset, offset + 1,
                                               dtype=dtype)
    xg_pln, xg_row, xg_col = np.meshgrid(range_vals, range_vals, range_vals,
                                         indexing='ij')
    cdef np_floats [:, :, ::1] w = np.ascontiguousarray(
        np.exp(-(xg_pln * xg_pln + xg_row * xg_row + xg_col * xg_col) /
               (2 * A * A)))
    w *= 1. / (np.sum(w) * h * h)

    cdef np_floats [:, :, :] central_patch
    var *= 2

    # Iterate over planes, taking padding into account
    with nogil:
        for pln in range(n_pln):
            i_start = pln - min(d, pln)
            i_end = pln + min(d + 1, n_pln - pln)
            # Iterate over rows, taking padding into account
            for row in range(n_row):
                j_start = row - min(d, row)
                j_end = row + min(d + 1, n_row - row)
                # Iterate over columns, taking padding into account
                for col in range(n_col):
                    k_start = col - min(d, col)
                    k_end = col + min(d + 1, n_col - col)

                    central_patch = padded[pln:pln+s, row:row+s, col:col+s]

                    new_value = 0
                    weight_sum = 0

                    # Iterate over local 3d patch for each pixel
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                weight = patch_distance_3d[np_floats](
                                    central_patch,
                                    padded[i:i+s, j:j+s, k:k+s],
                                    w, s, var)
                                # Collect results in weight sum
                                weight_sum += weight
#                                new_value += weight * padded[i+offset, <<< ORIGINAL
#                                                             j+offset,
#                                                             k+offset]
                                new_value += weight * padded[i+offset, # Rician Mod
                                                             j+offset,
                                                             k+offset] * padded[i+offset, 
                                                             j+offset,
                                                             k+offset]

                    # Normalize the result
#                    result[pln, row, col] = new_value / weight_sum <<< ORIGINAL
                    result[pln, row, col] = (new_value/weight_sum -var*var)**0.5  # Rician mod
            #        result[row, col, channel] = (new_values[channel] / weight_sum - var * var)**0.5

    return np.asarray(result)
