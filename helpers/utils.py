# helpers/utils.py
"""General-purpose numeric utilities for the Saturn ring analysis pipeline."""

import numpy as np


def to_native_endian(a):
    """Ensures a NumPy array uses native byte order for PyTorch compatibility.

    PyTorch does not accept non-native endian arrays.  This function is a
    no-op for non-NumPy objects and arrays that are already native-endian.

    Args:
        a: Input value.  If not a ``numpy.ndarray``, returned unchanged.

    Returns:
        The input array with native byte order, or the original object if
        it is not a ``numpy.ndarray``.
    """
    if not isinstance(a, np.ndarray):
        return a
    if not a.dtype.isnative:
        a = a.byteswap().view(a.dtype.newbyteorder('='))
    return a


def edges_from_centers(x):
    """Computes bin edges from an array of bin centers using midpoints.

    The first and last edges are extrapolated by half the first/last
    bin spacing, respectively.

    Args:
        x (numpy.ndarray): 1-D array of bin centers, assumed monotonically
            increasing.

    Returns:
        numpy.ndarray: 1-D array of length ``x.size + 1`` containing the
            bin edges.
    """
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = x[:-1] + dx / 2.0
    edges[0] = x[0] - dx[0] / 2.0
    edges[-1] = x[-1] + dx[-1] / 2.0
    return edges


def smooth_moving_average(x, W):
    """Applies a symmetric moving-average (boxcar) filter to a 1-D array.

    If ``W`` is even it is incremented by one to keep the kernel symmetric.
    Input arrays shorter than the window, or windows smaller than 3, are
    returned as a copy without modification.

    Args:
        x (numpy.ndarray): 1-D input signal.
        W (int): Nominal window width in samples.

    Returns:
        numpy.ndarray: Smoothed array of the same length as ``x``.
            Boundary effects are handled by NumPy's ``"same"`` convolution
            mode (zero-padding at the edges).
    """
    W = int(W)
    if W < 3:
        return x.copy()
    if W % 2 == 0:
        W += 1
    kernel = np.ones(W, float) / W
    return np.convolve(x, kernel, mode="same")


def safe_inv(x):
    """Safely computes the reciprocal of the mean of an array.

    Returns ``numpy.nan`` for ``None`` inputs, non-finite means, and
    means whose absolute value is at or below the floating-point threshold
    ``1e-12`` (to avoid division by zero).

    Args:
        x: Scalar, array-like, or ``None``. If array-like, the mean is taken
            before inverting.

    Returns:
        float: ``1 / mean(x)``, or ``numpy.nan`` if the input is invalid or
            the mean is effectively zero.
    """
    if x is None:
        return np.nan
    x = np.asarray(x, float)
    x_mean = np.nanmean(x)
    if not np.isfinite(x_mean) or abs(x_mean) <= 1e-12:
        return np.nan
    return 1.0 / x_mean
