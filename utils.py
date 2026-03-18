# utils.py
import numpy as np

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

def prepare_power(power, scales, *, mode="log", colnorm=False, scale_correct=True, eps=1e-12):
    """Prepares a CWT power spectrum for visualization.

    Applies optional scale correction (per-octave normalization), optional
    column-wise normalization, and a nonlinear intensity transform.

    Args:
        power (numpy.ndarray): 2-D array of shape ``(n_scales, n_times)``
            containing raw CWT power values.
        scales (numpy.ndarray): 1-D array of length ``n_scales`` containing
            the CWT scales corresponding to each row of ``power``.
        mode (str): Intensity transform to apply. ``"log"`` applies
            ``log10(P + eps)``; ``"asinh"`` applies an inverse-hyperbolic-sine
            stretch normalized to the 95th percentile. Defaults to ``"log"``.
        colnorm (bool): If ``True``, normalize each column by its maximum
            value so that the brightest feature in each time slice has unit
            power. Defaults to ``False``.
        scale_correct (bool): If ``True``, divide power by the scale to
            approximate a per-octave (cone-of-influence) correction.
            Defaults to ``True``.
        eps (float): Small constant added for numerical stability before
            division and logarithm. Defaults to ``1e-12``.

    Returns:
        numpy.ndarray: Transformed power array of the same shape as ``power``.
    """
    P = power.astype(float).copy()
    if scale_correct:
        P = P / (scales[:, None] + eps)
    if colnorm:
        col_max = np.nanmax(P, axis=0, keepdims=True) + eps
        P = P / col_max
    if mode == "log":
        P = np.log10(P + eps)
    elif mode == "asinh":
        s = np.nanpercentile(P, 95) + eps
        P = np.arcsinh(P / s)
    return P

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
