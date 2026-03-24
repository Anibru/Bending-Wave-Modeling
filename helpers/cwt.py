# helpers/cwt.py
"""CWT computation and power-spectrum preparation utilities."""

import numpy as np
import pywt


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


def run_cwt_and_ridge(
    b,
    r,
    wavenumbers,
    bandwidth_grid,
    target_cf,
    sampling_period,
    ridge_mask_r=None,
    ridge_mask_k=None,
    ridge_mask_r2=None,
    ridge_mask_k2=None,
):
    """Runs a complex Morlet CWT over a bandwidth grid and detects the power ridge.

    For each bandwidth parameter ``B`` in ``bandwidth_grid``, a complex Morlet
    wavelet ``cmor{B}-{target_cf}`` is constructed, the CWT is computed, and
    the dominant wavenumber at each radius position is found by taking the
    argmax of the power spectrum column (with optional wavenumber-band
    masking).  The result with the smoothest, most energy-concentrated ridge
    is selected as "best".

    The quality score minimized is::

        score = smoothness / energy_concentration

    where ``smoothness`` is the mean absolute first difference of the smoothed
    ridge, and ``energy_concentration`` is the fraction of total power in the
    top 10 % of pixels.

    Args:
        b (numpy.ndarray): 1-D input signal (e.g., normalized I/F brightness).
        r (numpy.ndarray): 1-D array of radial positions (km) corresponding
            to each sample of ``b``.
        wavenumbers (numpy.ndarray): 1-D array of wavenumbers in cycles/km
            over which to evaluate the CWT.
        bandwidth_grid (list[float]): Bandwidth parameters ``B`` to sweep.
            Each value generates a wavelet ``cmor{B}-{target_cf}``.
        target_cf (float): Central frequency of the Morlet wavelet in
            normalized units (typically ``morlet_w / (2 * pi)``).
        sampling_period (float): Sample spacing in km (``dr``), passed to
            ``pywt.cwt``.
        ridge_mask_r (tuple[float, float] | None): ``(rmin, rmax)`` radius
            range over which to apply the first wavenumber mask. If ``None``
            the mask is not applied. Defaults to ``None``.
        ridge_mask_k (tuple[float, float] | None): ``(kmin, kmax)`` wavenumber
            band (cycles/km) to suppress during ridge detection when inside
            ``ridge_mask_r``. Defaults to ``None``.
        ridge_mask_r2 (tuple[float, float] | None): Radius range for the
            optional second mask. Defaults to ``None``.
        ridge_mask_k2 (tuple[float, float] | None): Wavenumber band for the
            optional second mask. Defaults to ``None``.

    Returns:
        dict: A dictionary with the following keys:

            - ``"all_results"`` (list[tuple]): Per-bandwidth tuples of
              ``(B, score, power, k_est, k_est_smooth, wavelet, scales)``.
            - ``"B_best"`` (float): Bandwidth of the selected wavelet.
            - ``"score_best"`` (float): Quality score of the selected wavelet.
            - ``"power_best"`` (numpy.ndarray): CWT power array for the best
              wavelet, shape ``(n_scales, n_radii)``.
            - ``"k_est_best"`` (numpy.ndarray): Raw ridge wavenumber in
              rad/km, shape ``(n_radii,)``.
            - ``"k_est_smooth_best"`` (numpy.ndarray): Smoothed ridge
              wavenumber (21-sample boxcar), shape ``(n_radii,)``.
            - ``"wavelet_best"`` (str): PyWavelets wavelet string for the
              selected bandwidth.
            - ``"scales_best"`` (numpy.ndarray): CWT scales corresponding to
              ``power_best``.
    """
    results = []
    for B in bandwidth_grid:
        wavelet = f"cmor{B}-{target_cf:.6f}"
        cf = pywt.central_frequency(wavelet)
        scales = cf / (wavenumbers * sampling_period)
        coeffs, _ = pywt.cwt(b, scales, wavelet, sampling_period=sampling_period)
        power = np.abs(coeffs) ** 2

        ridge_idx = np.empty(power.shape[1], dtype=int)
        for j in range(power.shape[1]):
            rj = r[j]
            col = power[:, j]

            # first mask
            if ridge_mask_r is not None and ridge_mask_k is not None:
                (rmin, rmax) = ridge_mask_r
                (kmin, kmax) = ridge_mask_k
                if (rj >= rmin) and (rj <= rmax):
                    col = col.copy()
                    kband = (wavenumbers >= kmin) & (wavenumbers <= kmax)
                    col[kband] = -np.inf

            # second mask
            if ridge_mask_r2 is not None and ridge_mask_k2 is not None:
                (rmin2, rmax2) = ridge_mask_r2
                (kmin2, kmax2) = ridge_mask_k2
                if (rj >= rmin2) and (rj <= rmax2):
                    if col is power[:, j]:
                        col = col.copy()
                    kband2 = (wavenumbers >= kmin2) & (wavenumbers <= kmax2)
                    col[kband2] = -np.inf

            ridge_idx[j] = np.argmax(col)

        k_est = 2 * np.pi * wavenumbers[ridge_idx]

        # smooth ridge a bit
        win = 21
        pad = win // 2
        k_est_smooth = np.convolve(
            np.pad(k_est, (pad, pad), mode="edge"),
            np.ones(win) / win,
            mode="valid",
        )

        smoothness = np.mean(np.abs(np.diff(k_est_smooth)))
        flat = power.flatten()
        energy_concentration = (
            np.sum(np.sort(flat)[-int(0.1 * flat.size):]) / np.sum(flat)
        )
        score = smoothness / energy_concentration

        results.append((B, score, power, k_est, k_est_smooth, wavelet, scales))

    # pick best
    B_best, score_best, power_best, k_est_best, k_est_smooth_best, wavelet_best, scales_best = min(
        results, key=lambda x: x[1]
    )

    return {
        "all_results": results,
        "B_best": B_best,
        "score_best": score_best,
        "power_best": power_best,
        "k_est_best": k_est_best,
        "k_est_smooth_best": k_est_smooth_best,
        "wavelet_best": wavelet_best,
        "scales_best": scales_best,
    }
