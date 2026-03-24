# helpers/physics.py
"""Physics model functions for Saturn ring bending-wave analysis."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from helpers.utils import smooth_moving_average


def phase_from_k(r, k_rad):
    """Integrates a spatially-varying wavenumber to produce cumulative phase.

    Uses the trapezoidal rule::

        psi[0] = 0
        psi[i] = psi[i-1] + 0.5 * (k[i] + k[i-1]) * (r[i] - r[i-1])

    Args:
        r (array-like): 1-D array of radial positions in km.
        k_rad (array-like): 1-D array of wavenumbers in rad/km, same length
            as ``r``.

    Returns:
        numpy.ndarray: 1-D array of cumulative phase in radians, same length
            as ``r``.  The first element is always 0.
    """
    r = np.asarray(r, float)
    k_rad = np.asarray(k_rad, float)
    psi = np.zeros_like(r)
    if len(r) > 1:
        dr_local = np.diff(r)
        psi[1:] = np.cumsum(0.5 * (k_rad[1:] + k_rad[:-1]) * dr_local)
    return psi


def theta_max_from_IF(r, k, IF, L=401, cotB_eff=1.0, IF_avg=None):
    """Estimates the local bending amplitude from I/F using sliding-window least-squares.

    Fits the photometric model

    .. code-block:: text

        y = IF - IF_avg ≈ a * cos(psi) + b * sin(psi)

    in a sliding window of length ``L`` centered at each radius, where
    ``psi`` is the cumulative phase from ``k``.  The local amplitude is

    .. code-block:: text

        theta_max(r) = sqrt(a^2 + b^2) / cotB_eff
        A_V(r)       = theta_max(r) / |k(r)|

    Args:
        r (array-like): 1-D array of radial positions in km.
        k (array-like): 1-D array of wavenumbers in rad/km.
        IF (array-like): 1-D array of observed normalized I/F values.
        L (int): Window length in samples (forced odd, minimum 3). A window
            approximately one wavelength long is recommended. Defaults to
            ``401``.
        cotB_eff (float | array-like): Cotangent of the effective elevation
            angle.  Can be a scalar or a 1-D array of length ``N``.
            Defaults to ``1.0``.
        IF_avg (float | None): Baseline I/F reference level.  If ``None``, the
            mean of ``IF`` is used. Defaults to ``None``.

    Returns:
        tuple:
            - **r** (numpy.ndarray): Radial positions (copy of input).
            - **theta_max** (numpy.ndarray): Local maximum tilt
              ``k * A_V`` at each radius.
            - **A_V** (numpy.ndarray): Local vertical amplitude in km;
              ``NaN`` where ``|k| == 0``.

    Raises:
        ValueError: If ``r``, ``k``, and ``IF`` do not all have the same
            length.
    """
    r = np.asarray(r, float)
    k = np.asarray(k, float)
    IF = np.asarray(IF, float)
    N = len(r)
    if not (len(k) == N and len(IF) == N):
        raise ValueError("r, k, IF must have the same length")

    if L % 2 == 0:
        L += 1
    if L < 3:
        L = 3
    half = L // 2

    if IF_avg is None:
        IF_avg = float(np.mean(IF))
    y = IF - IF_avg

    psi = np.zeros_like(r)
    if N > 1:
        dr_local = np.diff(r)
        psi[1:] = np.cumsum(0.5 * (k[1:] + k[:-1]) * dr_local)

    cosψ = np.cos(psi)
    sinψ = np.sin(psi)

    ypad = np.pad(y, (half, half), mode="reflect")
    cpad = np.pad(cosψ, (half, half), mode="reflect")
    spad = np.pad(sinψ, (half, half), mode="reflect")

    def rollsum(x, L):
        cs = np.cumsum(np.pad(x, (1, 0), mode="constant"))
        return cs[L:] - cs[:-L]

    Scc = rollsum(cpad * cpad, L)
    Sss = rollsum(spad * spad, L)
    Scs = rollsum(cpad * spad, L)
    Syc = rollsum(ypad * cpad, L)
    Sys = rollsum(ypad * spad, L)

    det = Scc * Sss - Scs * Scs
    det = np.where(np.abs(det) < 1e-30, 1e-30, det)

    a = (Sss * Syc - Scs * Sys) / det
    b = (Scc * Sys - Scs * Syc) / det

    fit_amp = np.sqrt(a * a + b * b)  # = [cot B_eff] * k * A_V
    theta_max = fit_amp / cotB_eff    # = k * A_V
    A_V = np.divide(theta_max, np.abs(k),
                    out=np.full_like(theta_max, np.nan),
                    where=(np.abs(k) > 0))
    return r, theta_max, A_V


def reconstruct_bending_wave(r, k_cycles, A_V,
                             m=1,
                             lambdas_deg=(0,),
                             phase0=0.0,
                             cotB_eff=1.0):
    """Constructs a bending-wave model at one or more longitude values.

    For each longitude ``lambda`` in ``lambdas_deg``, computes:

    .. code-block:: text

        phase(r) = psi(r) - m * lambda + phase0
        z(r)     = A_V(r) * sin(phase)
        tan_theta(r) = k(r) * A_V(r) * cos(phase)
        IF_model(r)  = 1 - cotB_eff * k(r) * A_V(r) * cos(phase)

    Args:
        r (array-like): 1-D array of radial positions in km.
        k_cycles (array-like): 1-D array of wavenumbers in cycles/km.
        A_V (array-like): 1-D array of vertical amplitudes in km.
        m (int): Azimuthal mode number. Defaults to ``1``.
        lambdas_deg (sequence[float]): Longitude values in degrees at which
            to evaluate the model. Defaults to ``(0,)``.
        phase0 (float): Constant phase offset in radians added to the
            cumulative phase. Defaults to ``0.0``.
        cotB_eff (float): Cotangent of the effective elevation angle used in
            the photometric model. Defaults to ``1.0``.

    Returns:
        dict: A dictionary with the following keys:

            - ``"r"`` (numpy.ndarray): Radial positions.
            - ``"lambdas_deg"`` (list[float]): Longitude values used.
            - ``"curves"`` (list[dict]): One entry per longitude, each with:

                - ``"lambda_deg"`` (float): Longitude in degrees.
                - ``"z"`` (numpy.ndarray): Vertical displacement in km.
                - ``"tan_theta"`` (numpy.ndarray): Local ring-plane tilt.
                - ``"IF_model"`` (numpy.ndarray): Modeled I/F.
    """
    r = np.asarray(r, float)
    k_rad = 2.0 * np.pi * np.asarray(k_cycles, float)
    A_V = np.asarray(A_V, float)

    psi_r = phase_from_k(r, k_rad) + phase0
    lam_rads = np.deg2rad(np.asarray(lambdas_deg, float))

    out = {"r": r, "lambdas_deg": list(lambdas_deg), "curves": []}
    for lam in lam_rads:
        phase = psi_r - m * lam
        z = A_V * np.sin(phase)
        tan_theta = k_rad * A_V * np.cos(phase)
        IF_model = 1.0 - cotB_eff * (k_rad * A_V * np.cos(phase))
        out["curves"].append({
            "lambda_deg": float(np.rad2deg(lam)),
            "z": z,
            "tan_theta": tan_theta,
            "IF_model": IF_model,
        })
    return out


def reconstruct_and_compare_IF(r, k_cycles, A_V, IF,
                                cotB_eff=1.0,
                                IF_avg=None,
                                k_smooth_frac=0.5):
    """Fits the photometric bending-wave model to observed I/F via least squares.

    Constructs the two-column design matrix

    .. code-block:: text

        Xc = -cotB_eff * k * A_V * cos(psi)
        Xs = -cotB_eff * k * A_V * sin(psi)

    and solves for coefficients ``(a, b)`` that minimize
    ``||y - a*Xc - b*Xs||^2`` where ``y = IF - IF_avg``.  The fitted effective
    photometric scale is ``cotB_eff_fit = cotB_eff * sqrt(a^2 + b^2)``.

    Args:
        r (array-like): 1-D array of radial positions in km.
        k_cycles (array-like): 1-D array of wavenumbers in rad/km (note: used
            directly as rad/km internally; the argument name is kept for
            legacy compatibility).
        A_V (array-like): 1-D array of vertical amplitudes in km.
        IF (array-like): 1-D array of observed normalized I/F values.
        cotB_eff (float | array-like): Cotangent of the effective elevation
            angle; can be a scalar or a 1-D array of length ``N``.
            Defaults to ``1.0``.
        IF_avg (float | None): Baseline I/F.  If ``None``, the mean of ``IF``
            is used. Defaults to ``None``.
        k_smooth_frac (float): Fraction of the local wavelength to use as the
            moving-average window when smoothing ``k`` before integrating
            phase.  Set to ``0`` or ``None`` to skip smoothing.
            Defaults to ``0.5``.

    Returns:
        dict: A dictionary containing:

            - ``"r"`` (numpy.ndarray): Radial positions.
            - ``"y"`` (numpy.ndarray): Observed residual ``IF - IF_avg``.
            - ``"y_hat"`` (numpy.ndarray): Model fit ``a*Xc + b*Xs``.
            - ``"IF_avg"`` (float): Baseline level used.
            - ``"psi"`` (numpy.ndarray): Cumulative phase from smoothed k.
            - ``"k_rad_s"`` (numpy.ndarray): Smoothed wavenumber used for
              phase integration.
            - ``"coeffs"`` (tuple[float, float]): Fitted coefficients
              ``(a, b)``.
            - ``"cotB_eff_input"`` (numpy.ndarray): Input cotangent profile.
            - ``"cotB_eff_fit"`` (numpy.ndarray): Per-radius fitted effective
              cotangent.
            - ``"cotB_eff_fit_mean"`` (float): Mean of ``cotB_eff_fit``.
            - ``"rmse"`` (float): Root-mean-square residual.
            - ``"r2"`` (float): Coefficient of determination.
            - ``"corr"`` (float): Pearson correlation between ``y`` and
              ``y_hat``.

    Raises:
        ValueError: If ``r``, ``k_cycles``, ``A_V``, and ``IF`` do not all
            have the same length.
    """
    r = np.asarray(r, float)
    k_rad = np.asarray(k_cycles, float)
    A_V = np.asarray(A_V, float)
    IF = np.asarray(IF, float)

    N = len(r)
    if not (len(k_rad) == N and len(A_V) == N and len(IF) == N):
        raise ValueError("r, k_cycles, A_V, IF must have same length")

    if IF_avg is None:
        IF_avg = float(np.mean(IF))
    y = IF - IF_avg

    # optional smoothing of k before phase
    if N > 3 and k_smooth_frac and k_smooth_frac > 0:
        dr = np.median(np.diff(r)) if N > 1 else 1.0
        lam_local = 2.0 * np.pi / (np.abs(k_rad) + 1e-12)
        Wk = int(np.clip(np.round((k_smooth_frac * np.median(lam_local)) / max(dr, 1e-12)),
                         5, max(5, N // 8)))
        if Wk % 2 == 0:
            Wk += 1
        k_rad_s = smooth_moving_average(k_rad, Wk)
    else:
        k_rad_s = k_rad

    # phase from smoothed k
    psi = phase_from_k(r, k_rad_s)

    # geometry to array
    cotB_eff = np.asarray(cotB_eff, float)
    if cotB_eff.size == 1:
        cotB_eff = np.full_like(r, float(cotB_eff))

    # model bases
    base = k_rad_s * A_V  # this is tan(theta)
    Xc = -cotB_eff * base * np.cos(psi)
    Xs = -cotB_eff * base * np.sin(psi)

    X = np.vstack([Xc, Xs]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeffs

    y_hat = a * Xc + b * Xs
    resid = y - y_hat

    rmse = np.sqrt(np.mean(resid ** 2))
    var = np.var(y)
    r2 = 1.0 - (np.var(resid) / var if var > 0 else np.nan)
    corr = (np.corrcoef(y, y_hat)[0, 1]
            if np.std(y) > 0 and np.std(y_hat) > 0 else np.nan)

    geom_used = cotB_eff * np.hypot(a, b)
    cotB_eff_fit_mean = float(np.nanmean(geom_used))

    return {
        "r": r,
        "y": y,
        "y_hat": y_hat,
        "IF_avg": IF_avg,
        "psi": psi,
        "k_rad_s": k_rad_s,
        "coeffs": (a, b),
        "cotB_eff_input": cotB_eff,
        "cotB_eff_fit": geom_used,
        "cotB_eff_fit_mean": cotB_eff_fit_mean,
        "rmse": rmse,
        "r2": r2,
        "corr": corr,
    }


def fit_stats(IF_obs, IF_model):
    """Computes goodness-of-fit statistics between observed and modeled I/F.

    Args:
        IF_obs (array-like): 1-D array of observed I/F values.
        IF_model (array-like): 1-D array of modeled I/F values, same length
            as ``IF_obs``.

    Returns:
        dict: A dictionary containing:

            - ``"rmse"`` (float): Root-mean-square error between ``IF_model``
              and ``IF_obs``.
            - ``"r2"`` (float): Coefficient of determination
              ``1 - Var(residual) / Var(observed)``.
            - ``"corr"`` (float): Pearson correlation coefficient between
              ``IF_obs`` and ``IF_model``.  ``NaN`` if either array has zero
              variance.
    """
    IF_obs = np.asarray(IF_obs, float)
    IF_model = np.asarray(IF_model, float)

    resid = IF_obs - IF_model
    rmse = np.sqrt(np.mean(resid ** 2))
    var = np.var(IF_obs)
    r2 = 1.0 - (np.var(resid) / var if var > 0 else np.nan)
    corr = (np.corrcoef(IF_obs, IF_model)[0, 1]
            if np.std(IF_obs) > 0 and np.std(IF_model) > 0 else np.nan)

    return {"rmse": rmse, "r2": r2, "corr": corr}
