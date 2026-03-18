# 02_LS_wave_analysis.py

import numpy as np
import matplotlib
# matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import os

import config
from reconstruction import (
    theta_max_from_IF,
    reconstruct_bending_wave,
    plot_bending_wave,
    reconstruct_and_compare_IF,
    plot_IF_fit,
)

os.makedirs(config.PLOTS_DIR, exist_ok=True)

def main():
    """Performs least-squares bending-wave analysis for waves 1 and 2 only (excludes wave 5).

    Identical to ``LS_wave_analysis.main()`` except that the wave 5
    segment is omitted entirely.  Intended for images or configurations
    where the wave 5 region is not usable.

    Execution steps:

    1. Load ``r``, ``b_norm``, and ``k_est_smooth_best`` from
       ``config.CWT_FILE``.
    2. Build a constant ``cotB_eff`` profile from ``config.B_EFF``.
    3. Mask out wave 1 and wave 2 radial segments.
    4. Compute per-segment sliding-window amplitude ``A_V`` for waves 1 and
       2 using :func:`reconstruction.theta_max_from_IF`.
    5. Plot and save ``A_V`` vs. radius for both waves.
    6. Reconstruct the bending-wave geometry and plot via
       :func:`reconstruction.reconstruct_bending_wave`.
    7. Fit the photometric model via
       :func:`reconstruction.reconstruct_and_compare_IF` and report RMSE,
       R², correlation, and fitted ``B_eff`` for each wave.
    8. Plot and save the I/F fit comparison panels.

    Side effects:
        Creates ``config.PLOTS_DIR/LS/`` if it does not exist.
        Saves amplitude PNGs and I/F fit PNGs to
        ``config.PLOTS_DIR/LS/``.
        Prints fit statistics to stdout.
        Displays interactive matplotlib figures.
    """
    data = np.load(config.CWT_FILE)
    r = data["r"]
    b_norm = data["b_norm"]
    k_est = data["k_est_best"]  # or k_est_smooth_best
    k_est_smooth = data["k_est_smooth_best"]

    # geometry (depends only on r)
    cotB_eff_profile = np.full_like(r, (1 / np.tan(config.B_EFF)))
    B_eff_profile = np.full_like(r, config.B_EFF)

    # define segments
    w1_mask = (r > config.WAVE1_RADIUS_MIN) & (r < config.WAVE1_RADIUS_MAX)
    w2_mask = (r > config.WAVE2_RADIUS_MIN) & (r < config.WAVE2_RADIUS_MAX)

    wave_1_rs = r[w1_mask]
    wave_1_ks = k_est_smooth[w1_mask]
    wave_1_IF = b_norm[w1_mask]

    wave_2_rs = r[w2_mask]
    wave_2_ks = k_est_smooth[w2_mask]
    wave_2_IF = b_norm[w2_mask]

    # matching geometry
    cotB_eff_1 = cotB_eff_profile[w1_mask]
    cotB_eff_2 = cotB_eff_profile[w2_mask]

    # amplitude windows
    L1 = round((2 * np.pi) / np.average(wave_1_ks))
    L2 = round((2 * np.pi) / np.average(wave_2_ks))

    _, _, wave_1_Amp = theta_max_from_IF(
        wave_1_rs,
        wave_1_ks,
        wave_1_IF,
        L=L1,
        cotB_eff=cotB_eff_1,
    )

    _, _, wave_2_Amp = theta_max_from_IF(
        wave_2_rs,
        wave_2_ks,
        wave_2_IF,
        L=L2,
        cotB_eff=cotB_eff_2,
    )

    # Ensure LS directory exists
    os.makedirs(config.PLOTS_DIR + "/LS", exist_ok=True)

    # Wave 1 amplitude
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(wave_1_rs, np.abs(wave_1_Amp * 1000))
    ax1.set_title("Wave 1 LS Amplitude (A_V)")
    ax1.set_xlabel("Radius (km)")
    ax1.set_ylabel("Amplitude (m)")
    plt.tight_layout()
    amp1_path = os.path.join(config.PLOTS_DIR  + "/LS", config.IMG_PRFX + "Wave_1_LS_Amplitude.png")
    fig1.savefig(amp1_path, dpi=200)
    plt.show()

    # Wave 2 amplitude
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(wave_2_rs, np.abs(wave_2_Amp * 1000))
    ax2.set_title("Wave 2 LS Amplitude (A_V)")
    ax2.set_xlabel("Radius (km)")
    ax2.set_ylabel("Amplitude (m)")
    plt.tight_layout()
    amp2_path = os.path.join(config.PLOTS_DIR  + "/LS", config.IMG_PRFX + "Wave_2_LS_Amplitude.png")
    fig2.savefig(amp2_path, dpi=200)
    plt.show()

    # reconstructions
    wave_1_rec = reconstruct_bending_wave(
        wave_1_rs,
        wave_1_ks,
        wave_1_Amp,
        m=1,
        lambdas_deg=[0],
        cotB_eff=cotB_eff_1,
    )
    plot_bending_wave(wave_1_rec, which=("z", "tan_theta", "IF_model"))

    wave_2_rec = reconstruct_bending_wave(
        wave_2_rs,
        wave_2_ks,
        wave_2_Amp,
        m=1,
        lambdas_deg=[0],
        cotB_eff=cotB_eff_2,
    )
    plot_bending_wave(wave_2_rec, which=("z", "tan_theta", "IF_model"))

    # model vs data
    res1 = reconstruct_and_compare_IF(
        wave_1_rs, wave_1_ks, wave_1_Amp, wave_1_IF,
        cotB_eff=cotB_eff_1,
    )

    res2 = reconstruct_and_compare_IF(
        wave_2_rs, wave_2_ks, wave_2_Amp, wave_2_IF,
        cotB_eff=cotB_eff_2,
    )

    # convert fitted cot to B_eff (deg)
    B_eff_fit_1_deg = (np.arctan(1.0 / res1["cotB_eff_fit_mean"]))
    B_eff_fit_2_deg = (np.arctan(1.0 / res2["cotB_eff_fit_mean"]))

    # true means from geometric profile
    B_eff_true_1 = float(np.nanmean(B_eff_profile[w1_mask]))
    B_eff_true_2 = float(np.nanmean(B_eff_profile[w2_mask]))

    print(f"Wave 1: RMSE = {res1['rmse']:.4g}  R² = {res1['r2']:.3f}  "
          f"corr = {res1['corr']:.3f}  B_eff_fit ≈ {B_eff_fit_1_deg:.3f}°  "
          f"(true ≈ {B_eff_true_1:.3f}°)")
    plot_IF_fit(res1, title="Wave 1 LS Reconstruction", save_dir=config.PLOTS_DIR  + "/LS")

    print(f"Wave 2: RMSE = {res2['rmse']:.4g}  R² = {res2['r2']:.3f}  "
          f"corr = {res2['corr']:.3f}  B_eff_fit ≈ {B_eff_fit_2_deg:.3f}°  "
          f"(true ≈ {B_eff_true_2:.3f}°)")
    plot_IF_fit(res2, title="Wave 2 LS Reconstruction", save_dir=config.PLOTS_DIR  + "/LS")

if __name__ == "__main__":
    main()
