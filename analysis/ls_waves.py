# analysis/ls_waves.py
"""Least-squares bending-wave amplitude analysis for Saturn ring wave regions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

import config
from helpers.physics import theta_max_from_IF, reconstruct_and_compare_IF
from helpers.plotting import plot_IF_fit

os.makedirs(config.PLOTS_DIR, exist_ok=True)


def main(waves=None):
    """Performs least-squares bending-wave analysis for the specified wave regions.

    Loads the precomputed CWT/ridge file, extracts each requested wave's
    radial segment, estimates local vertical amplitude via
    :func:`helpers.physics.theta_max_from_IF`, compares the photometric
    model to the data, and saves amplitude and I/F-fit plots.

    Args:
        waves (list[int] | None): Wave numbers to analyse.  Valid values are
            ``1``, ``2``, and ``5``.  If ``None``, all three waves are
            processed. Defaults to ``None``.

    Side effects:
        Creates ``config.PLOTS_DIR/LS/`` if it does not exist.
        Saves amplitude PNGs and I/F fit PNGs to ``config.PLOTS_DIR/LS/``.
        Prints fit statistics to stdout.
        Displays interactive matplotlib figures.
    """
    if waves is None:
        waves = [1, 2, 5]

    data = np.load(config.CWT_FILE)
    r = data["r"]
    b_norm = data["b_norm"]
    k_est_smooth = data["k_est_smooth_best"]

    cotB_eff_profile = np.full_like(r, (1.0 / np.tan(config.B_EFF)))
    B_eff_profile = np.full_like(r, config.B_EFF)

    os.makedirs(config.PLOTS_DIR + "/LS", exist_ok=True)

    wave_configs = {
        1: (config.WAVE1_RADIUS_MIN, config.WAVE1_RADIUS_MAX),
        2: (config.WAVE2_RADIUS_MIN, config.WAVE2_RADIUS_MAX),
        5: (config.WAVE5_RADIUS_MIN, config.WAVE5_RADIUS_MAX),
    }

    for wave_num in waves:
        rmin, rmax = wave_configs[wave_num]
        mask = (r > rmin) & (r < rmax)

        wave_rs = r[mask]
        wave_ks = k_est_smooth[mask]
        cotB_eff_w = cotB_eff_profile[mask]

        # Wave 5 uses local re-normalization; others use b_norm directly
        if wave_num == 5:
            wave_raw = b_norm[mask]
            wave_IF = wave_raw / np.average(wave_raw)
            IF_avg_for_fit = 1.0
        else:
            wave_IF = b_norm[mask]
            IF_avg_for_fit = None  # theta_max_from_IF will use mean(IF)

        # Window length: approx one wavelength
        L = round((2 * np.pi) / np.average(wave_ks))

        _, _, wave_Amp = theta_max_from_IF(
            wave_rs,
            wave_ks,
            wave_IF,
            L=L,
            cotB_eff=cotB_eff_w,
            IF_avg=IF_avg_for_fit,
        )

        # Save amplitude for use by summary plots
        ls_npz_path = os.path.join(
            config.OUT_DIR, config.IMG_PRFX + f"ls_wave{wave_num}_amp.npz"
        )
        np.savez(ls_npz_path, r=wave_rs, A_V=np.abs(wave_Amp), IF_data=wave_IF, k=wave_ks)

        # Amplitude plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(wave_rs, np.abs(wave_Amp * 1000))
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(
            f"Rev {config.IMG_REV} · Image {config.IMG_NUM} – Wave {wave_num} LS Amplitude (A_V)"
        )
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Amplitude (m)")
        plt.tight_layout()
        amp_path = os.path.join(
            config.PLOTS_DIR + "/LS",
            config.IMG_PRFX + f"w{wave_num}_A_V.png",
        )
        fig.savefig(amp_path, dpi=200)
        plt.show()

        # I/F fit
        res = reconstruct_and_compare_IF(
            wave_rs, wave_ks, wave_Amp, wave_IF,
            cotB_eff=cotB_eff_w,
            IF_avg=IF_avg_for_fit,
        )

        B_eff_fit_deg = np.arctan(1.0 / res["cotB_eff_fit_mean"])
        B_eff_true = float(np.nanmean(B_eff_profile[mask]))

        print(
            f"Wave {wave_num}: RMSE = {res['rmse']:.4g}  R² = {res['r2']:.3f}  "
            f"corr = {res['corr']:.3f}  B_eff_fit ≈ {B_eff_fit_deg:.3f}°  "
            f"(true ≈ {B_eff_true:.3f}°)"
        )
        plot_IF_fit(
            res,
            title=f"Rev {config.IMG_REV} · Image {config.IMG_NUM} – Wave {wave_num} LS Reconstruction",
            save_dir=config.PLOTS_DIR + "/LS",
            filename=config.IMG_PRFX + f"w{wave_num}_IF_fit.png",
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LS analysis for one or all waves.")
    parser.add_argument(
        "wave_num",
        type=int,
        choices=[1, 2, 5],
        nargs="?",
        default=None,
        help="Wave number to analyse (1, 2, or 5). Omit to run all.",
    )
    args = parser.parse_args()
    main(waves=[args.wave_num] if args.wave_num is not None else None)
