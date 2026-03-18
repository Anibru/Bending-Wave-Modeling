# 01_make_cwt_and_ridge.py

import os
import numpy as np
import matplotlib
# matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from scipy.io import readsav

import config
from utils import edges_from_centers, prepare_power
from wavelet_cwt import run_cwt_and_ridge

OUT_FILE = os.path.join(config.OUT_DIR, "cwt_ridge_result.npz")

def main():
    """Loads a raw brightness profile, normalizes it, runs CWT ridge detection, and saves results.

    Execution steps:

    1. Load the IDL ``.sav`` profile from ``config.DATA_PATH``.
    2. Globally normalize brightness using reference bands
       ``config.GLOBAL_BAND_1`` and ``config.GLOBAL_BAND_2``.
    3. Run :func:`wavelet_cwt.run_cwt_and_ridge` over the configured
       bandwidth grid with optional ridge masking for wave 2.
    4. Plot the wavelet power spectrum and the raw/smoothed ridge trace,
       saving both figures to ``config.PLOTS_DIR``.
    5. Save all CWT/ridge arrays to ``config.OUT_DIR/cwt_ridge_result.npz``
       and export a ``ridge_r_k.csv`` summary.

    Side effects:
        Creates output and plot directories if they do not exist.
        Writes ``cwt_ridge_result.npz`` and ``ridge_r_k.csv`` to
        ``config.OUT_DIR``.  Saves ``wavelet_power.png`` and
        ``ridge_trace.png`` to ``config.PLOTS_DIR``.
        Displays two interactive matplotlib figures.
    """
    os.makedirs(config.OUT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # 1) load raw profile
    data = readsav(config.DATA_PATH)
    r = data["radi"]
    b = data["val"]

    # 2) global normalization
    m1 = (r > config.GLOBAL_BAND_1[0]) & (r < config.GLOBAL_BAND_1[1])
    m2 = (r > config.GLOBAL_BAND_2[0]) & (r < config.GLOBAL_BAND_2[1])
    ref_vals = np.concatenate([b[m1], b[m2]])
    if ref_vals.size == 0:
        raise ValueError("Reference bands are empty; update GLOBAL_BAND_1/2.")
    I0_over_F_global = np.mean(ref_vals)
    b_norm = b / I0_over_F_global

    # 3) run CWT + ridge
    dr = np.mean(np.diff(r))
    wavenumbers = np.linspace(config.KMIN_CYC, config.KMAX_CYC, config.NUM_K)
    sampling_period = dr
    target_cf = config.MORLET_W / (2.0 * np.pi)

    cwt_res = run_cwt_and_ridge(
        b_norm,
        r,
        wavenumbers,
        config.BANDWIDTH_GRID,
        target_cf,
        sampling_period,
        ridge_mask_r=(config.RIDGE_MASK_RMIN_2, config.RIDGE_MASK_RMAX_2),
        ridge_mask_k=(config.RIDGE_MASK_KMIN_2, config.RIDGE_MASK_KMAX_2),
    )

    power_best = cwt_res["power_best"]
    scales_best = cwt_res["scales_best"]
    k_est_best = cwt_res["k_est_best"]
    k_est_smooth_best = cwt_res["k_est_smooth_best"]
    wavelet_best = cwt_res["wavelet_best"]

    print("Tuning results (B → score):")
    for B, score, *_ in cwt_res["all_results"]:
        print(f"  B={B:<3} → {score:.3e}")
    print(f"\nSelected wavelet: {wavelet_best} (score={cwt_res['score_best']:.3e})\n")

    # 4) plot power
    power_plot = prepare_power(
        power_best,
        scales_best,
        mode=config.SCALE_MODE,
        colnorm=config.USE_COLNORM,
        scale_correct=config.USE_PER_OCT,
    )
    vmin = np.nanpercentile(power_plot, config.CLIP_PCT[0])
    vmax = np.nanpercentile(power_plot, config.CLIP_PCT[1])

    r_edges = edges_from_centers(r)
    k_edges = edges_from_centers(wavenumbers)

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(r_edges, k_edges, power_plot,
                         shading="auto", vmin=vmin, vmax=vmax)
    ax.set_yscale("log")
    ax.set_ylim((config.KMIN_CYC, config.KMAX_CYC))
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Wavenumber [cycles/km]")
    ax.set_title(f"Wavelet Power — {wavelet_best}")
    fig.colorbar(mesh, ax=ax, label="Power")
    plt.tight_layout()
    # save
    power_plot_path = os.path.join(config.PLOTS_DIR, config.IMG_PRFX + "wavelet_power.png")
    fig.savefig(power_plot_path, dpi=200)
    plt.show()

    # 5) plot ridge (raw vs smoothed)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(r, k_est_best, label="Ridge (raw)", alpha=0.7)
    ax2.plot(r, k_est_smooth_best, label="Ridge (smoothed)", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_ylim((config.KMIN, config.KMAX))
    ax2.set_xlabel("Radius (km)")
    ax2.set_ylabel("Wavenumber [rad/km]")
    ax2.set_title("Dominant Wavenumber")
    ax2.legend()
    plt.tight_layout()
    ridge_plot_path = os.path.join(config.PLOTS_DIR, config.IMG_PRFX + "ridge_trace.png")
    fig2.savefig(ridge_plot_path, dpi=200)
    plt.show()

    # 6) save data for later scripts
    np.savez(
        OUT_FILE,
        r=r,
        b_norm=b_norm,
        wavenumbers=wavenumbers,
        k_est_best=k_est_best,
        k_est_smooth_best=k_est_smooth_best,
        power_best=power_best,
        scales_best=scales_best,
        wavelet_best=wavelet_best,
        I0_over_F_global=I0_over_F_global,
        dr=dr,
    )
    # optional: also a CSV for quick inspection
    np.savetxt(
        os.path.join(config.OUT_DIR, "ridge_r_k.csv"),
        np.column_stack([r, k_est_smooth_best]),
        header="r,k_smooth_cycles_per_km",
        delimiter=",",
    )

    print(f"Saved CWT/ridge results to {OUT_FILE}")
    print(f"Saved plots to {config.PLOTS_DIR}")

if __name__ == "__main__":
    main()
