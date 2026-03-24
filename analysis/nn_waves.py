# analysis/nn_waves.py
"""Physics-informed neural network amplitude analysis for Saturn ring bending waves."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import config
from helpers.utils import to_native_endian
from helpers.nn_model import ContextAmpCSNet, build_features, forward_physics
from helpers.physics import fit_stats
from helpers.plotting import plot_IF_fit, plot_wave_summary

os.makedirs(config.PLOTS_DIR, exist_ok=True)

# Wave-specific configuration: (rmin, rmax, epochs, npz_suffix)
_WAVE_CONFIG = {
    1: (config.WAVE1_RADIUS_MIN, config.WAVE1_RADIUS_MAX, 2000, "nn_wave1_amp.npz"),
    2: (config.WAVE2_RADIUS_MIN, config.WAVE2_RADIUS_MAX, 2000, "nn_wave2_amp.npz"),
    5: (config.WAVE5_RADIUS_MIN, config.WAVE5_RADIUS_MAX, 4000, "nn_wave5_amp.npz"),
}


def main(wave_num):
    """Trains a physics-informed neural network to estimate bending-wave amplitude.

    Loads precomputed CWT/ridge results, isolates the radial segment for the
    requested wave, and trains :class:`helpers.nn_model.ContextAmpCSNet` to
    predict complex amplitude components ``(A_c, A_s)`` supervised by the
    photometric I/F model.

    Args:
        wave_num (int): Wave number to analyse.  Must be one of ``1``, ``2``,
            or ``5``.

    Side effects:
        Creates ``config.PLOTS_DIR/NN/`` if it does not exist.
        Writes a ``.npz`` amplitude file to ``config.OUT_DIR``.
        Saves multiple PNG plots to ``config.PLOTS_DIR/NN/``.
        Prints per-epoch loss breakdown to stdout every 100 epochs.
        Displays interactive matplotlib figures.

    Raises:
        ValueError: If ``wave_num`` is not 1, 2, or 5.
    """
    if wave_num not in _WAVE_CONFIG:
        raise ValueError(f"wave_num must be 1, 2, or 5; got {wave_num}")

    rmin, rmax, n_epochs, npz_suffix = _WAVE_CONFIG[wave_num]

    # ------------------------------------------------
    # 1) load precomputed CWT/ridge results
    # ------------------------------------------------
    data = np.load(config.CWT_FILE)
    r = data["r"]
    b_norm = data["b_norm"]
    k_est_smooth = data["k_est_smooth_best"]

    cotB_eff_profile = np.full_like(r, (1.0 / np.tan(config.B_EFF)))

    # ------------------------------------------------
    # 2) wave mask
    # ------------------------------------------------
    mask = (r > rmin) & (r < rmax)

    r_w = r[mask]
    k_w = k_est_smooth[mask]
    IF_w = b_norm[mask]
    cotB_w = cotB_eff_profile[mask]

    # endian fix for PyTorch
    r_w = to_native_endian(r_w)
    k_w = to_native_endian(k_w)
    IF_w = to_native_endian(IF_w)
    cotB_w = to_native_endian(cotB_w)

    # ------------------------------------------------
    # 3) torch tensors
    # ------------------------------------------------
    device = torch.device("cpu")
    r_t = torch.tensor(r_w, dtype=torch.float32, device=device)
    k_t = torch.tensor(k_w, dtype=torch.float32, device=device)
    cotB_t = torch.tensor(cotB_w, dtype=torch.float32, device=device)
    IF_t = torch.tensor(IF_w, dtype=torch.float32, device=device)

    # ------------------------------------------------
    # 4) build NN inputs
    # ------------------------------------------------
    X, psi = build_features(r_t, k_t, cotB_t)

    # ------------------------------------------------
    # 5) model + optimizer
    # ------------------------------------------------
    model = ContextAmpCSNet(in_dim=X.shape[1], hidden=64).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    IF0 = float(IF_w.mean())

    # ------------------------------------------------
    # 6) training loop
    # ------------------------------------------------
    for epoch in range(n_epochs):
        opt.zero_grad()

        A_c_pred, A_s_pred = model(X)

        IF_model = forward_physics(r_t, k_t, cotB_t, A_c_pred, A_s_pred, IF0=IF0)

        data_loss = torch.mean((IF_model - IF_t) ** 2)

        diff_c = A_c_pred[1:] - A_c_pred[:-1]
        diff_s = A_s_pred[1:] - A_s_pred[:-1]
        smooth_loss = (diff_c ** 2).mean() + (diff_s ** 2).mean()

        diff2_c = diff_c[1:] - diff_c[:-1]
        diff2_s = diff_s[1:] - diff_s[:-1]
        curv_loss = (diff2_c ** 2).mean() + (diff2_s ** 2).mean()

        loss = (data_loss
                + config.NN_SMOOTH_LOSS_COEF * smooth_loss
                + config.NN_CURV_LOSS_COEF * curv_loss)

        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"epoch {epoch+1}: loss={loss.item():.4e}  "
                f"(data={data_loss.item():.4e}, smooth={smooth_loss.item():.4e}, "
                f"curv={curv_loss.item():.4e})"
            )

    # ------------------------------------------------
    # 7) detach results
    # ------------------------------------------------
    A_c_final = A_c_pred.detach().cpu().numpy()
    A_s_final = A_s_pred.detach().cpu().numpy()
    A_v_final = np.sqrt(A_c_final**2 + A_s_final**2)

    IF_model_final = IF_model.detach().cpu().numpy()
    psi_np = psi.detach().cpu().numpy()

    # ------------------------------------------------
    # 8) I/F fit plot
    # ------------------------------------------------
    result = {
        "r": r_w,
        "y": IF_w - IF0,
        "y_hat": IF_model_final - IF0,
        "IF_avg": IF0,
        "cotB_eff_input": cotB_w,
        "cotB_eff_fit_mean": float(np.nanmean(cotB_w)),
    }
    plot_IF_fit(
        result,
        title=f"Rev {config.IMG_REV} · Image {config.IMG_NUM} – Wave {wave_num} NN Reconstruction",
        save_dir=config.PLOTS_DIR + "/NN",
    )

    # ------------------------------------------------
    # 9) save amplitude npz
    # ------------------------------------------------
    theta_max = A_v_final * k_w

    out_path = os.path.join(config.OUT_DIR, config.IMG_PRFX + npz_suffix)
    np.savez(
        out_path,
        r=r_w,
        A_V=A_v_final,
        theta_max=theta_max,
        IF_data=IF_w,
        IF_model=IF_model_final,
    )
    print(f"Saved NN amplitude to {out_path}")

    # ------------------------------------------------
    # 10) fit statistics
    # ------------------------------------------------
    stats = fit_stats(IF_w, IF_model_final)
    mean_A_V_m = float(np.mean(A_v_final) * 1000.0)
    max_A_V_m  = float(np.max(A_v_final)  * 1000.0)
    print(
        f"Wave {wave_num} NN:  RMSE = {stats['rmse']:.4g}  "
        f"R² = {stats['r2']:.3f}  corr = {stats['corr']:.3f}  "
        f"mean A_V = {mean_A_V_m:.2f} m  max A_V = {max_A_V_m:.2f} m"
    )

    # ------------------------------------------------
    # 11) diagnostic plots
    # ------------------------------------------------
    os.makedirs(config.PLOTS_DIR + "/NN", exist_ok=True)

    label = f"Rev {config.IMG_REV} · Image {config.IMG_NUM} – Wave {wave_num} NN"

    # A_V
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(r_w, A_v_final * 1000.0)
    ax1.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax1.set_title(f"{label} Amplitude (A_V)")
    ax1.set_xlabel("Radius (km)")
    ax1.set_ylabel("Amplitude (m)")
    plt.tight_layout()
    fig1.savefig(os.path.join(config.PLOTS_DIR + "/NN",
                              config.IMG_PRFX + f"w{wave_num}_A_V.png"), dpi=200)
    plt.show()

    # A_c
    fig_c, ax_c = plt.subplots(figsize=(10, 4))
    ax_c.plot(r_w, A_c_final * 1000.0)
    ax_c.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax_c.set_title(f"{label} Cosine Amplitude (A_c)")
    ax_c.set_xlabel("Radius (km)")
    ax_c.set_ylabel("Amplitude (m)")
    plt.tight_layout()
    fig_c.savefig(os.path.join(config.PLOTS_DIR + "/NN",
                               config.IMG_PRFX + f"w{wave_num}_A_c.png"), dpi=200)
    plt.show()

    # A_s
    fig_s, ax_s = plt.subplots(figsize=(10, 4))
    ax_s.plot(r_w, A_s_final * 1000.0)
    ax_s.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax_s.set_title(f"{label} Sine Amplitude (A_s)")
    ax_s.set_xlabel("Radius (km)")
    ax_s.set_ylabel("Amplitude (m)")
    plt.tight_layout()
    fig_s.savefig(os.path.join(config.PLOTS_DIR + "/NN",
                               config.IMG_PRFX + f"w{wave_num}_A_s.png"), dpi=200)
    plt.show()

    # psi(r)
    fig_psi, ax_psi = plt.subplots(figsize=(10, 4))
    ax_psi.plot(r_w, psi_np)
    ax_psi.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax_psi.set_title(f"{label} Phase (psi)")
    ax_psi.set_xlabel("Radius (km)")
    ax_psi.set_ylabel("Phase (rad)")
    plt.tight_layout()
    fig_psi.savefig(os.path.join(config.PLOTS_DIR + "/NN",
                                 config.IMG_PRFX + f"w{wave_num}_psi.png"), dpi=200)
    plt.show()

    # k(r)
    fig_k, ax_k = plt.subplots(figsize=(10, 4))
    ax_k.plot(r_w, k_w)
    ax_k.set_title(f"{label} Wave Number (k)")
    ax_k.set_xlabel("Radius (km)")
    ax_k.set_ylabel("Wave Number (rad/km)")
    plt.tight_layout()
    fig_k.savefig(os.path.join(config.PLOTS_DIR + "/NN",
                               config.IMG_PRFX + f"w{wave_num}_k.png"), dpi=200)
    plt.show()

    # ------------------------------------------------
    # 12) four-panel summary (I/F, k, psi, A_V with NN+LS)
    # ------------------------------------------------
    ls_npz_path = os.path.join(config.OUT_DIR, config.IMG_PRFX + f"ls_wave{wave_num}_amp.npz")
    A_V_ls = None
    if os.path.exists(ls_npz_path):
        ls_data = np.load(ls_npz_path)
        # LS r may differ slightly in length; interpolate onto NN grid if needed
        if len(ls_data["r"]) == len(r_w) and np.allclose(ls_data["r"], r_w):
            A_V_ls = ls_data["A_V"]
        else:
            A_V_ls = np.interp(r_w, ls_data["r"], ls_data["A_V"])
    else:
        print(f"  [summary] LS npz not found at {ls_npz_path} — plotting NN only.")

    plot_wave_summary(
        r=r_w,
        IF_data=IF_w,
        k=k_w,
        psi=psi_np,
        A_V_nn=A_v_final,
        A_V_ls=A_V_ls,
        wave_num=wave_num,
        save_dir=config.PLOTS_DIR + "/NN",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NN analysis for a single wave.")
    parser.add_argument("wave_num", type=int, choices=[1, 2, 5],
                        help="Wave number to analyse (1, 2, or 5)")
    args = parser.parse_args()
    main(args.wave_num)
