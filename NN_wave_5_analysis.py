# Train NN to predict complex amplitude components (A_c, A_s) on wave 5 segment
# using precomputed CWT/ridge results. Physical vertical amplitude is
#   A_V(r) = sqrt(A_c(r)^2 + A_s(r)^2).

import os
import numpy as np
import torch
import torch.optim as optim
# import matplotlib
# matplotlib.use("MacOSX")  # or "TkAgg"
import matplotlib.pyplot as plt

from reconstruction import plot_IF_fit
import config
from nn_wave_model import (
    ContextAmpCSNet,   # updated class name
    build_features,
    forward_physics,
)

os.makedirs(config.PLOTS_DIR, exist_ok=True)


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


def main():
    """Trains a neural-network bending-wave model on the wave 5 radial segment.

    Identical workflow to ``NN_wave_1_analysis.main()`` targeting the wave 5
    segment (``config.WAVE5_RADIUS_MIN`` to ``config.WAVE5_RADIUS_MAX``).
    The training run is longer (4000 epochs) and uses stronger smoothing
    penalties (1st-derivative: 1e4, 2nd-derivative: 1e3) to handle the
    noisier wave 5 region.  Output is saved as ``nn_wave5_amp.npz`` with
    the ``tan_max`` key instead of ``theta_max``.

    Execution steps:

    1. Load ``r``, ``b_norm``, and ``k_est_smooth_best`` from
       ``config.CWT_FILE``.
    2. Mask the wave 5 radial segment and fix byte order for PyTorch.
    3. Convert arrays to ``torch.float32`` tensors on CPU.
    4. Build feature matrix ``X`` and initial phase ``psi`` via
       :func:`nn_wave_model.build_features`.
    5. Instantiate ``ContextAmpCSNet`` with ``hidden=64`` and optimize with
       Adam (lr=1e-3) for 4000 epochs using a composite loss:

       - MSE data loss between modeled and observed I/F.
       - First-derivative smoothness penalty on ``A_c`` and ``A_s``
         (coefficient 1e4).
       - Second-derivative curvature penalty (coefficient 1e3).

    6. Plot the I/F fit using :func:`reconstruction.plot_IF_fit`.
    7. Save ``A_V``, ``tan_max``, and model I/F to
       ``config.OUT_DIR/{IMG_PRFX}nn_wave5_amp.npz``.
    8. Plot and save ``A_V``, ``A_c``, ``A_s``, ``psi``, and ``k`` traces
       to ``config.PLOTS_DIR/NN/``.

    Side effects:
        Creates ``config.PLOTS_DIR/NN/`` if it does not exist.
        Writes ``.npz`` amplitude file to ``config.OUT_DIR``.
        Saves multiple PNG plots to ``config.PLOTS_DIR/NN/``.
        Prints per-epoch loss breakdown to stdout every 100 epochs.
        Displays interactive matplotlib figures.
    """
    # ------------------------------------------------
    # 1) load precomputed CWT/ridge results
    # ------------------------------------------------
    data = np.load(config.CWT_FILE)
    r = data["r"]                  # full profile radii
    b_norm = data["b_norm"]        # normalized I/F (full)
    k_est_smooth = data["k_est_smooth_best"]  # k(r) (full, rad/km)

    cotB_eff_profile = np.full_like(r, (1.0 / np.tan(config.B_EFF)))

    # ------------------------------------------------
    # 2) wave-5 mask FIRST
    # ------------------------------------------------
    mask = (r > config.WAVE5_RADIUS_MIN) & (r < config.WAVE5_RADIUS_MAX)

    r_w = r[mask]
    k_w = k_est_smooth[mask]
    IF_w = b_norm[mask]
    cotB_w = cotB_eff_profile[mask]

    # endian fix AFTER masking
    r_w = to_native_endian(r_w)
    k_w = to_native_endian(k_w)
    IF_w = to_native_endian(IF_w)
    cotB_w = to_native_endian(cotB_w)

    # ------------------------------------------------
    # 3) torch tensors (all windowed, same length)
    # ------------------------------------------------
    device = torch.device("cpu")
    r_t = torch.tensor(r_w, dtype=torch.float32, device=device)
    k_t = torch.tensor(k_w, dtype=torch.float32, device=device)
    cotB_t = torch.tensor(cotB_w, dtype=torch.float32, device=device)
    IF_t = torch.tensor(IF_w, dtype=torch.float32, device=device)

    # ------------------------------------------------
    # 4) build NN inputs (use feature builder from model file)
    # ------------------------------------------------
    X, psi = build_features(r_t, k_t, cotB_t)   # X: [Nwin, F]

    # ------------------------------------------------
    # 5) model + optimizer
    # ------------------------------------------------
    model = ContextAmpCSNet(in_dim=X.shape[1], hidden=64).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    IF0 = float(IF_w.mean())

    # ------------------------------------------------
    # 6) training loop
    # ------------------------------------------------
    for epoch in range(4000):
        opt.zero_grad()

        # NN predicts A_c(r), A_s(r)
        A_c_pred, A_s_pred = model(X)  # each [Nwin]

        # physics model uses ONLY windowed tensors
        IF_model = forward_physics(r_t, k_t, cotB_t, A_c_pred, A_s_pred, IF0=IF0)

        # data loss
        data_loss = torch.mean((IF_model - IF_t) ** 2)

        # first derivative smoothing on A_c and A_s
        diff_c = A_c_pred[1:] - A_c_pred[:-1]
        diff_s = A_s_pred[1:] - A_s_pred[:-1]
        smooth_loss = (diff_c ** 2).mean() + (diff_s ** 2).mean()

        # second derivative smoothing on A_c and A_s
        diff2_c = diff_c[1:] - diff_c[:-1]
        diff2_s = diff_s[1:] - diff_s[:-1]
        curv_loss = (diff2_c ** 2).mean() + (diff2_s ** 2).mean()

        # Smoothing coefficients:
        smooth_loss_coef = 1e4
        curve_loss_coef = 1e3

        loss = data_loss + smooth_loss_coef * smooth_loss + curve_loss_coef * curv_loss

        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"epoch {epoch+1}: loss={loss.item():.4e}  "
                f"(data={data_loss.item():.4e}, smooth={smooth_loss.item():.4e}, "
                f"curv={curv_loss.item():.4e})"
            )

    # ------------------------------------------------
    # 7) detach (still window-length!)
    # ------------------------------------------------
    A_c_final = A_c_pred.detach().cpu().numpy()
    A_s_final = A_s_pred.detach().cpu().numpy()
    A_v_final = np.sqrt(A_c_final**2 + A_s_final**2)   # physical amplitude

    IF_model_final = IF_model.detach().cpu().numpy()
    psi_np = psi.detach().cpu().numpy()

    # ------------------------------------------------
    # 8) plot using existing helper
    # ------------------------------------------------
    result = {
        "r": r_w,
        "y": IF_w - IF0,
        "y_hat": IF_model_final - IF0,
        "IF0": IF0,
        "cotB_eff_input": cotB_w,
        "cotB_eff_fit_mean": float(np.nanmean(cotB_w)),
    }

    plot_IF_fit(result, title="Wave 5 NN Reconstruction", save_dir=config.PLOTS_DIR + "/NN")

    # ------------------------------------------------
    # 9) save learned amplitude and related quantities
    # ------------------------------------------------
    tan_max = A_v_final * k_w  # theta_max(r) ≈ k(r) * A_V(r)

    out_path = os.path.join(config.OUT_DIR, config.IMG_PRFX + "nn_wave5_amp.npz")
    np.savez(
        out_path,
        r=r_w,
        A_V=A_v_final,
        tan_max=tan_max,
        IF_data=IF_w,
        IF_model=IF_model_final,
    )
    print("Saved NN amplitude to " + out_path)

    # ------------------------------------------------
    # 10) plots
    # ------------------------------------------------

    # Ensure NN directory exists
    os.makedirs(config.PLOTS_DIR + "/NN", exist_ok=True)

    # A_V (in meters)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(r_w, A_v_final * 1000.0)
    ax1.set_title("Wave 5 NN Amplitude (A_V)")
    ax1.set_xlabel("Radius (km)")
    ax1.set_ylabel("Amplitude (m)")
    plt.tight_layout()
    amp1_path = os.path.join(config.PLOTS_DIR + "/NN", config.IMG_PRFX + "Wave_5_NN_Amplitude.png")
    fig1.savefig(amp1_path, dpi=200)
    plt.show()

    # A_c (in meters)
    fig1c, ax1c = plt.subplots(figsize=(10, 4))
    ax1c.plot(r_w, A_c_final * 1000.0)
    ax1c.set_title("Wave 5 NN Cosine Amplitude (A_c)")
    ax1c.set_xlabel("Radius (km)")
    ax1c.set_ylabel("Amplitude (m)")
    plt.tight_layout()
    amp1c_path = os.path.join(config.PLOTS_DIR + "/NN", config.IMG_PRFX + "Wave_5_NN_Cosine_Amplitude.png")
    fig1c.savefig(amp1c_path, dpi=200)
    plt.show()

    # A_s (in meters)
    fig1s, ax1s = plt.subplots(figsize=(10, 4))
    ax1s.plot(r_w, A_s_final * 1000.0)
    ax1s.set_title("Wave 5 NN Sine Amplitude (A_s)")
    ax1s.set_xlabel("Radius (km)")
    ax1s.set_ylabel("Amplitude (m)")
    plt.tight_layout()
    amp1s_path = os.path.join(config.PLOTS_DIR + "/NN", config.IMG_PRFX + "Wave_5_NN_Sine_Amplitude.png")
    fig1c.savefig(amp1s_path, dpi=200)
    plt.show()

    # psi(r)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(r_w, psi_np)
    ax2.set_title("Wave 5 NN psi")
    ax2.set_xlabel("Radius (km)")
    ax2.set_ylabel("Phase (rad)")
    plt.tight_layout()
    psi_path = os.path.join(config.PLOTS_DIR + "/NN", config.IMG_PRFX + "Wave_5_NN_psi.png")
    fig2.savefig(psi_path, dpi=200)
    plt.show()

    # k(r)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(r_w, k_w)
    ax3.set_title("Wave 5 NN k")
    ax3.set_xlabel("Radius (km)")
    ax3.set_ylabel("Wave Number (rad/km)")
    plt.tight_layout()
    k_path = os.path.join(config.PLOTS_DIR + "/NN", config.IMG_PRFX + "Wave_5_NN_k.png")
    fig3.savefig(k_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
