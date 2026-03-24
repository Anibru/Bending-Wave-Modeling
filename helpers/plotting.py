# helpers/plotting.py
"""Matplotlib plotting helpers for the bending-wave analysis pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import config


def plot_bending_wave(out, which=("z", "tan_theta", "IF_model"), titles=None,
                      save_dir=None, prefix="bending_wave"):
    """Plots and saves selected bending-wave reconstruction curves.

    Creates a vertically stacked multi-panel figure, one panel per quantity
    in ``which``.  All curves in ``out["curves"]`` are overplotted together,
    labeled by their longitude.  The figure is saved to ``save_dir`` before
    being displayed.

    Args:
        out (dict): Output dictionary from
            :func:`helpers.physics.reconstruct_bending_wave`.
        which (tuple[str]): Keys to plot from each curve dict.  Valid values
            are ``"z"``, ``"tan_theta"``, and ``"IF_model"``.  Defaults to
            ``("z", "tan_theta", "IF_model")``.
        titles (dict[str, str] | None): Optional mapping from key name to
            panel title string.  If ``None``, default physics-descriptive
            titles are used.
        save_dir (str | None): Directory in which to save the figure.
            Defaults to ``config.PLOTS_DIR``.
        prefix (str): Filename prefix for the saved PNG.  Defaults to
            ``"bending_wave"``.
    """
    if save_dir is None:
        save_dir = config.PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    r = out["r"]
    Ls = out["lambdas_deg"]

    if titles is None:
        titles = {
            "z": "Vertical displacement z(r, λ)",
            "tan_theta": "Local tilt tanθ(r, λ)",
            "IF_model": "Modeled I/F (paper form)",
        }

    nplots = len(which)
    fig, axs = plt.subplots(nplots, 1, figsize=(8, 2.6 * nplots), sharex=True)
    if nplots == 1:
        axs = [axs]

    for ax, key in zip(axs, which):
        for c in out["curves"]:
            if key in c:
                ax.plot(r, c[key], label=f"λ = {c['lambda_deg']:.0f}°")
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set_ylabel(key)
        ax.set_title(titles.get(key, key))
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, ncols=min(3, len(Ls)))

    axs[-1].set_xlabel("Radius r")
    plt.tight_layout()

    save_path = os.path.join(save_dir, config.IMG_PRFX + f"{prefix}_{'_'.join(which)}.png")
    fig.savefig(save_path, dpi=200)
    plt.show()


def plot_IF_fit(result, title="", save_dir=None, prefix="", filename=None):
    """Plots observed vs modeled I/F residuals and saves the figure.

    Produces a two-panel figure:

    - **Upper panel**: observed residual ``y = IF - IF_avg`` versus the model
      fit ``y_hat``, with a legend reporting the input and fitted cotangent
      values.
    - **Lower panel**: raw residual ``y - y_hat``.

    The figure is saved automatically before being displayed.

    Args:
        result (dict): Output dictionary from
            :func:`helpers.physics.reconstruct_and_compare_IF`, expected to
            contain keys ``"r"``, ``"y"``, ``"y_hat"``, ``"IF_avg"``,
            ``"cotB_eff_input"`` (optional), and
            ``"cotB_eff_fit_mean"`` (optional).
        title (str): Figure title string. Defaults to ``""``.
        save_dir (str | None): Directory in which to save the figure.
            Defaults to ``config.PLOTS_DIR``.
        prefix (str): Optional filename prefix prepended when ``filename`` is
            not given.  Defaults to ``""``.
        filename (str | None): Explicit output filename (basename only, no
            directory).  When provided, ``prefix`` is ignored and the title is
            not used to derive the name.  Defaults to ``None``.
    """
    if save_dir is None:
        save_dir = config.PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    r = result["r"]
    y = result["y"]
    y_hat = result["y_hat"]
    IF_avg = result.get("IF_avg", 1.0)

    cot_in = result.get("cotB_eff_input", None)
    cot_fit_mean = result.get("cotB_eff_fit_mean", None)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # upper panel: data vs model
    ax[0].plot(r, y, label=f"Measured residual = IF - {IF_avg:.3f}", alpha=0.8)
    ax[0].plot(r, y_hat, label="Model", linestyle="--")
    ax[0].axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax[0].set_ylabel("Residual I/F")
    ax[0].set_title(title)

    legend_lines = []
    if cot_in is not None:
        cot_in_mean = float(np.nanmean(cot_in))
        legend_lines.append(f"cotB_eff(in)={cot_in_mean:.3f}")
    if cot_fit_mean is not None:
        legend_lines.append(f"cotB_eff(fit)={cot_fit_mean:.3f}")

    if legend_lines:
        ax[0].legend(frameon=False, title=", ".join(legend_lines))
    else:
        ax[0].legend(frameon=False)

    ax[0].grid(alpha=0.3)

    # lower panel: residuals
    ax[1].plot(r, y - y_hat, label="Residual (data - model)")
    ax[1].axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax[1].set_ylabel("Residual")
    ax[1].set_xlabel("Radius r")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()

    if filename is not None:
        fname = filename
    else:
        fname = config.IMG_PRFX + f"{prefix}_{title.replace(' ', '_')}.png"
    save_path = os.path.join(save_dir, fname)
    fig.savefig(save_path, dpi=200)
    plt.show()


def plot_wave_summary(r, IF_data, k, psi, A_V_nn, A_V_ls=None,
                      wave_num=None, save_dir=None):
    """Plots a four-panel summary of a single wave's analysis results.

    Panels (all share the same x-axis):

    1. **I/F profile** — observed normalized brightness.
    2. **Wavenumber k(r)** — CWT ridge estimate in rad/km.
    3. **Phase psi(r)** — cumulative phase from the NN in radians.
    4. **Amplitude A_V(r)** — NN result; LS result overlaid if provided.

    Args:
        r (numpy.ndarray): 1-D array of radial positions in km.
        IF_data (numpy.ndarray): 1-D array of observed I/F values.
        k (numpy.ndarray): 1-D array of wavenumbers in rad/km.
        psi (numpy.ndarray): 1-D array of cumulative phase in radians.
        A_V_nn (numpy.ndarray): 1-D array of NN vertical amplitudes in km.
        A_V_ls (numpy.ndarray | None): 1-D array of LS vertical amplitudes
            in km.  If ``None``, only the NN curve is plotted in panel 4.
            Defaults to ``None``.
        wave_num (int | None): Wave number used in the figure title and
            filename.  Defaults to ``None``.
        save_dir (str | None): Directory in which to save the figure.
            Defaults to ``config.PLOTS_DIR``.
    """
    if save_dir is None:
        save_dir = config.PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    wave_label = f"Wave {wave_num}" if wave_num is not None else "Wave"
    title = f"Rev {config.IMG_REV} · Image {config.IMG_NUM} – {wave_label} Summary"

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=12)

    # Panel 1: I/F profile
    axs[0].plot(r, IF_data, linewidth=0.9)
    axs[0].axhline(0, color="black", linestyle=":", linewidth=0.8)
    axs[0].set_ylabel("I/F")
    axs[0].grid(True, alpha=0.3)

    # Panel 2: wavenumber k(r)
    axs[1].plot(r, k, linewidth=0.9)
    axs[1].set_ylabel("k (rad/km)")
    axs[1].grid(True, alpha=0.3)

    # Panel 3: phase psi(r)
    axs[2].plot(r, psi, linewidth=0.9)
    axs[2].axhline(0, color="black", linestyle=":", linewidth=0.8)
    axs[2].set_ylabel("ψ (rad)")
    axs[2].grid(True, alpha=0.3)

    # Panel 4: amplitude A_V
    axs[3].plot(r, A_V_nn * 1000.0, label="NN", linewidth=1.2)
    if A_V_ls is not None:
        axs[3].plot(r, A_V_ls * 1000.0, label="LS", linestyle="--", linewidth=1.2)
        axs[3].legend(frameon=False)
    axs[3].axhline(0, color="black", linestyle=":", linewidth=0.8)
    axs[3].set_ylabel("A_V (m)")
    axs[3].set_xlabel("Radius (km)")
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()

    wave_tag = f"w{wave_num}" if wave_num is not None else "w"
    save_path = os.path.join(save_dir, config.IMG_PRFX + f"{wave_tag}_summary.png")
    fig.savefig(save_path, dpi=200)
    print(f"Saved summary: {save_path}")
    plt.show()
