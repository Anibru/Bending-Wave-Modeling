# scripts/plot_amplitudes.py
"""Produces NN amplitude comparison figures across all images of a revolution.

For each wave (1, 2, 5) two figures are saved:

- **Overlay**: all images' A_V curves on a single set of axes.
- **Stacked**: each image in its own thin subplot, arranged vertically,
  with a shared x-axis so radial positions align across images.

Run after all NN wave analysis scripts have been executed so that the .npz
amplitude files exist under ``outputs/Rev {IMG_REV}/Image {IMG_NUM}/``.

Usage::

    python scripts/plot_amplitudes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import matplotlib.pyplot as plt

import config


def _load_wave_data(image_dirs, rev, npz_suffix, wave_label):
    """Loads A_V and r arrays for one wave from all available image directories.

    Args:
        image_dirs (list[str]): Sorted list of image output directories to
            search.
        rev (str): Revolution identifier (e.g. ``"115"``).
        npz_suffix (str): Filename suffix of the npz file (e.g.
            ``"nn_wave1_amp.npz"``).
        wave_label (str): Human-readable wave name used in warning messages.

    Returns:
        list[tuple[str, numpy.ndarray, numpy.ndarray]]: Each element is
            ``(img_num, r, A_V)`` for images where the npz was found.
    """
    records = []
    for img_dir in image_dirs:
        img_num = os.path.basename(img_dir).replace("Image ", "")
        npz_path = os.path.join(img_dir, f"{rev}.{img_num}_" + npz_suffix)
        if not os.path.exists(npz_path):
            print(f"  [{wave_label}] Image {img_num}: npz not found, skipping")
            continue
        data = np.load(npz_path)
        records.append((img_num, data["r"], data["A_V"]))
    return records


def plot_overlay(records, wave_label, rev, save_dir):
    """Plots all images' A_V curves overlaid on a single axes.

    Args:
        records (list[tuple[str, numpy.ndarray, numpy.ndarray]]): Output of
            :func:`_load_wave_data`.
        wave_label (str): Human-readable wave name used in the figure title.
        rev (str): Revolution identifier used in the title and filename.
        save_dir (str): Directory in which to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Rev {rev} – {wave_label} NN Amplitude (A_V) Across Images")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("A_V (m)")
    ax.grid(True, alpha=0.3)

    for img_num, r, A_V in records:
        ax.plot(r, A_V * 1000.0, label=f"Image {img_num}")

    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.legend(frameon=False)
    plt.tight_layout()

    fname = f"{rev}_NN_{wave_label.replace(' ', '_')}_overlay.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=200)
    print(f"Saved overlay:  {os.path.join(save_dir, fname)}")
    plt.show()


def plot_stacked(records, wave_label, rev, save_dir):
    """Plots each image's A_V curve in its own thin subplot, stacked vertically.

    All subplots share the same x-axis so radial positions align when
    comparing across images.

    Args:
        records (list[tuple[str, numpy.ndarray, numpy.ndarray]]): Output of
            :func:`_load_wave_data`.
        wave_label (str): Human-readable wave name used in the figure title.
        rev (str): Revolution identifier used in the title and filename.
        save_dir (str): Directory in which to save the figure.
    """
    n = len(records)
    fig, axs = plt.subplots(
        n, 1,
        figsize=(10, 1.8 * n),
        sharex=True,
    )
    if n == 1:
        axs = [axs]

    fig.suptitle(f"Rev {rev} – {wave_label} NN Amplitude (A_V) per Image", fontsize=12)

    for ax, (img_num, r, A_V) in zip(axs, records):
        ax.plot(r, A_V * 1000.0, linewidth=1.0)
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set_ylabel(f"Img {img_num}\nA_V (m)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    axs[-1].set_xlabel("Radius (km)")
    plt.tight_layout()

    fname = f"{rev}_NN_{wave_label.replace(' ', '_')}_stacked.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=200)
    print(f"Saved stacked:  {os.path.join(save_dir, fname)}")
    plt.show()


REVOLUTIONS = ["114", "115"]

waves = [
    ("Wave 1", "nn_wave1_amp.npz"),
    ("Wave 2", "nn_wave2_amp.npz"),
    ("Wave 5", "nn_wave5_amp.npz"),
]


def _run_rev(rev):
    """Produces overlay and stacked amplitude figures for all images in *rev*.

    Scans ``outputs/Rev {rev}/Image */`` for per-wave ``.npz`` amplitude files.
    For each wave with at least one file available, saves both an overlay
    figure and a vertically stacked figure to ``plots/Rev {rev}/``.

    Args:
        rev (str): Revolution identifier string (e.g. ``"114"``).

    Side effects:
        Creates ``plots/Rev {rev}/`` if it does not exist.
        Saves two PNGs per wave (overlay + stacked) to that directory.
        Prints progress messages to stdout.
        Displays interactive matplotlib figures.
    """
    base_out = config._p("outputs", f"Rev {rev}")
    save_dir = config._p("plots", f"Rev {rev}")
    os.makedirs(save_dir, exist_ok=True)

    image_dirs = sorted(glob.glob(os.path.join(base_out, "Image *")))
    if not image_dirs:
        print(f"[Rev {rev}] No image directories found under {base_out} — skipping.")
        return

    print(f"\n=== Rev {rev}: {len(image_dirs)} image(s) found ===")

    for wave_label, npz_suffix in waves:
        records = _load_wave_data(image_dirs, rev, npz_suffix, wave_label)

        if not records:
            print(f"  No data for {wave_label} — skipping.")
            continue

        print(f"\n  {wave_label}: {len(records)} image(s) with data.")
        plot_overlay(records, wave_label, rev, save_dir)
        plot_stacked(records, wave_label, rev, save_dir)


def main():
    """Runs amplitude plots for all configured revolutions.

    Iterates over :data:`REVOLUTIONS` and calls :func:`_run_rev` for each.

    Side effects:
        Saves figures and prints progress for every revolution in
        :data:`REVOLUTIONS`.
    """
    for rev in REVOLUTIONS:
        _run_rev(rev)


if __name__ == "__main__":
    main()
