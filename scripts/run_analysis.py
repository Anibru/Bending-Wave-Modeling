# scripts/run_analysis.py
"""Master pipeline: runs CWT, LS, and NN analysis for the image set in config.py.

Edit ``config.py`` to target a different image before running this script.

Usage::

    python scripts/run_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config

# Minimum number of data points required in a wave region to analyse it.
# Must be >= 5 (Conv1d kernel size) and large enough for meaningful fitting.
MIN_POINTS = 20


def check_wave_coverage(cwt_file):
    """Counts data points within each wave's radial bounds from the CWT file.

    Args:
        cwt_file (str): Path to the ``.npz`` file written by the CWT pipeline.

    Returns:
        dict[int, int]: Mapping from wave number (1, 2, 5) to the number of
            data points whose radius falls strictly inside that wave's
            configured radial range.

    Raises:
        FileNotFoundError: If ``cwt_file`` does not exist.
    """
    if not os.path.exists(cwt_file):
        raise FileNotFoundError(f"CWT output not found: {cwt_file}")

    r = np.load(cwt_file)["r"]

    regions = {
        1: (config.WAVE1_RADIUS_MIN, config.WAVE1_RADIUS_MAX),
        2: (config.WAVE2_RADIUS_MIN, config.WAVE2_RADIUS_MAX),
        5: (config.WAVE5_RADIUS_MIN, config.WAVE5_RADIUS_MAX),
    }
    return {
        wave: int(np.sum((r > rmin) & (r < rmax)))
        for wave, (rmin, rmax) in regions.items()
    }


def main():
    """Runs the full analysis pipeline for the image specified in config.py.

    Execution steps:

    1. Run CWT + ridge detection (normalized input variant).
    2. Load the resulting ``cwt_ridge_result.npz`` and count data points in
       each wave region.  A wave is considered usable when it has at least
       ``MIN_POINTS`` data points.
    3. Run Least-Squares analysis:

       - All three waves  → :func:`analysis.ls_waves.main` with waves [1, 2, 5]
       - Waves 1 & 2 only → :func:`analysis.ls_waves.main` with waves [1, 2]
       - Fewer than 2 usable waves → skip with a warning.

    4. Run NN analysis for each usable wave independently via
       :func:`analysis.nn_waves.main`.

    Side effects:
        All output files and plots are written to the directories defined in
        ``config.OUT_DIR`` and ``config.PLOTS_DIR``.
        Prints a coverage summary and step headers to stdout.
        Displays interactive matplotlib figures for each analysis step.
    """
    print(f"{'=' * 60}")
    print(f"  Rev {config.IMG_REV}  ·  Image {config.IMG_NUM}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Step 1: CWT + ridge
    # ------------------------------------------------------------------
    print("--- Step 1: CWT + Ridge Detection ---")
    from analysis.cwt_pipeline import main as run_cwt
    run_cwt()

    # ------------------------------------------------------------------
    # Step 2: Check wave coverage
    # ------------------------------------------------------------------
    coverage = check_wave_coverage(config.CWT_FILE)

    print("\nWave coverage:")
    usable = {}
    for wave, n in coverage.items():
        status = "OK" if n >= MIN_POINTS else f"SKIP (only {n} pts, need {MIN_POINTS})"
        print(f"  Wave {wave}: {n:4d} points  [{status}]")
        usable[wave] = n >= MIN_POINTS

    has_w1, has_w2, has_w5 = usable[1], usable[2], usable[5]

    # ------------------------------------------------------------------
    # Step 3: Least-Squares
    # ------------------------------------------------------------------
    print("\n--- Step 2: Least-Squares Analysis ---")
    from analysis.ls_waves import main as run_ls

    if has_w1 and has_w2 and has_w5:
        print("  Running LS for waves 1, 2, and 5.")
        run_ls(waves=[1, 2, 5])
    elif has_w1 and has_w2:
        print("  Wave 5 region insufficient — running LS for waves 1 and 2 only.")
        run_ls(waves=[1, 2])
    else:
        print("  WARNING: fewer than 2 usable wave regions — skipping LS analysis.")

    # ------------------------------------------------------------------
    # Step 4: NN analysis per wave
    # ------------------------------------------------------------------
    from analysis.nn_waves import main as run_nn

    if has_w1:
        print("\n--- Step 3a: NN Analysis — Wave 1 ---")
        run_nn(1)
    else:
        print("\n  Skipping NN Wave 1 (insufficient data).")

    if has_w2:
        print("\n--- Step 3b: NN Analysis — Wave 2 ---")
        run_nn(2)
    else:
        print("\n  Skipping NN Wave 2 (insufficient data).")

    if has_w5:
        print("\n--- Step 3c: NN Analysis — Wave 5 ---")
        run_nn(5)
    else:
        print("\n  Skipping NN Wave 5 (insufficient data).")

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete for Rev {config.IMG_REV} · Image {config.IMG_NUM}")
    print(f"  Outputs → {config.OUT_DIR}")
    print(f"  Plots   → {config.PLOTS_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
