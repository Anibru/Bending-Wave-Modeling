# scripts/run_all.py
"""Runs the full analysis pipeline for every image profile in data_profiles/.

Scans ``data_profiles/`` for files matching the pattern
``{REV}_profile_{IMG_NUM}_norm.sav``, then processes each one in sequence
using the same pipeline as ``scripts/run_analysis.py``.

Usage::

    python scripts/run_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import glob
import traceback

import config

# Minimum points required in a wave region to run analysis on it.
MIN_POINTS = 20

# Pattern: {REV}_profile_{IMG_NUM}_norm.sav
_PROFILE_RE = re.compile(r"^(\d+)_profile_(\d+)_norm\.sav$")


def discover_profiles():
    """Finds all analysis-ready profile files in the data_profiles/ directory.

    Returns:
        list[tuple[str, str]]: Sorted list of ``(rev, img_num)`` string pairs,
            one per ``.sav`` file whose name matches the expected pattern.
            Files that do not match the pattern are silently skipped.
    """
    profile_dir = config._p("data_profiles")
    sav_files = sorted(glob.glob(os.path.join(profile_dir, "*.sav")))

    profiles = []
    for path in sav_files:
        m = _PROFILE_RE.match(os.path.basename(path))
        if m:
            profiles.append((m.group(1), m.group(2)))
        else:
            print(f"  [skip] unrecognised filename: {os.path.basename(path)}")

    return profiles


def check_wave_coverage(cwt_file):
    """Counts data points within each wave's radial bounds from the CWT file.

    Args:
        cwt_file (str): Path to the ``.npz`` file written by the CWT pipeline.

    Returns:
        dict[int, int]: Mapping from wave number (1, 2, 5) to point count.

    Raises:
        FileNotFoundError: If ``cwt_file`` does not exist.
    """
    import numpy as np

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


def run_one(rev, img_num):
    """Runs the complete analysis pipeline for a single image.

    Updates ``config`` to point at the given image, then runs CWT, LS, and
    NN analysis in sequence with automatic wave-coverage checking.

    Args:
        rev (str): Revolution identifier (e.g. ``"115"``).
        img_num (str): Image number (e.g. ``"210"``).
    """
    config.reconfigure(rev, img_num)

    print(f"\n{'=' * 60}")
    print(f"  Rev {rev}  ·  Image {img_num}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Step 1: CWT + ridge
    # ------------------------------------------------------------------
    print("\n--- Step 1: CWT + Ridge Detection ---")
    from analysis.cwt_pipeline import main as run_cwt
    run_cwt()

    # ------------------------------------------------------------------
    # Step 2: Wave coverage check
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

    for wave_num, has_wave in [(1, has_w1), (2, has_w2), (5, has_w5)]:
        label = f"Wave {wave_num}"
        if has_wave:
            print(f"\n--- Step 3: NN Analysis — {label} ---")
            run_nn(wave_num)
        else:
            print(f"\n  Skipping NN {label} (insufficient data).")

    print(f"\n  Outputs → {config.OUT_DIR}")
    print(f"  Plots   → {config.PLOTS_DIR}")


def main():
    """Discovers all profiles and runs the full pipeline for each one.

    Images that raise an exception are logged and skipped so that a single
    failure does not abort the entire batch.

    Side effects:
        Writes outputs and plots for every successfully processed image.
        Prints a summary of passed/failed images at the end.
    """
    profiles = discover_profiles()

    if not profiles:
        print("No profile files found in data_profiles/ — nothing to do.")
        return

    print(f"Found {len(profiles)} profile(s) to process.")

    passed, failed = [], []

    for rev, img_num in profiles:
        key = f"{rev}.{img_num}"
        if key not in config.B_EFF_TABLE:
            print(f"\n  [skip] Rev {rev} · Image {img_num}: "
                  f"no B_eff entry in config.B_EFF_TABLE — add {key!r} to process it.")
            failed.append((rev, img_num))
            continue
        try:
            run_one(rev, img_num)
            passed.append((rev, img_num))
        except Exception:
            print(f"\n  ERROR processing Rev {rev} · Image {img_num}:")
            traceback.print_exc()
            failed.append((rev, img_num))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Batch complete:  {len(passed)} passed,  {len(failed)} failed")
    if failed:
        print("  Failed images:")
        for rev, img_num in failed:
            print(f"    Rev {rev} · Image {img_num}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
