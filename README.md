# Bending-Wave Modeling

> **AI Disclosure:** In-code documentation (docstrings) and this README were
> written with the partial assistance of [Claude](https://claude.ai) (Anthropic).

Photometric analysis of Cassini ISS occultation profiles to extract the
vertical amplitude, radial wavenumber, and phase of density/bending waves in
Saturn's rings. I/F is extracted from a normalized radial brightness profile
using a wavelet transform. Three analysis methods are applied to each image:

1. **CWT** — Continuous Wavelet Transform ridge detection to estimate k(r).
2. **Least-Squares (LS)** — Sliding-window two-term fit of the photometric model.
3. **Physics-informed Neural Network (NN)** — Conv1D network trained with MSE +
   first- and second-derivative smoothness penalties.

Data is from Cassini ISS, Revolutions 114 and 115.

Wave regions: **Wave 1** (~117,960–118,070 km), **Wave 2** (~118,350–118,460 km),
**Wave 5** (~119,460–119,560 km).

---

## Directory structure

```
data_analysis/
├── config.py                  # Central configuration (edit this per image)
├── data_profiles/             # Input .sav occultation profiles
├── outputs/                   # Intermediate .npz files (CWT, LS, NN)
├── plots/                     # All saved figures
│
├── helpers/                   # Shared low-level functions
│   ├── cwt.py                 # run_cwt_and_ridge, prepare_power
│   ├── nn_model.py            # ContextAmpCSNet, build_phase, build_features
│   ├── physics.py             # theta_max_from_IF, reconstruct_and_compare_IF, fit_stats
│   ├── plotting.py            # plot_IF_fit, plot_wave_summary
│   └── utils.py               # edges_from_centers, smooth_moving_average, safe_inv
│
├── analysis/                  # Per-step analysis modules
│   ├── cwt_pipeline.py        # CWT + ridge detection
│   ├── ls_waves.py            # Least-squares amplitude fitting
│   └── nn_waves.py            # Neural-network amplitude fitting
│
└── scripts/                   # Entry-point scripts
    ├── run_analysis.py        # Full pipeline for the image in config.py
    ├── run_all.py             # Full pipeline for every profile in data_profiles/
    ├── plot_amplitudes.py     # Cross-image amplitude overlay/stack plots
    └── clean_old_plots.py     # Removes stale plot files from older runs
```

---

## Setup

```bash
pip install numpy scipy matplotlib pywavelets torch
```

All scripts must be run from the **project root** (`data_analysis/`).

---

## Configuration

Open `config.py` and verify these values before running any script:

| Variable | Purpose |
|---|---|
| `IMG_REV` / `IMG_NUM` | Target image (used by `run_analysis.py`) |
| `B_EFF_TABLE` | Effective elevation angle B_eff (degrees) for each known image |
| `NN_SMOOTH_LOSS_COEF` | NN first-derivative smoothness penalty weight |
| `NN_CURV_LOSS_COEF` | NN second-derivative curvature penalty weight |

### Known B_eff values

| Image | B_eff (°) |
|---|---|
| 114.34 | −0.480 |
| 114.35 | −0.480 |
| 114.36 | −0.480 |
| 114.37 | −0.480 |
| 114.62 | −0.480 |
| 114.63 | −0.480 |
| 115.109 | 0.293 |
| 115.140 | 0.331 |
| 115.150 | 0.351 |
| 115.160 | 0.376 |
| 115.170 | 0.408 |
| 115.190 | 0.498 |
| 115.210 | 0.645 |
| 115.217 | 0.714 |

To add a new image, place its `.sav` file in `data_profiles/` using the naming
convention `{REV}_profile_{IMG_NUM}_norm.sav` and add an entry to `B_EFF_TABLE`
in `config.py`:

```python
B_EFF_TABLE = {
    ...
    "116.42": 0.812,   # new entry
}
```

---

## Running the scripts

### Full pipeline — single image

Runs CWT → LS → NN for the image set in `config.py`.
Wave 5 is automatically skipped if the profile does not cover that radial region.

```bash
python scripts/run_analysis.py
```

To target a different image, edit `IMG_REV` and `IMG_NUM` in `config.py` first:

```python
# config.py
IMG_REV = "114"
IMG_NUM = "34"
```

```bash
python scripts/run_analysis.py
```

---

### Full pipeline — all images

Scans `data_profiles/` for every `{REV}_profile_{IMG_NUM}_norm.sav` file and runs
the complete pipeline for each one in sequence. Images with no `B_EFF_TABLE` entry
are skipped with a warning; errors on individual images are caught and logged so
the batch continues uninterrupted.

```bash
python scripts/run_all.py
```

Example output:

```
Found 14 profile(s) to process.

============================================================
  Rev 114  ·  Image 34
============================================================
--- Step 1: CWT + Ridge Detection ---
Wave coverage:
  Wave 1:  87 points  [OK]
  Wave 2:  91 points  [OK]
  Wave 5:   0 points  [SKIP (only 0 pts, need 20)]
--- Step 2: Least-Squares Analysis ---
  Wave 5 region insufficient — running LS for waves 1 and 2 only.
...
============================================================
  Batch complete:  14 passed,  0 failed
============================================================
```

---

### Individual analysis steps

Each step can be run on its own. The LS and NN steps both require that the CWT
output `.npz` already exists in `outputs/Rev {REV}/Image {NUM}/`. LS and NN are
independent of each other.

#### CWT + ridge detection

```bash
python analysis/cwt_pipeline.py
```

Writes `cwt_ridge_result.npz` to `outputs/Rev {REV}/Image {NUM}/`.

#### Least-squares amplitude fit

```bash
# All waves covered by the current profile
python analysis/ls_waves.py

# Specific wave(s) only
python analysis/ls_waves.py --waves 1 2
python analysis/ls_waves.py --waves 5
```

#### Neural-network amplitude fit

Accepts a single wave number (1, 2, or 5). Only the CWT output is required.

```bash
python analysis/nn_waves.py 1   # Wave 1
python analysis/nn_waves.py 2   # Wave 2
python analysis/nn_waves.py 5   # Wave 5
```

For example, to run only the NN for Wave 5 on Rev 114 Image 62, set
`IMG_REV = "114"` / `IMG_NUM = "62"` in `config.py`, ensure the CWT output
exists, then:

```bash
python analysis/nn_waves.py 5
```

---

### Cross-image amplitude plots

Produces overlay and stacked A_V comparison figures across all images of
Revolutions 114 and 115. Run after NN analysis has completed for all images.

```bash
python scripts/plot_amplitudes.py
```

Saves to `plots/Rev {REV}/` (one pair of figures per wave):

| File | Description |
|---|---|
| `{REV}_NN_Wave_1_overlay.png` | All images' A_V curves on one axis |
| `{REV}_NN_Wave_1_stacked.png` | One thin subplot per image, shared x-axis |


---

## Output structure

```
outputs/
└── Rev 115/
    └── Image 210/
        ├── cwt_ridge_result.npz        # CWT ridge (r, k, power, ...)
        ├── 115.210_ls_wave1_amp.npz    # LS amplitude arrays
        ├── 115.210_ls_wave2_amp.npz
        ├── 115.210_nn_wave1_amp.npz    # NN amplitude arrays
        └── 115.210_nn_wave2_amp.npz

plots/
└── Rev 115/
    ├── 115_NN_Wave_1_overlay.png       # Cross-image summary (all images)
    ├── 115_NN_Wave_1_stacked.png
    └── Image 210/
        ├── 115.210_wavelet_power.png
        ├── 115.210_ridge_trace.png
        ├── LS/
        │   ├── 115.210_w1_A_V.png
        │   ├── 115.210_w1_IF_fit.png
        │   └── ...
        └── NN/
            ├── 115.210_w1_A_V.png
            ├── 115.210_w1_IF_fit.png
            ├── 115.210_w1_summary.png  # 4-panel: I/F, k, psi, A_V (NN + LS)
            └── ...
```

---

## Wave model

The photometric model fitted to each wave region is:

```
IF(r) = IF0 · [1 − cot(B_eff(r)) · k(r) · (A_c·cos(ψ) + A_s·sin(ψ))]
```

where `ψ(r) = ∫ k(r') dr'` is the cumulative phase from the CWT ridge,
`A_c` and `A_s` are the cosine/sine amplitude components, and
`A_V = sqrt(A_c² + A_s²)` is the vertical amplitude in km.
