# config.py
"""Central configuration for Saturn ring bending-wave analysis.

Edit the constants in this module to target a different image or
observation geometry before running any pipeline script.

Sections:
    - **Paths**: Data file, output directories, and plot directories.
    - **Image metadata**: Revision/image identifiers and filename prefix.
    - **Geometry**: Solar elevation angle ``B_sun``, observer elevation range
      ``B_obs``, and effective elevation angle ``B_eff`` used in the
      photometric model.
    - **Reference bands**: Radial ranges used for global I/F normalization.
    - **Wave regions**: Radial extents of waves 1, 2, and 5.
    - **CWT / ridge parameters**: Wavenumber range, grid resolution, Morlet
      omega, and bandwidth candidates for the complex Morlet wavelet.
    - **Ridge masking**: Optional wavenumber-band suppression applied during
      wave 2 ridge detection.
    - **Plotting options**: Colormap normalization and dynamic-range clipping.
"""
import os
import numpy as np

# Root of the project (directory containing this file)
_ROOT = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    """Joins path parts relative to the project root."""
    return os.path.join(_ROOT, *parts)

DATA_PATH = _p("data_profiles", "115_profile_210_norm.sav")

# EDIT FOR NEW IMAGE
IMG_REV = "115"
IMG_NUM = "210"
IMG_PRFX = IMG_REV + "." + IMG_NUM + "_"

CWT_FILE = _p("outputs", "Rev " + IMG_REV, "Image " + IMG_NUM, "cwt_ridge_result.npz")
OUT_DIR  = _p("outputs", "Rev " + IMG_REV, "Image " + IMG_NUM)
PLOTS_DIR = _p("plots",  "Rev " + IMG_REV, "Image " + IMG_NUM)


# Per-image effective elevation angle B_eff in degrees.
# Key format: "{REV}.{IMG_NUM}"
# Add new images here as they become available.
B_EFF_TABLE = {
    "114.34": -0.480,
    "114.35": -0.480,
    "114.36": -0.480,
    "114.37": -0.480,
    "114.62": -0.480,
    "114.63": -0.480,
    "115.109": 0.293,
    "115.140": 0.331,
    "115.150": 0.351,
    "115.160": 0.376,
    "115.170": 0.408,
    "115.190": 0.498,
    "115.210": 0.645,
    "115.217": 0.714,
}


def reconfigure(rev, img_num):
    """Updates all image-dependent config values for a new rev/image pair.

    Looks up ``B_EFF_DEG`` from ``B_EFF_TABLE`` using the key
    ``"{rev}.{img_num}"``.  If no entry is found a ``KeyError`` is raised so
    that missing geometry values are caught early rather than silently using
    a stale value.

    Call this before running any analysis function when processing multiple
    images in a loop.  All module-level constants are updated in-place so
    that subsequent references to ``config`` see the new values.

    Args:
        rev (str): Revolution identifier (e.g. ``"115"``).
        img_num (str): Image number (e.g. ``"210"``).

    Raises:
        KeyError: If ``"{rev}.{img_num}"`` is not present in ``B_EFF_TABLE``.
    """
    import sys
    module = sys.modules[__name__]

    key = f"{rev}.{img_num}"
    if key not in B_EFF_TABLE:
        raise KeyError(
            f"No B_eff entry for {key!r}. Add it to config.B_EFF_TABLE."
        )

    b_eff_deg = B_EFF_TABLE[key]

    module.IMG_REV    = rev
    module.IMG_NUM    = img_num
    module.IMG_PRFX   = rev + "." + img_num + "_"
    module.DATA_PATH  = _p("data_profiles", f"{rev}_profile_{img_num}_norm.sav")
    module.CWT_FILE   = _p("outputs", f"Rev {rev}", f"Image {img_num}", "cwt_ridge_result.npz")
    module.OUT_DIR    = _p("outputs", f"Rev {rev}", f"Image {img_num}")
    module.PLOTS_DIR  = _p("plots",   f"Rev {rev}", f"Image {img_num}")
    module.B_EFF_DEG  = b_eff_deg
    module.B_EFF      = b_eff_deg * np.pi / 180.0


# geometry (from image metadata)
B_EFF_DEG = 0.645
B_EFF = B_EFF_DEG * np.pi / 180.0

# global I0/F reference bands (for first normalization)
GLOBAL_BAND_1 = (118650.0, 118850.0)
GLOBAL_BAND_2 = (119000.0, 119200.0)

# wave 1 region
WAVE1_RADIUS_MIN = 117960.0
WAVE1_RADIUS_MAX = 118070.0

# wave 2 region
WAVE2_RADIUS_MIN = 118350.0
WAVE2_RADIUS_MAX = 118460.0

# wave 5 region
WAVE5_RADIUS_MIN = 119460.0
WAVE5_RADIUS_MAX = 119560.0

# wave 5 local baseline band (for re-normalization)
WAVE5_BASELINE_BAND = (119450.0, 119724.0)

# CWT / ridge parameters
KMIN_CYC = 0.02          # cycles/km
KMIN = 0.02 * 2 * np.pi
KMAX_CYC = 0.3           # cycles/km
KMAX = 0.3 * 2 * np.pi
NUM_K = 10000
MORLET_W = 6.0
BANDWIDTH_GRID = [0.8, 1.0, 1.2, 1.5, 2.0]

# ridge masking for wave 2
RIDGE_MASK_RMIN_2 = 118497.0
RIDGE_MASK_RMAX_2 = 118590.0
RIDGE_MASK_KMIN_2_CYC = 0.05   # cycles/km
RIDGE_MASK_KMIN_2 = 0.05 * 2 * np.pi
RIDGE_MASK_KMAX_2_CYC = 0.095  # cycles/km
RIDGE_MASK_KMAX_2 = 0.095 * 2 * np.pi

# NN training loss coefficients
NN_SMOOTH_LOSS_COEF = 1e4   # first-derivative smoothness penalty weight
NN_CURV_LOSS_COEF   = 1e3   # second-derivative curvature penalty weight

# plotting options
USE_COLNORM = True
USE_PER_OCT = True
SCALE_MODE = "log"
CLIP_PCT = (2, 99.5)