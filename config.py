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
import numpy as np

DATA_PATH = "data_profiles/115_profile_109_norm.sav"

# EDIT FOR NEW IMAGE
IMG_REV = "115"
IMG_NUM = "109"
IMG_PRFX = IMG_REV + "." + IMG_NUM + "_"

CWT_FILE = "outputs/Rev " + IMG_REV + "/Image " + IMG_NUM + "/cwt_ridge_result.npz"
OUT_DIR = "outputs/Rev " + IMG_REV + "/Image " + IMG_NUM
PLOTS_DIR = "plots/Rev " + IMG_REV + "/Image " + IMG_NUM

# geometry (from image metadata)
B_EFF_DEG = 0.29
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

# plotting options
USE_COLNORM = True
USE_PER_OCT = True
SCALE_MODE = "log"
CLIP_PCT = (2, 99.5)