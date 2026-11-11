"""fibermodes: guided LP modes and dispersion of step-index optical fibres."""

from .dispersion import (
    group_index,
    material_dispersion,
    silica_index,
    zero_dispersion_wavelength,
)
from .stepindex import (
    LPMode,
    StepIndexFiber,
    cutoff_v,
    marcuse_mfd,
    numerical_aperture,
    v_number,
)

__version__ = "0.1.0"
