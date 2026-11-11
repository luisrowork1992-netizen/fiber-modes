"""Material dispersion of fused silica from its Sellmeier equation.

Wavelengths in micrometres. Group index and dispersion use analytic
derivatives of the Sellmeier form, so there's no finite-difference noise.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "silica_index",
    "group_index",
    "material_dispersion",
    "zero_dispersion_wavelength",
    "C_UM_PER_PS",
]

# Malitson (1965)
_B = np.array([0.6961663, 0.4079426, 0.8974794])
_L = np.array([0.0684043, 0.1162414, 9.896161])  # resonance wavelengths, um
_C2 = _L**2

C_UM_PER_PS = 299.792458  # speed of light in um/ps


def silica_index(lam):
    lam = np.asarray(lam, dtype=float)
    l2 = lam[..., None] ** 2
    return np.sqrt(1 + np.sum(_B * l2 / (l2 - _C2), axis=-1))


def _dn(lam):
    """First and second derivatives dn/dlam, d2n/dlam2 (per um)."""
    lam = np.asarray(lam, dtype=float)
    l = lam[..., None]
    l2 = l**2
    d = l2 - _C2
    # n^2 = 1 + sum B l^2 / (l^2 - C)
    f1 = np.sum(-2 * _B * _C2 * l / d**2, axis=-1)                        # d(n^2)/dl
    f2 = np.sum(_B * _C2 * (6 * l2 + 2 * _C2) / d**3, axis=-1)            # d2(n^2)/dl2
    n = silica_index(lam)
    dn = f1 / (2 * n)
    d2n = (f2 - 2 * dn**2) / (2 * n)
    return dn, d2n


def group_index(lam):
    """Group index ``n_g = n - lam dn/dlam``."""
    dn, _ = _dn(lam)
    return silica_index(lam) - np.asarray(lam) * dn


def material_dispersion(lam):
    """Material dispersion parameter D in ps / (nm km)."""
    _, d2n = _dn(lam)
    D_ps_per_um2 = -np.asarray(lam) / C_UM_PER_PS * d2n   # ps / (um * um)
    # 1 ps/(um um) = 1 ps / (1e3 nm * 1e-9 km) = 1e6 ps/(nm km)
    return D_ps_per_um2 * 1e6


def zero_dispersion_wavelength(lo: float = 1.1, hi: float = 1.5) -> float:
    """Wavelength (um) at which the material dispersion of silica crosses zero."""
    return brentq(lambda x: float(material_dispersion(x)), lo, hi, xtol=1e-10)
