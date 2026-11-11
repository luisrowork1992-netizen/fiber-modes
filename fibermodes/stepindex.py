"""Linearly polarised (LP) modes of a weakly guiding step-index fibre.

The LP_lm eigenvalue equation (Gloge, 1971) written without poles is::

    u J_{l+1}(u) K_l(w) - w K_{l+1}(w) J_l(u) = 0,   u^2 + w^2 = V^2

where ``u = a k0 sqrt(n1^2 - neff^2)`` and ``w = a k0 sqrt(neff^2 - n2^2)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, kve

__all__ = [
    "StepIndexFiber",
    "LPMode",
    "numerical_aperture",
    "v_number",
    "cutoff_v",
    "marcuse_mfd",
]


def numerical_aperture(n_core: float, n_clad: float) -> float:
    """``NA = sqrt(n_core^2 - n_clad^2)``."""
    if n_core <= n_clad:
        raise ValueError("core index must exceed cladding index")
    return math.sqrt(n_core**2 - n_clad**2)


def v_number(radius: float, wavelength: float, n_core: float, n_clad: float) -> float:
    """Normalised frequency ``V = 2 pi a NA / lambda``."""
    return 2 * math.pi * radius * numerical_aperture(n_core, n_clad) / wavelength


def cutoff_v(l: int, m: int) -> float:
    """Cut-off V value of LP_lm, the (m)th zero of J_{l-1} (LP01 has no cut-off)."""
    if l < 0 or m < 1:
        raise ValueError("need l >= 0 and m >= 1")
    if l == 0:
        # zeros of J_{-1} = -J_1, counting V = 0 as the first
        return 0.0 if m == 1 else float(jn_zeros(1, m - 1)[-1])
    return float(jn_zeros(l - 1, m)[-1])


def marcuse_mfd(V: float, radius: float) -> float:
    """Marcuse (1977) approximation of the LP01 mode-field *diameter*.

    ``w / a = 0.65 + 1.619 V^-1.5 + 2.879 V^-6``, accurate to about 1 %
    for 1.2 < V < 2.4.
    """
    return 2 * radius * (0.65 + 1.619 * V**-1.5 + 2.879 * V**-6)


def _char(u, V: float, l: int):
    """Pole-free characteristic function; vectorised over ``u``.

    Divided by K_{l+1}(w) (positive) to keep values bounded as w -> 0;
    kve scales both K's by the same exp(w), so the ratio is unchanged.
    """
    u = np.asarray(u, dtype=float)
    w = np.sqrt(np.maximum(V * V - u * u, 0.0))
    ratio = kve(l, w) / kve(l + 1, w)
    return u * jv(l + 1, u) * ratio - w * jv(l, u)


@dataclass(frozen=True)
class LPMode:
    l: int
    m: int
    u: float
    w: float
    b: float
    n_eff: float

    @property
    def name(self) -> str:
        return f"LP{self.l}{self.m}"

    @property
    def degeneracy(self) -> int:
        """Number of field patterns: 2 polarisations x (1 if l == 0 else 2 orientations)."""
        return 2 if self.l == 0 else 4


@dataclass(frozen=True)
class StepIndexFiber:
    """A step-index fibre with core radius ``radius`` (same unit as wavelength)."""

    radius: float
    n_core: float
    n_clad: float

    @property
    def na(self) -> float:
        return numerical_aperture(self.n_core, self.n_clad)

    def V(self, wavelength: float) -> float:
        return v_number(self.radius, wavelength, self.n_core, self.n_clad)

    def single_mode_cutoff(self) -> float:
        """Wavelength above which only LP01 is guided."""
        return 2 * math.pi * self.radius * self.na / cutoff_v(1, 1)

    def modes_of_order(self, l: int, wavelength: float, samples_per_unit: int = 40) -> list[LPMode]:
        """Guided LP_lm modes for one azimuthal order ``l`` (m = 1, 2, ...)."""
        V = self.V(wavelength)
        out = []
        for m, u in enumerate(_roots(V, l, samples_per_unit), start=1):
            w = math.sqrt(V * V - u * u)
            b = (w / V) ** 2
            n_eff = math.sqrt(self.n_clad**2 + b * (self.n_core**2 - self.n_clad**2))
            out.append(LPMode(l, m, u, w, b, n_eff))
        return out

    def modes(self, wavelength: float, samples_per_unit: int = 40) -> list[LPMode]:
        """All guided LP modes at ``wavelength``, sorted by decreasing n_eff."""
        found: list[LPMode] = []
        l = 0
        while True:
            order = self.modes_of_order(l, wavelength, samples_per_unit)
            if not order:
                break
            found.extend(order)
            l += 1
        return sorted(found, key=lambda md: -md.n_eff)

    def mode_count(self, wavelength: float) -> int:
        """Total number of guided field patterns including degeneracy."""
        return sum(md.degeneracy for md in self.modes(wavelength))


def _roots(V: float, l: int, samples_per_unit: int) -> list[float]:
    # A mode of order l exists only above the LP_l1 cut-off.
    if V <= 0 or V <= cutoff_v(l, 1):
        return []
    # For l >= 1 the root u lies above the first zero of J_{l-1}; starting the
    # scan there also avoids the trivial (underflowing) zero of the function at u -> 0.
    lower = cutoff_v(l, 1) if l >= 1 else 0.0
    n = max(400, int((V - lower) * samples_per_unit))
    eps = 1e-9 * V
    grid = np.linspace(lower + eps, V - eps, n)
    vals = _char(grid, V, l)
    f = lambda x: float(_char(x, V, l))
    idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    return [brentq(f, grid[i], grid[i + 1], xtol=1e-14) for i in idx]
