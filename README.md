# fiber-modes

![tests](https://github.com/luisrowork1992-netizen/fiber-modes/actions/workflows/tests.yml/badge.svg)

`fibermodes` solves the **LP modes of weakly guiding step-index optical fibres** and computes the **material dispersion of fused silica**.

## Features

- Numerical aperture, V-number and single-mode cut-off wavelength
- A robust LP_lm eigenvalue solver: a pole-free form of the characteristic equation, scanned for sign changes and refined with Brent's method
- Per-mode `u`, `w`, normalised propagation constant `b`, and effective index
- Mode counting with polarisation and orientation degeneracy
- LP_lm cut-off V values from Bessel-function zeros
- Marcuse's mode-field-diameter approximation
- Silica group index and the material dispersion parameter D, using analytic Sellmeier derivatives with no finite differences
- Zero-dispersion wavelength

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/fiber-modes.git
cd fiber-modes
pip install -e .
```

## Quick start

```python
import fibermodes as fm

smf = fm.StepIndexFiber(radius=4.1, n_core=1.4504, n_clad=1.4447)   # lengths in um
print(smf.V(1.55))                       # ~2.14  -> single mode
print([m.name for m in smf.modes(1.55)]) # ['LP01']
print(smf.single_mode_cutoff())          # ~1.38 um

print(fm.zero_dispersion_wavelength())   # ~1.273 um for bulk silica
print(fm.material_dispersion(1.55))      # ~21.9 ps/(nm km)
```

## Notebook

[`notebooks/fiber_modes.ipynb`](notebooks/fiber_modes.ipynb) covers four topics:

- the b–V dispersion diagram
- single-mode behaviour of an SMF-28-like fibre
- mode count compared with V²/2
- the silica dispersion curve

## Tests

```bash
pip install pytest
pytest
```

The tests check the following:

- cut-off values: 2.405 for LP11 and 3.832 for LP21 and LP02
- single-mode operation at 1550 nm
- LP11 appearing just below the cut-off wavelength
- u² + w² = V²
- the large-V limit of LP01, where u → 2.405
- the V²/2 mode-count rule
- the Rudolph–Neumann approximation for b
- the silica zero-dispersion wavelength, near 1.27 µm, and D(1550 nm)

## Scope and limitations

- This is the scalar (weak-guidance) LP approximation, so it is valid when Δ = (n₁ − n₂)/n₁ ≪ 1.
- Waveguide dispersion is not included. The dispersion module covers **bulk silica material dispersion** only.

## References

- D. Gloge, "Weakly guiding fibers," *Appl. Opt.* 10, 2252 (1971).
- D. Marcuse, "Loss analysis of single-mode fiber splices," *Bell Syst. Tech. J.* 56, 703 (1977).
- I. H. Malitson, *J. Opt. Soc. Am.* 55, 1205 (1965).
- A. W. Snyder and J. D. Love, *Optical Waveguide Theory*, Chapman & Hall.

## License

MIT
