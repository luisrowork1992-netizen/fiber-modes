import math

import numpy as np
import pytest
from scipy.special import jv

import fibermodes as fm

# SMF-28-like fibre: 4.1 um core radius, NA ~ 0.12 (lengths in um)
SMF = fm.StepIndexFiber(radius=4.1, n_core=1.4504, n_clad=1.4447)


def test_na_and_v_number():
    na = fm.numerical_aperture(1.4504, 1.4447)
    assert na == pytest.approx(math.sqrt(1.4504**2 - 1.4447**2))
    assert fm.v_number(4.1, 1.55, 1.4504, 1.4447) == pytest.approx(2 * math.pi * 4.1 * na / 1.55)


def test_invalid_indices():
    with pytest.raises(ValueError):
        fm.numerical_aperture(1.44, 1.45)


def test_cutoff_values():
    assert fm.cutoff_v(0, 1) == 0.0
    assert fm.cutoff_v(1, 1) == pytest.approx(2.404826, abs=1e-6)
    assert fm.cutoff_v(2, 1) == pytest.approx(3.831706, abs=1e-6)
    assert fm.cutoff_v(0, 2) == pytest.approx(3.831706, abs=1e-6)


def test_smf_is_single_mode_at_1550():
    modes = SMF.modes(1.55)
    assert [m.name for m in modes] == ["LP01"]
    assert SMF.mode_count(1.55) == 2


def test_lp11_appears_below_cutoff_wavelength():
    lc = SMF.single_mode_cutoff()
    assert len(SMF.modes(lc * 1.02)) == 1
    names = [m.name for m in SMF.modes(lc * 0.95)]
    assert names == ["LP01", "LP11"]


def test_mode_solution_satisfies_eigen_equation():
    fib = fm.StepIndexFiber(radius=25.0, n_core=1.48, n_clad=1.46)
    for md in fib.modes(0.85):
        assert md.u**2 + md.w**2 == pytest.approx(fib.V(0.85) ** 2, rel=1e-10)
        assert 0 < md.b < 1
        assert fib.n_clad < md.n_eff < fib.n_core


def test_modes_lie_between_cutoffs():
    fib = fm.StepIndexFiber(radius=10.0, n_core=1.47, n_clad=1.46)
    V = fib.V(1.0)
    for md in fib.modes(1.0):
        assert fm.cutoff_v(md.l, md.m) < V


def test_large_v_limit_u_approaches_bessel_zero():
    # As V -> inf, LP_lm u -> m-th zero of J_l (here LP01 -> 2.4048)
    fib = fm.StepIndexFiber(radius=200.0, n_core=1.50, n_clad=1.40)
    lp01 = fib.modes_of_order(0, 1.0)[0]
    assert lp01.name == "LP01"
    assert lp01.u == pytest.approx(2.404826, rel=0.01)
    assert abs(jv(0, lp01.u)) < 0.02


def test_mode_count_approaches_v_squared_over_two():
    fib = fm.StepIndexFiber(radius=25.0, n_core=1.48, n_clad=1.46)
    V = fib.V(0.85)
    assert V > 20
    assert fib.mode_count(0.85) == pytest.approx(V**2 / 2, rel=0.1)


def test_lp01_b_matches_rudolph_neumann_approximation():
    # b ~ (1.1428 - 0.996 / V)^2, quoted as accurate to ~0.2 % for 1.5 < V < 2.5
    for V_target in (1.6, 2.0, 2.4):
        lam = 2 * math.pi * SMF.radius * SMF.na / V_target
        lp01 = SMF.modes(lam)[0]
        approx = (1.1428 - 0.996 / V_target) ** 2
        assert lp01.b == pytest.approx(approx, rel=0.01)


def test_marcuse_mfd_smf28_ballpark():
    # SMF-28 datasheet MFD at 1550 nm is about 10.4 um
    mfd = fm.marcuse_mfd(SMF.V(1.55), SMF.radius)
    assert 9.5 < mfd < 11.0


def test_silica_zero_dispersion_wavelength():
    zdw = fm.zero_dispersion_wavelength()
    assert 1.26 < zdw < 1.29


def test_silica_dispersion_at_1550():
    # Bulk silica material dispersion is about +21-22 ps/(nm km) at 1550 nm
    assert 20.5 < float(fm.material_dispersion(1.55)) < 23.0
    assert float(fm.material_dispersion(0.85)) < -80


def test_group_index_exceeds_phase_index():
    lam = np.linspace(0.6, 1.6, 11)
    assert np.all(fm.group_index(lam) > fm.silica_index(lam))
    assert float(fm.group_index(1.55)) == pytest.approx(1.4626, abs=1e-3)
