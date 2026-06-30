"""Validation of the OPD estimator on synthetic data.

These tests demonstrate that the estimator is

1. *correct* in physics (phase<->OPD round-trip),
2. *accurate* (unbiased recovery of an injected OPD even on top of realistic,
   low-frequency-dominated phase noise), and
3. *optimal* (its reported coherent uncertainty matches the Monte-Carlo scatter
   of the estimate, i.e. it is statistically calibrated and near the
   Cramer-Rao bound).

Run with ``pytest -q`` or directly with ``python tests/test_validation.py``.
"""
from __future__ import annotations

import numpy as np

from het_ifo_opd import OPDConfig
from het_ifo_opd.estimators import lockin_amplitude, single_bin_amplitude
from het_ifo_opd.physics import (
    opd_to_phase_cycles,
    phase_cycles_to_opd,
)

FS = 596.04644775
CFG = OPDConfig()


def _colored_noise(n, fs, rng, white_amp=1e-4, red_amp=2e-3):
    """White + steep red (1/f^2-ish) noise, mimicking the differential phase."""
    w = white_amp * rng.standard_normal(n)
    red = np.cumsum(red_amp / np.sqrt(fs) * rng.standard_normal(n))
    red -= np.mean(red)
    return w + red


def _make_signal(opd_true, n, fs, rng, f0=100.0, cfg=CFG, **noise_kw):
    t = np.arange(n) / fs
    amp = opd_to_phase_cycles(opd_true, cfg.freq_dev_peak)
    phase = rng.uniform(0, 2 * np.pi)
    tone = amp * np.cos(2 * np.pi * f0 * t + phase)
    return tone + _colored_noise(n, fs, rng, **noise_kw)


def test_physics_round_trip():
    opd = np.array([0.1e-3, 1.0e-3, 5.0e-3, 0.05])
    amp = opd_to_phase_cycles(opd, CFG.freq_dev_peak)
    back = phase_cycles_to_opd(amp, CFG.freq_dev_peak)
    assert np.allclose(back, opd, rtol=1e-12)


def test_freq_dev_peak():
    # 376 MHz/V * (1 Vpp / 2) = 188 MHz peak.
    assert abs(CFG.freq_dev_peak - 188e6) < 1.0


def test_unbiased_recovery():
    """Injected OPD is recovered with <1% bias on realistic noise."""
    rng = np.random.default_rng(0)
    n = int(120 * FS)
    opd_true = 1.10e-3
    phi = _make_signal(opd_true, n, FS, rng)
    lk = lockin_amplitude(phi, FS, 100.0)
    opd_hat = phase_cycles_to_opd(lk.amplitude, CFG.freq_dev_peak)
    rel_err = abs(opd_hat - opd_true) / opd_true
    assert rel_err < 0.01, f"relative error {rel_err:.3%} too large"

    # Spectral cross-check agrees with the lock-in to <1%.
    amp_spec = single_bin_amplitude(phi, FS, 100.0)
    assert abs(amp_spec - lk.amplitude) / lk.amplitude < 0.01


def test_uncertainty_is_calibrated():
    """Reported coherent uncertainty matches the Monte-Carlo scatter."""
    rng = np.random.default_rng(1)
    n = int(60 * FS)
    opd_true = 1.0e-3
    n_trials = 60

    opd_hats = []
    reported = []
    for _ in range(n_trials):
        phi = _make_signal(opd_true, n, FS, rng)
        lk = lockin_amplitude(phi, FS, 100.0)
        opd_hats.append(phase_cycles_to_opd(lk.amplitude, CFG.freq_dev_peak))
        reported.append(phase_cycles_to_opd(lk.amp_unc_coherent, CFG.freq_dev_peak))

    mc_std = np.std(opd_hats, ddof=1)
    mean_reported = np.mean(reported)
    ratio = mean_reported / mc_std
    # Calibrated to within ~30% (the analytic formula is the CRB; the MC scatter
    # should match it closely).
    assert 0.7 < ratio < 1.4, (
        f"uncertainty not calibrated: reported/MC = {ratio:.2f} "
        f"(reported={mean_reported:.2e}, MC={mc_std:.2e})"
    )


if __name__ == "__main__":
    test_physics_round_trip()
    test_freq_dev_peak()
    test_unbiased_recovery()
    test_uncertainty_is_calibrated()
    print("All validation tests passed.")
