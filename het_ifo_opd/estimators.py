"""Signal-processing estimators for the modulation-tone amplitude.

Two independent estimators of the phase-tone amplitude ``A`` [cycles] at the
modulation frequency are provided:

* :func:`lockin_amplitude` -- a least-squares synchronous detector (digital
  lock-in).  For a single tone of known frequency this is the maximum-likelihood
  / minimum-variance unbiased estimator; nuisance regressors (a polynomial
  trend and a few harmonics) make it robust to slow drift and drive distortion.
  This is the *primary* estimator.

* :func:`single_bin_amplitude` -- an LPSD single-bin spectral estimate via
  speckit, used as an independent cross-check.

The analytic (coherent-integration) uncertainty is computed from the local
background noise PSD: for integration time ``T`` and one-sided noise PSD
``S_n(f0)`` the amplitude standard deviation is ``sigma_A = sqrt(S_n / T)``,
which equals the Cramer-Rao bound ``sigma * sqrt(2/N)`` for white noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from speckit import compute_single_bin, compute_spectrum


@dataclass
class LockinResult:
    """Result of a least-squares lock-in fit."""

    freq: float                 # tone frequency used [Hz]
    amplitude: float            # fundamental amplitude [cycles, zero-to-peak]
    phase: float                # fundamental phase [rad]
    amp_unc_fit: float          # amplitude uncertainty from fit covariance
    amp_unc_coherent: float     # amplitude uncertainty from local noise PSD
    residual_std: float         # std of fit residual [cycles]
    noise_psd: float            # local one-sided noise PSD at f0 [cyc^2/Hz]
    harmonic_amps: np.ndarray   # amplitudes of all fitted harmonics [cycles]
    duration: float             # integration time [s]


def _projection_power(x: np.ndarray, n: np.ndarray, fs: float, f: float) -> float:
    """Squared magnitude of the DTFT of ``x`` at frequency ``f``."""
    ph = (2.0 * np.pi / fs) * f * n
    a = np.dot(x, np.cos(ph))
    b = np.dot(x, np.sin(ph))
    return a * a + b * b


def refine_frequency(
    phi: np.ndarray,
    fs: float,
    f0: float,
    halfwidth: float = 0.5,
    n_grid: int = 2001,
) -> float:
    """Refine the tone frequency by maximising the periodogram near ``f0``.

    A Goertzel-style projection is evaluated on a fine grid; this is robust to
    small offsets between the drive oscillator and the digitizer clock.
    """
    n = np.arange(phi.size)
    x = phi - np.mean(phi)
    grid = np.linspace(f0 - halfwidth, f0 + halfwidth, n_grid)
    powers = np.array([_projection_power(x, n, fs, f) for f in grid])
    return float(grid[int(np.argmax(powers))])


def _design_matrix(t, f0, n_harmonics, detrend_order):
    cols = []
    for k in range(1, n_harmonics + 1):
        w = 2.0 * np.pi * k * f0
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    for p in range(detrend_order + 1):
        cols.append(t ** p)
    return np.vstack(cols).T


def local_noise_psd(
    residual: np.ndarray,
    fs: float,
    f0: float,
    band_halfwidth: float = 10.0,
    window: str = "kaiser",
) -> float:
    """Estimate the one-sided background noise PSD [cyc^2/Hz] near ``f0``.

    This must be evaluated on the **lock-in residual** (the time series with the
    tone and its harmonics already subtracted).  Because the coherent tone is
    removed in the time domain, its spectral leakage is removed too, so the PSD
    in a band around ``f0`` is the genuine noise floor that limits the amplitude
    estimate.  The median over the band is robust to any leftover spurious lines.
    """
    x = residual - np.mean(residual)
    res = compute_spectrum(x, fs, win=window, olap="default")
    f = res.f
    psd = res.psd

    lo, hi = f0 - band_halfwidth, f0 + band_halfwidth
    band = (f >= lo) & (f <= hi)
    if not np.any(band):
        # fall back to the nearest bin
        band = np.zeros_like(f, dtype=bool)
        band[int(np.argmin(np.abs(f - f0)))] = True
    return float(np.median(psd[band]))


def lockin_amplitude(
    phi: np.ndarray,
    fs: float,
    f0: float,
    n_harmonics: int = 3,
    detrend_order: int = 3,
    noise_band_halfwidth: float = 10.0,
    window: str = "kaiser",
) -> LockinResult:
    """Least-squares lock-in amplitude of the fundamental tone at ``f0``.

    Fits ``phi(t) = sum_k [a_k cos + b_k sin](2*pi*k*f0*t) + sum_p c_p t^p`` and
    returns the fundamental amplitude ``sqrt(a_1^2 + b_1^2)`` with two
    uncertainty estimates (fit-covariance and coherent/PSD-based).
    """
    n = phi.size
    t = np.arange(n) / fs
    T = n / fs

    A = _design_matrix(t, f0, n_harmonics, detrend_order)
    coef, *_ = np.linalg.lstsq(A, phi, rcond=None)
    resid = phi - A @ coef
    resid_std = float(np.std(resid, ddof=A.shape[1]))

    harmonic_amps = np.array(
        [np.hypot(coef[2 * k], coef[2 * k + 1]) for k in range(n_harmonics)]
    )
    a1, b1 = coef[0], coef[1]
    amplitude = float(np.hypot(a1, b1))
    phase = float(np.arctan2(-b1, a1))  # phi ~ amplitude*cos(w t + phase)

    # Fit-covariance uncertainty on the amplitude (propagated from a1, b1).
    # cov(coef) = sigma^2 (A^T A)^-1 ; here noise treated as white at resid_std.
    ata_inv = np.linalg.inv(A.T @ A)
    cov = resid_std ** 2 * ata_inv
    if amplitude > 0:
        grad = np.array([a1, b1]) / amplitude
        var_amp = grad @ cov[:2, :2] @ grad
        amp_unc_fit = float(np.sqrt(max(var_amp, 0.0)))
    else:
        amp_unc_fit = float(np.sqrt(np.mean(np.diag(cov[:2, :2]))))

    # Coherent uncertainty from the local background noise PSD, estimated on the
    # residual (tone removed) so that tone leakage does not inflate the floor.
    s_n = local_noise_psd(
        resid, fs, f0,
        band_halfwidth=noise_band_halfwidth,
        window=window,
    )
    amp_unc_coherent = float(np.sqrt(s_n / T))

    return LockinResult(
        freq=float(f0),
        amplitude=amplitude,
        phase=phase,
        amp_unc_fit=amp_unc_fit,
        amp_unc_coherent=amp_unc_coherent,
        residual_std=resid_std,
        noise_psd=s_n,
        harmonic_amps=harmonic_amps,
        duration=T,
    )


def single_bin_amplitude(
    phi: np.ndarray,
    fs: float,
    f0: float,
    fres: float = 0.1,
    window: str = "kaiser",
) -> float:
    """Independent spectral cross-check of the tone amplitude [cycles].

    Uses an LPSD single-bin estimate; for a coherent sinusoid the one-sided
    power spectrum value equals ``A^2/2``, hence ``A = sqrt(2 * ps)``.
    """
    x = phi - np.mean(phi)
    res = compute_single_bin(x, fs, freq=f0, fres=fres, win=window)
    ps = float(np.atleast_1d(res.ps)[0])
    return float(np.sqrt(2.0 * ps))


def segmented_amplitudes(
    phi: np.ndarray,
    fs: float,
    f0: float,
    n_segments: int,
    n_harmonics: int = 3,
    detrend_order: int = 3,
) -> np.ndarray:
    """Lock-in amplitude on each of ``n_segments`` contiguous blocks.

    Used to assess stationarity and to provide an empirical uncertainty on the
    full-record estimate (scatter of the mean).
    """
    n = phi.size
    seg_len = n // n_segments
    if seg_len < int(10 * fs / f0):  # need many tone cycles per segment
        n_segments = max(1, n // int(10 * fs / f0))
        seg_len = n // n_segments

    amps = []
    t_full = np.arange(n) / fs
    for s in range(n_segments):
        sl = slice(s * seg_len, (s + 1) * seg_len)
        seg = phi[sl]
        t = t_full[sl] - t_full[sl][0]
        A = _design_matrix(t, f0, n_harmonics, detrend_order)
        coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
        amps.append(np.hypot(coef[0], coef[1]))
    return np.asarray(amps)
