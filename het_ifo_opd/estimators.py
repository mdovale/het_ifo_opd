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

* :func:`demodulate` -- a complex-baseband (heterodyne + low-pass) demodulator
  that is the *primary* OPD estimator.  It shifts the tone to DC and low-passes
  to a narrow band, which makes it immune to spectral leakage from the very
  large low-frequency differential-phase wander (the ``nu0 * OPD(t) / c`` term,
  tens of cycles RMS) that would otherwise bias a rectangular-window lock-in
  through its ``1/f`` sidelobes.  It returns the instantaneous tone phasor
  ``z(t)`` -- hence the instantaneous OPD ``|z(t)|`` and the true tone phase --
  and chooses coherent vs incoherent integration from a measured coherence.

The analytic (coherent-integration) uncertainty is computed from the local
background noise PSD: for integration time ``T`` and one-sided noise PSD
``S_n(f0)`` the amplitude standard deviation is ``sigma_A = sqrt(S_n / T)``,
which equals the Cramer-Rao bound ``sigma * sqrt(2/N)`` for white noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve, firwin

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


@dataclass
class DemodResult:
    """Complex-baseband demodulation of the modulation tone.

    All amplitudes are zero-to-peak, in cycles, so that ``|z|`` is directly the
    phase-tone amplitude (and hence maps to an OPD via
    :func:`het_ifo_opd.physics.phase_cycles_to_opd`).
    """

    f0: float                    # heterodyne (tone) frequency [Hz]
    bandwidth: float             # one-sided low-pass bandwidth [Hz]
    t: np.ndarray                # baseband time vector [s]
    z: np.ndarray                # complex baseband phasor, |z| = amplitude [cyc]
    fs_bb: float                 # baseband sample rate [Hz]
    amp_inst: np.ndarray         # |z(t)| instantaneous amplitude [cyc]
    phase_inst: np.ndarray       # unwrapped arg z(t) [rad]
    noise_ms: float              # off-tone mean-square E|noise|^2 in baseband [cyc^2]
    n_eff: float                 # effective independent looks (T * 2 * bandwidth)
    amp_coherent: float          # coherent amplitude, const-offset removed [cyc]
    amp_incoherent: float        # noise-debiased incoherent amplitude [cyc]
    amp_unc: float               # coherent noise-floor uncertainty on amplitude [cyc]
    coherence: float             # amp_coherent / amp_incoherent, in [0, ~1]
    freq_offset: float           # fitted constant tone frequency offset from f0 [Hz]
    mode: str                    # "coherent" | "incoherent"
    amplitude: float             # headline (mode-selected) amplitude [cyc]
    numtaps: int                 # FIR length used


def _heterodyne_lowpass(phi, fs, f0, bandwidth, fs_bb_target):
    """Shift ``f0`` to DC, low-pass to ``+/-bandwidth``, and decimate.

    Returns ``(t_bb, z_bb, fs_bb, numtaps)`` where ``|z_bb|`` is the zero-to-peak
    tone amplitude (the factor of two undoes the ``A/2`` of real-signal
    down-conversion).  A linear-phase FIR is applied with :func:`fftconvolve`
    (``mode="same"``, so no net group delay) and the filter-transient edges are
    trimmed after decimation.
    """
    phi = np.asarray(phi, dtype=float)
    n = phi.size
    t = np.arange(n) / fs
    z = (phi - phi.mean()) * np.exp(-2j * np.pi * f0 * t)

    # FIR low-pass at the one-sided bandwidth; Kaiser for a deep, controlled
    # stopband so the huge out-of-band low-frequency wander cannot leak in.
    numtaps = int(round(6.0 * fs / bandwidth)) | 1
    numtaps = min(numtaps, (n // 2) * 2 - 1)
    fir = firwin(numtaps, bandwidth, fs=fs, window=("kaiser", 8.6))
    zf = 2.0 * fftconvolve(z, fir, mode="same")

    d = max(1, int(round(fs / fs_bb_target)))
    zf = zf[::d]
    t_bb = t[::d]
    fs_bb = fs / d
    edge = max(1, int(round((numtaps // 2) / d)))
    if zf.size > 2 * edge + 2:
        zf = zf[edge:-edge]
        t_bb = t_bb[edge:-edge]
    return t_bb, zf, fs_bb, numtaps


def demodulate(
    phi: np.ndarray,
    fs: float,
    f0: float,
    bandwidth: float = 0.5,
    fs_bb_target: Optional[float] = None,
    off_tone: float = 7.0,
    coherence_threshold: float = 0.7,
) -> DemodResult:
    """Complex-baseband demodulation of the tone at ``f0`` (primary estimator).

    Parameters
    ----------
    phi : np.ndarray
        Differential phase [cycles].
    fs : float
        Sampling frequency [Hz].
    f0 : float
        Tone frequency [Hz] (already refined, if desired).
    bandwidth : float
        One-sided low-pass bandwidth [Hz] retained around the tone.  It must be
        wide enough to pass any real amplitude/phase dynamics of the tone but
        narrow enough to reject the broadband differential-phase noise.
    fs_bb_target : float, optional
        Target baseband sample rate [Hz].  Defaults to ``8 * bandwidth``.
    off_tone : float
        Offset [Hz] of the noise-reference demodulation (``f0 +/- off_tone``),
        placed away from the tone and its harmonics, used to measure the
        in-band noise power for the incoherent debiasing and the uncertainty.
    coherence_threshold : float
        If the coherent/incoherent amplitude ratio is at least this value the
        tone is treated as phase-coherent and coherent integration is reported;
        otherwise the noise-debiased incoherent amplitude is reported.

    Returns
    -------
    DemodResult
    """
    if fs_bb_target is None:
        fs_bb_target = 8.0 * bandwidth

    t_bb, z, fs_bb, numtaps = _heterodyne_lowpass(
        phi, fs, f0, bandwidth, fs_bb_target
    )

    # Off-tone noise reference with the *same* filter.  Keep it below Nyquist and
    # flip to f0 - off_tone if f0 + off_tone would run past the usable band.
    nyq = fs / 2.0
    f_off = f0 + off_tone
    if f_off > 0.9 * nyq or f_off <= 0:
        f_off = f0 - off_tone
    _, z_off, _, _ = _heterodyne_lowpass(phi, fs, f_off, bandwidth, fs_bb_target)
    noise_ms = float(np.mean(np.abs(z_off) ** 2))

    amp_inst = np.abs(z)
    mean_sq = float(np.mean(amp_inst ** 2))
    amp_incoherent = float(np.sqrt(max(mean_sq - noise_ms, 0.0)))

    # Coherent amplitude allowing a constant tone frequency offset (a residual
    # linear phase ramp left by frequency refinement): de-rotate then average.
    phase_inst = np.unwrap(np.angle(z))
    slope, _ = np.polyfit(t_bb, phase_inst, 1)
    freq_offset = float(slope / (2.0 * np.pi))
    z_derot = z * np.exp(-1j * (slope * t_bb))
    amp_coherent = float(np.abs(np.mean(z_derot)))

    coherence = amp_coherent / amp_incoherent if amp_incoherent > 0 else 0.0

    duration = float(t_bb[-1] - t_bb[0]) if t_bb.size > 1 else 0.0
    n_eff = max(duration * 2.0 * bandwidth, 1.0)
    amp_unc = float(np.sqrt(noise_ms / n_eff))

    if coherence >= coherence_threshold:
        mode, amplitude = "coherent", amp_coherent
    else:
        mode, amplitude = "incoherent", amp_incoherent

    return DemodResult(
        f0=float(f0),
        bandwidth=float(bandwidth),
        t=t_bb,
        z=z,
        fs_bb=float(fs_bb),
        amp_inst=amp_inst,
        phase_inst=phase_inst,
        noise_ms=noise_ms,
        n_eff=float(n_eff),
        amp_coherent=amp_coherent,
        amp_incoherent=amp_incoherent,
        amp_unc=amp_unc,
        coherence=float(coherence),
        freq_offset=freq_offset,
        mode=mode,
        amplitude=float(amplitude),
        numtaps=int(numtaps),
    )
