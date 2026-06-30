"""Configuration for the HET IFO OPD estimation pipeline.

The :class:`OPDConfig` dataclass bundles the *physical* parameters of the
experiment (laser, actuator, modulation) together with the *analysis*
parameters that control the signal-processing.  All defaults match the FM1
data set described in the task statement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

# Speed of light in vacuum [m/s] (exact, SI).
C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class OPDConfig:
    """Physical and analysis parameters for an OPD measurement.

    Physics
    -------
    laser_wavelength : float
        Laser wavelength [m].  Used only for *reporting* equivalent quantities
        (e.g. carrier phase); the frequency-modulation OPD estimate itself is
        wavelength independent.
    actuator_tf : float
        Laser-frequency actuator transfer function [Hz/V], assumed flat at the
        modulation frequency.
    mod_vpp : float
        Peak-to-peak modulation voltage [V] applied to the actuator.
    mod_freq : float
        Nominal modulation frequency [Hz] (the tone whose amplitude encodes the
        OPD).  Used as the sole candidate when ``mod_freq_candidates`` is None.
    mod_freq_candidates : sequence of float | None
        Optional list of *candidate* nominal modulation frequencies.  When set,
        the pipeline auto-selects the candidate carrying the strongest coherent
        tone (e.g. ``(95.0, 100.0)`` to handle acquisitions taken with either
        modulation).  This makes a single configuration valid across a mixed
        data set without manual per-file tuning.

    Analysis
    --------
    channels : tuple[int, int] | None
        1-indexed phasemeter channels whose *difference* forms the differential
        interferometer phase ``phi = ch[0] - ch[1]``.  If ``None`` the package
        auto-selects ``(1, 2)`` when two channels are present, or the single
        available channel otherwise.
    refine_frequency : bool
        If True, refine the tone frequency around ``mod_freq`` from the data
        (robust to small clock offsets between the drive and the digitizer).
    freq_search_halfwidth : float
        Half-width [Hz] of the frequency-refinement search window.
    n_harmonics : int
        Number of modulation harmonics included as nuisance regressors in the
        least-squares lock-in (the fundamental carries the OPD; harmonics guard
        against drive/actuator distortion biasing the fundamental).
    detrend_order : int
        Polynomial order used to model slow phase drift (laser frequency noise)
        as a nuisance in the lock-in fit.
    n_stability_segments : int
        Number of contiguous segments used to assess estimate stationarity and
        provide an empirical (per-segment) uncertainty cross-check.
    noise_band_halfwidth : float
        Half-width [Hz] of the band around the tone, evaluated on the lock-in
        residual, used to estimate the local background noise PSD for the
        analytic (coherent) uncertainty.
    window : str
        speckit window used for the single-bin spectral cross-check.
    start_time : float | None
        Start time [s] of the analysis window (passed to the loader).
    duration : float | None
        Duration [s] to analyse (``None`` = whole file).
    """

    # --- Physics ---
    laser_wavelength: float = 1550e-9
    actuator_tf: float = 376e6
    mod_vpp: float = 1.0
    mod_freq: float = 100.0
    mod_freq_candidates: Optional[Sequence[float]] = None

    # --- Analysis: channel selection ---
    channels: Optional[Tuple[int, int]] = None

    # --- Analysis: tone estimation ---
    refine_frequency: bool = True
    freq_search_halfwidth: float = 0.5
    n_harmonics: int = 3
    detrend_order: int = 3

    # --- Analysis: uncertainty ---
    n_stability_segments: int = 10
    noise_band_halfwidth: float = 10.0

    # --- Analysis: spectral cross-check ---
    window: str = "kaiser"

    # --- Loading ---
    start_time: Optional[float] = None
    duration: Optional[float] = None

    @property
    def freq_dev_peak(self) -> float:
        """Peak (zero-to-peak) laser frequency deviation [Hz].

        ``delta_nu_pk = (dnu/dV) * Vpp/2`` since a peak-to-peak drive of
        ``Vpp`` corresponds to an amplitude of ``Vpp/2``.
        """
        return self.actuator_tf * self.mod_vpp / 2.0

    @property
    def laser_frequency(self) -> float:
        """Nominal optical carrier frequency [Hz] = c / wavelength."""
        return C_LIGHT / self.laser_wavelength

    @property
    def candidate_freqs(self) -> Tuple[float, ...]:
        """Tuple of candidate nominal modulation frequencies [Hz]."""
        if self.mod_freq_candidates:
            return tuple(float(f) for f in self.mod_freq_candidates)
        return (float(self.mod_freq),)
