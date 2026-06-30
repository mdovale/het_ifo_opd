"""High-level OPD-estimation pipeline.

``estimate_opd`` turns one phasemeter file into an :class:`OPDResult` carrying
the residual differential OPD, its uncertainty, and a full set of diagnostics.
``estimate_opd_dataset`` runs the pipeline over many files and returns a tidy
summary table.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import OPDConfig
from .estimators import (
    LockinResult,
    detect_tone_frequency,
    lockin_amplitude,
    segmented_amplitudes,
    single_bin_amplitude,
)
from .io import PhasemeterData, load_phasemeter
from .physics import opd_to_delay, phase_cycles_to_opd


@dataclass
class OPDResult:
    """Outcome of an OPD estimation for a single acquisition."""

    name: str
    path: str

    # Headline result (differential / residual OPD).
    opd: float                       # [m]
    opd_unc: float                   # [m] 1-sigma (coherent)
    opd_unc_empirical: float         # [m] 1-sigma (per-segment scatter of mean)

    # Underlying tone amplitude estimate.
    observable: str                  # e.g. "ch1-ch2"
    tone_freq: float                 # [Hz] refined modulation tone frequency
    tone_freq_nominal: float         # [Hz] selected nominal (candidate) frequency
    tone_snr: float                  # peak-power / median-power significance
    amp_cycles: float                # fundamental amplitude [cycles]
    amp_unc_cycles: float            # [cycles] coherent uncertainty

    # Cross-check & quality diagnostics.
    amp_cycles_spectral: float       # speckit single-bin amplitude [cycles]
    method_agreement: float          # |lockin-spectral|/lockin (relative)
    delay: float                     # equivalent differential delay [s]
    residual_std: float              # lock-in residual std [cycles]
    noise_psd: float                 # local noise PSD at f0 [cyc^2/Hz]
    harmonic_ratio: float            # 2nd-harmonic / fundamental amplitude
    duration: float                  # [s]
    fs: float                        # [Hz]

    # Stationarity diagnostics.
    segment_opds: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-channel (absolute) OPDs, for reference.
    channel_opds: Dict[str, float] = field(default_factory=dict)

    def summary_row(self) -> Dict[str, object]:
        """Flat dict suitable for a results DataFrame."""
        return {
            "name": self.name,
            "observable": self.observable,
            "OPD_mm": self.opd * 1e3,
            "OPD_unc_um": self.opd_unc * 1e6,
            "OPD_unc_emp_um": self.opd_unc_empirical * 1e6,
            "f_nom_Hz": self.tone_freq_nominal,
            "tone_freq_Hz": self.tone_freq,
            "tone_snr": self.tone_snr,
            "amp_cyc": self.amp_cycles,
            "amp_spec_cyc": self.amp_cycles_spectral,
            "method_agreement": self.method_agreement,
            "delay_ns": self.delay * 1e9,
            "harmonic_ratio": self.harmonic_ratio,
            "duration_s": self.duration,
        }


def estimate_opd(
    source: Union[str, PhasemeterData],
    config: Optional[OPDConfig] = None,
    logger: Optional[logging.Logger] = None,
    mod_freq: Optional[float] = None,
) -> OPDResult:
    """Estimate the residual differential OPD from one acquisition.

    Parameters
    ----------
    source : str or PhasemeterData
        File path or an already-loaded :class:`PhasemeterData`.
    config : OPDConfig, optional
        Physics/analysis configuration (defaults match the FM1 set-up).
    logger : logging.Logger, optional
    mod_freq : float, optional
        Known modulation frequency [Hz] for *this* acquisition.  When given it
        overrides ``config.mod_freq``/``mod_freq_candidates`` (recommended when
        the drive frequency is known, since candidate auto-selection is
        unreliable for weak tones).
    """
    if config is None:
        config = OPDConfig()
    if logger is None:
        logger = logging.getLogger(__name__)

    if isinstance(source, PhasemeterData):
        data = source
    else:
        data = load_phasemeter(
            source,
            start_time=config.start_time,
            duration=config.duration,
            logger=logger,
        )

    label, phi = data.differential(config.channels)
    fs = data.fs

    # 1) Tone frequency: pick the strongest candidate; the detector also returns
    #    the sub-bin-refined peak frequency within that candidate's window.
    candidates = (float(mod_freq),) if mod_freq is not None else config.candidate_freqs
    f_detected, snr, f_nominal = detect_tone_frequency(
        phi, fs, candidates, halfwidth=config.freq_search_halfwidth
    )
    f0 = f_detected if config.refine_frequency else f_nominal

    # 2) Primary estimator: least-squares lock-in.
    lk: LockinResult = lockin_amplitude(
        phi, fs, f0,
        n_harmonics=config.n_harmonics,
        detrend_order=config.detrend_order,
        noise_band_halfwidth=config.noise_band_halfwidth,
        window=config.window,
    )

    # 3) Independent spectral cross-check.
    amp_spec = single_bin_amplitude(phi, fs, f0, window=config.window)
    agreement = (
        abs(lk.amplitude - amp_spec) / lk.amplitude if lk.amplitude > 0 else np.nan
    )

    # 4) Stationarity / empirical uncertainty.
    seg_amps = segmented_amplitudes(
        phi, fs, f0,
        n_segments=config.n_stability_segments,
        n_harmonics=config.n_harmonics,
        detrend_order=config.detrend_order,
    )
    seg_opds = phase_cycles_to_opd(seg_amps, config.freq_dev_peak)
    if seg_opds.size > 1:
        opd_unc_emp = float(np.std(seg_opds, ddof=1) / np.sqrt(seg_opds.size))
    else:
        opd_unc_emp = float("nan")

    # 5) Convert to OPD.
    dnu_pk = config.freq_dev_peak
    opd = float(phase_cycles_to_opd(lk.amplitude, dnu_pk))
    opd_unc = float(phase_cycles_to_opd(lk.amp_unc_coherent, dnu_pk))

    harmonic_ratio = (
        float(lk.harmonic_amps[1] / lk.harmonic_amps[0])
        if lk.harmonic_amps.size > 1 and lk.harmonic_amps[0] > 0
        else float("nan")
    )

    # Per-channel absolute OPDs (diagnostic).
    channel_opds: Dict[str, float] = {}
    for ch in range(1, data.nchan + 1):
        _, phi_ch = data.differential((ch,))
        amp_ch = single_bin_amplitude(phi_ch, fs, f0, window=config.window)
        channel_opds[f"ch{ch}"] = float(phase_cycles_to_opd(amp_ch, dnu_pk))

    return OPDResult(
        name=data.name,
        path=data.path,
        opd=opd,
        opd_unc=opd_unc,
        opd_unc_empirical=opd_unc_emp,
        observable=label,
        tone_freq=f0,
        tone_freq_nominal=f_nominal,
        tone_snr=float(snr),
        amp_cycles=lk.amplitude,
        amp_unc_cycles=lk.amp_unc_coherent,
        amp_cycles_spectral=amp_spec,
        method_agreement=float(agreement),
        delay=float(opd_to_delay(opd)),
        residual_std=lk.residual_std,
        noise_psd=lk.noise_psd,
        harmonic_ratio=harmonic_ratio,
        duration=lk.duration,
        fs=fs,
        segment_opds=seg_opds,
        channel_opds=channel_opds,
    )


def estimate_opd_dataset(
    paths: Union[str, Sequence[str]],
    config: Optional[OPDConfig] = None,
    logger: Optional[logging.Logger] = None,
    freq_resolver: Optional["Callable[[str], Optional[float]]"] = None,
) -> "DatasetResult":
    """Run :func:`estimate_opd` over many files.

    Parameters
    ----------
    paths : str or sequence of str
        A glob pattern, a directory, or an explicit list of file paths.
    freq_resolver : callable, optional
        ``path -> mod_freq`` (or ``None``).  Use it to assign the known
        modulation frequency per file (e.g. 95 Hz vs 100 Hz by acquisition day).
        When it returns ``None`` the configured candidates/auto-detection apply.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    file_list = _resolve_paths(paths)
    if not file_list:
        raise FileNotFoundError(f"No phasemeter files matched: {paths!r}")

    results: List[OPDResult] = []
    for p in file_list:
        logger.info("Processing %s", os.path.basename(p))
        mf = freq_resolver(p) if freq_resolver is not None else None
        results.append(estimate_opd(p, config=config, logger=logger, mod_freq=mf))

    return DatasetResult(results=results, config=config or OPDConfig())


def _resolve_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    exts = (".zip", ".csv", ".mat")
    items = [paths] if isinstance(paths, str) else list(paths)

    files: List[str] = []
    for item in items:
        if os.path.isdir(item):
            files.extend(
                os.path.join(item, f)
                for f in os.listdir(item)
                if f.lower().endswith(exts)
            )
        elif any(ch in item for ch in "*?[") or not os.path.exists(item):
            files.extend(f for f in glob.glob(item) if f.lower().endswith(exts))
        else:
            files.append(item)

    # De-duplicate acquisitions that exist in several formats (e.g. a .zip and an
    # extracted .csv of the same data): keep one per basename stem, preferring
    # the compressed archive.
    ext_priority = {".zip": 0, ".mat": 1, ".csv": 2}
    chosen: dict = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        ext = os.path.splitext(f)[1].lower()
        if stem not in chosen or ext_priority.get(ext, 9) < ext_priority.get(
            os.path.splitext(chosen[stem])[1].lower(), 9
        ):
            chosen[stem] = f
    return sorted(chosen.values())


@dataclass
class DatasetResult:
    """Collection of :class:`OPDResult` objects with a summary table."""

    results: List[OPDResult]
    config: OPDConfig

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.summary_row() for r in self.results])

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i):
        return self.results[i]
