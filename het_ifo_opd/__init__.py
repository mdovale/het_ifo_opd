"""het_ifo_opd -- optimal estimation of interferometer OPD from frequency-modulated
Moku:Pro Phasemeter data.

A known single-tone modulation of the laser frequency imprints a phase tone on
the differential interferometer phase whose amplitude is proportional to the
residual optical path-length difference (OPD).  This package ingests Moku:Pro
Phasemeter files, estimates that tone amplitude with an optimal least-squares
lock-in (cross-checked spectrally), and converts it into an OPD with a rigorous
uncertainty.

Quick start
-----------
>>> from het_ifo_opd import estimate_opd
>>> r = estimate_opd("FM1/some_acquisition.zip")
>>> print(r.opd * 1e3, "mm +/-", r.opd_unc * 1e6, "um")
"""
from __future__ import annotations

from .config import C_LIGHT, OPDConfig
from .io import PhasemeterData, load_phasemeter
from .physics import (
    opd_to_carrier_cycles,
    opd_to_delay,
    opd_to_phase_cycles,
    phase_cycles_to_opd,
)
from .estimators import (
    LockinResult,
    detect_tone_frequency,
    lockin_amplitude,
    local_noise_psd,
    refine_frequency,
    segmented_amplitudes,
    single_bin_amplitude,
)
from .pipeline import (
    DatasetResult,
    OPDResult,
    estimate_opd,
    estimate_opd_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "C_LIGHT",
    "OPDConfig",
    "PhasemeterData",
    "load_phasemeter",
    "phase_cycles_to_opd",
    "opd_to_phase_cycles",
    "opd_to_delay",
    "opd_to_carrier_cycles",
    "LockinResult",
    "lockin_amplitude",
    "single_bin_amplitude",
    "refine_frequency",
    "detect_tone_frequency",
    "local_noise_psd",
    "segmented_amplitudes",
    "OPDResult",
    "DatasetResult",
    "estimate_opd",
    "estimate_opd_dataset",
]
