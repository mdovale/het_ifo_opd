"""Physics of frequency-modulation OPD readout.

A two-beam interferometer with optical path-length difference ``OPD`` converts
the instantaneous laser (optical) frequency ``nu`` into an interferometric
phase

    phi(t) = nu(t) * OPD / c            [cycles]

so a small laser-frequency modulation ``delta_nu(t)`` produces a phase
modulation

    delta_phi(t) = (OPD / c) * delta_nu(t).

Driving the laser-frequency actuator with a single tone at ``f_mod`` therefore
imprints a tone of amplitude ``A_phi = (OPD/c) * delta_nu_pk`` on the phase.
Inverting this relation turns a measured phase-tone amplitude directly into an
OPD -- *independently of the laser wavelength* and without any need to know the
absolute (DC) interferometer phase.

The relation is linear in ``delta_nu``, so every Fourier component of the drive
maps to the same Fourier component of the phase; only the *fundamental* tone is
used for the OPD, while harmonics (from drive/actuator distortion) are
nuisances.
"""
from __future__ import annotations

import numpy as np

from .config import C_LIGHT


def phase_cycles_to_opd(amp_cycles, freq_dev_peak):
    """Convert a phase-tone amplitude into an OPD.

    Parameters
    ----------
    amp_cycles : float or array
        Zero-to-peak amplitude of the phase tone [cycles].
    freq_dev_peak : float
        Zero-to-peak laser frequency deviation [Hz] (see
        :pyattr:`OPDConfig.freq_dev_peak`).

    Returns
    -------
    opd : float or array
        Optical path-length difference [m].
    """
    return np.asarray(amp_cycles) * C_LIGHT / freq_dev_peak


def opd_to_phase_cycles(opd, freq_dev_peak):
    """Inverse of :func:`phase_cycles_to_opd` (mainly for testing)."""
    return np.asarray(opd) * freq_dev_peak / C_LIGHT


def opd_to_delay(opd):
    """Convert an OPD [m] to the equivalent differential time delay [s]."""
    return np.asarray(opd) / C_LIGHT


def opd_to_carrier_cycles(opd, wavelength):
    """Equivalent number of carrier cycles in ``opd`` (context/reporting only)."""
    return np.asarray(opd) / wavelength
