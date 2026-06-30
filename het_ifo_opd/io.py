"""Loading Moku:Pro Phasemeter data into a tidy in-memory record.

Thin wrapper around :class:`mokutools.phasemeter.MokuPhasemeterObject` that
exposes exactly what the OPD estimator needs: the sampling frequency, the
per-channel phase in cycles, and the differential phase formed from the two
channels (with its DC offset removed).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from mokutools.phasemeter import MokuPhasemeterObject


@dataclass
class PhasemeterData:
    """Container for a loaded phasemeter acquisition.

    Attributes
    ----------
    path : str
        Source file path.
    name : str
        Basename without extension (used to label results/plots).
    fs : float
        Sampling frequency [Hz].
    t : np.ndarray
        Time vector [s], starting at 0.
    channel_cycles : list[np.ndarray]
        Per-channel phase [cycles], one array per phasemeter channel.
    nchan : int
        Number of channels.
    date : object
        Acquisition timestamp parsed from the header (may be ``None``).
    """

    path: str
    name: str
    fs: float
    t: np.ndarray
    channel_cycles: List[np.ndarray]
    date: object = None

    @property
    def nchan(self) -> int:
        return len(self.channel_cycles)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size > 1 else 0.0

    def differential(self, channels: Optional[Tuple[int, int]] = None):
        """Return ``(label, phi)`` for the chosen interferometric phase.

        With two channels the differential ``ch[a] - ch[b]`` is returned; this
        cancels the laser frequency noise common to both beams and isolates the
        residual (differential) OPD.  With a single channel that channel is
        returned directly.  The DC offset is always removed.
        """
        if channels is None:
            channels = (1, 2) if self.nchan >= 2 else (1,)

        if self.nchan == 1 or len(channels) == 1:
            a = channels[0]
            phi = self.channel_cycles[a - 1].astype(float)
            label = f"ch{a}"
        else:
            a, b = channels[0], channels[1]
            phi = (self.channel_cycles[a - 1] - self.channel_cycles[b - 1]).astype(float)
            label = f"ch{a}-ch{b}"

        phi = phi - np.mean(phi)
        return label, phi


def load_phasemeter(
    path: str,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> PhasemeterData:
    """Load a Moku:Pro Phasemeter ``.csv``/``.zip``/``.mat`` file.

    Parameters
    ----------
    path : str
        Path to the acquisition file.
    start_time, duration : float, optional
        Time window [s] to load (passed through to mokutools).
    logger : logging.Logger, optional

    Returns
    -------
    PhasemeterData
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    obj = MokuPhasemeterObject(
        path, start_time=start_time, duration=duration, logger=logger
    )

    fs = float(obj.fs)
    t = obj.df["time"].to_numpy(dtype=float)
    t = t - t[0]

    channel_cycles = [
        obj.df[f"{i + 1}_cycles"].to_numpy(dtype=float) for i in range(obj.nchan)
    ]

    name = os.path.splitext(os.path.basename(path))[0]
    return PhasemeterData(
        path=path,
        name=name,
        fs=fs,
        t=t,
        channel_cycles=channel_cycles,
        date=getattr(obj, "date", None),
    )
