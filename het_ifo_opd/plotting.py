"""Diagnostic plots for OPD estimates."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import OPDConfig
from .estimators import _design_matrix
from .io import PhasemeterData, load_phasemeter
from .physics import phase_cycles_to_opd
from .pipeline import OPDResult


def plot_diagnostics(
    result: OPDResult,
    config: Optional[OPDConfig] = None,
    data: Optional[PhasemeterData] = None,
    savepath: Optional[str] = None,
):
    """Three-panel diagnostic figure for one acquisition.

    (1) Welch ASD of the differential phase (linear frequency axis) with the
        modulation tone marked.
    (2) A few cycles of the differential phase with the lock-in fit overlaid.
    (3) Per-segment OPD stability with the mean and 1-sigma band.

    Requires the source data; if ``data`` is not supplied it is reloaded from
    ``result.path``.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    if config is None:
        config = OPDConfig()
    if data is None:
        data = load_phasemeter(
            result.path, start_time=config.start_time, duration=config.duration
        )

    label, phi = data.differential(config.channels)
    fs = data.fs
    f0 = result.tone_freq

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # (1) ASD via scipy.welch (linear frequency axis, log ASD).
    x = (phi - np.mean(phi)) * (2 * np.pi)  # cycles -> radians
    nperseg = min(int(60.0 * fs), x.size)
    noverlap = nperseg // 2
    f, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    asd = np.sqrt(psd)
    ax = axes[0]
    ax.semilogy(f, asd, lw=0.8)
    ax.axvline(f0, color="C3", ls="--", lw=1, label=f"{f0:.3f} Hz")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"ASD [rad/$\sqrt{\mathrm{Hz}}$]")
    ax.set_title(f"{result.name}\nDifferential phase ({label})", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # (2) phase-folded (synchronous-averaged) tone.
    # The fundamental is far below the broadband differential phase noise in any
    # raw snippet; folding at the tone period and averaging over the tens of
    # thousands of cycles in the record suppresses the out-of-band noise by
    # ~sqrt(N_cycles) and exposes the tone at any realistic SNR.
    ax = axes[1]
    t = np.arange(phi.size) / fs
    A = _design_matrix(t, f0, config.n_harmonics, config.detrend_order)
    coef, *_ = np.linalg.lstsq(A, phi, rcond=None)

    # Band-pass around the tone so the low-frequency differential-phase wander
    # does not dominate the per-bin variance; then fold and average.
    from scipy.signal import butter, filtfilt
    bw = max(15.0, 0.2 * f0)
    lo = max(f0 - bw, 0.5) / (fs / 2)
    hi = min(f0 + bw, fs / 2 * 0.98) / (fs / 2)
    bcoef, acoef = butter(4, [lo, hi], btype="band")
    detr = filtfilt(bcoef, acoef, phi - np.mean(phi))

    n_bins = 48
    ph = np.mod(f0 * t, 1.0)
    idx = np.minimum((ph * n_bins).astype(int), n_bins - 1)
    folded = np.array([detr[idx == k].mean() if np.any(idx == k) else np.nan
                       for k in range(n_bins)])
    folded_err = np.array([detr[idx == k].std() / max(np.sqrt(np.sum(idx == k)), 1)
                           if np.any(idx == k) else np.nan for k in range(n_bins)])
    centers = (np.arange(n_bins) + 0.5) / n_bins
    a1, b1 = coef[0], coef[1]
    model = a1 * np.cos(2 * np.pi * centers) + b1 * np.sin(2 * np.pi * centers)

    ax.errorbar(centers, folded * 1e3, yerr=folded_err * 1e3, fmt=".", ms=4,
                alpha=0.7, label="folded data")
    ax.plot(centers, model * 1e3, "C3", lw=1.5, label="lock-in fundamental")
    ax.set_xlabel(f"Phase of {f0:.3f} Hz tone [cycles]")
    ax.set_ylabel("Phase [mcyc]")
    ax.set_title(f"Folded tone: A = {result.amp_cycles:.3e} cyc "
                 f"(SNR {result.tone_snr:.3g})", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (3) stability
    ax = axes[2]
    seg = result.segment_opds * 1e3  # mm
    ax.plot(np.arange(seg.size) + 1, seg, "o-", ms=4)
    ax.axhline(result.opd * 1e3, color="C3", lw=1.2, label="full-record")
    band = result.opd_unc * 1e3
    ax.axhspan((result.opd - result.opd_unc) * 1e3,
               (result.opd + result.opd_unc) * 1e3, color="C3", alpha=0.15)
    ax.set_xlabel("Segment")
    ax.set_ylabel("OPD [mm]")
    ax.set_title(
        f"OPD = {result.opd * 1e3:.4f} ± {result.opd_unc * 1e6:.2f} µm",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


def plot_dataset_summary(dataset, savepath: Optional[str] = None):
    """Bar chart comparing OPD across all acquisitions in a dataset."""
    import matplotlib.pyplot as plt

    results = list(dataset)
    names = [r.name for r in results]
    opds = np.array([r.opd * 1e3 for r in results])
    uncs = np.array([r.opd_unc * 1e6 for r in results]) / 1e3  # to mm

    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, opds, xerr=uncs, color="C0", alpha=0.8, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Residual OPD [mm]")
    ax.set_title("Residual differential OPD across acquisitions")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig
