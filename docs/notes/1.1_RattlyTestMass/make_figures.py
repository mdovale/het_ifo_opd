#!/usr/bin/env python
"""Figures and numeric macros for the rattly-test-mass note.

This is the reproducible back-end of ``main_RattlyTestMass.tex``.  It

  * quantifies the in-band amplification factor ``nu0 / delta_nu_pk``;
  * shows, on FM1 Day09 released acquisitions, that a rattly (impulsively
    moving) test mass floods the narrow demodulation band around ``fmod`` and
    inflates the standard incoherent OPD estimate;
  * validates, on synthetic data with a known OPD, that the motion-robust
    estimator (:func:`het_ifo_opd.estimators.demodulate_robust`) recovers the
    true OPD where the standard estimator is biased high;
  * runs the standard and the robust pipeline on every Day09 released file and
    tabulates the before/after numbers, the absolute near-tone contamination
    floor, and the reliability flag.

Run it from anywhere; paths are resolved relative to the repository root.

    python make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from het_ifo_opd import OPDConfig, estimate_opd, estimate_opd_robust, load_phasemeter  # noqa: E402
from het_ifo_opd.estimators import demodulate, demodulate_robust, refine_frequency  # noqa: E402
from het_ifo_opd.physics import opd_to_phase_cycles, phase_cycles_to_opd  # noqa: E402
from speckit import compute_spectrum  # noqa: E402

DATA = REPO / "data" / "FM1"
FIGS = HERE / "Figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
})

cfg = OPDConfig()
FMOD = 95.0
DNU = cfg.freq_dev_peak
NU0 = cfg.laser_frequency
macros: dict[str, str] = {}


def m(name: str, value: str) -> None:
    macros[name] = value


def num(x: float, fmt: str = "{:.3g}") -> str:
    return fmt.format(x)


def to_opd_mm(amp_cyc: float) -> float:
    return phase_cycles_to_opd(amp_cyc, DNU) * 1e3


# Day09 released acquisitions (all of them), grouped by system.
FILES = {
    "R1": [
        ("NOSHIMS", "FM1Day09_AirReleasedOPD_NOSHIMS_R1_20260629_113414.zip"),
        ("NOSHIMS+torqued",
         "FM1Day09_AirReleasedOPD_NOSHIMS_FLIGHTTORQUED_R1_20260629_120744.zip"),
        ("shimmed+torqued",
         "FM1Day09_AirReleasedOPD_SHIMMED_FLIGHTTORQUED_R1_20260629_130407.zip"),
        ("shimmed+torqued (vac)",
         "FM1Day09_VacuumReleasedOPD_SHIMMED_FLIGHTTORQUED_R1_20260629_140109.zip"),
    ],
    "R2": [
        ("NOSHIMS", "FM1Day09_AirReleasedOPD_NOSHIMS_R2_20260629_113415.zip"),
        ("NOSHIMS+torqued",
         "FM1Day09_AirReleasedOPD_NOSHIMS_FLIGHTTORQUED_R2_20260629_120742.zip"),
        ("shimmed+torqued",
         "FM1Day09_AirReleasedOPD_SHIMMED_FLIGHTTORQUED_R2_20260629_130406.zip"),
        ("shimmed+torqued (vac)",
         "FM1Day09_VacuumReleasedOPD_SHIMMED_FLIGHTTORQUED_R2_20260629_140110.zip"),
    ],
}
# The matched clean/contaminated pair used for the mechanism figures.
CLEAN = "FM1Day09_AirReleasedOPD_SHIMMED_FLIGHTTORQUED_R1_20260629_130407.zip"
RATTLY = "FM1Day09_AirReleasedOPD_NOSHIMS_FLIGHTTORQUED_R2_20260629_120742.zip"
OUTLIER = "FM1Day09_AirReleasedOPD_SHIMMED_FLIGHTTORQUED_R2_20260629_130406.zip"

# ---------------------------------------------------------------------------
# 0) Physics constants: the in-band amplification factor.
# ---------------------------------------------------------------------------
ratio = NU0 / DNU
m("nuzero", num(NU0, "{:.3e}"))
m("dnupkval", num(DNU / 1e6, "{:.0f}"))
m("gainratio", num(ratio, "{:.2e}"))
m("gainratioround", num(ratio / 1e6, "{:.1f}"))
m("lambdanm", num(cfg.laser_wavelength * 1e9, "{:.0f}"))
m("fmod", num(FMOD, "{:.0f}"))
m("demodbw", num(cfg.demod_bandwidth, "{:.1f}"))
# A 2 nm OPD motion component at fmod reads as this apparent OPD in the tone bin.
m("nmexample", num(2.0 * ratio / 1e6, "{:.1f}"))  # mm-equivalent of 2 nm x ratio

# ---------------------------------------------------------------------------
# 1) Raw differential phase: clean ring-down (R1) vs rattly (R2).
# ---------------------------------------------------------------------------
print("Figure: ring-down time series ...")


def load_diff(fname):
    d = load_phasemeter(str(DATA / fname))
    _, phi = d.differential(cfg.channels)
    return d, phi


dclean, phiclean = load_diff(CLEAN)
drat, phirat = load_diff(RATTLY)

fig, ax = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True)
ax[0].plot(dclean.t, phiclean - phiclean.mean(), lw=0.4, color="C0")
ax[0].set(ylabel="diff. phase [cyc]",
          title="Clean ring-down (R1, shimmed+torqued): smooth pendulum swing")
ax[1].plot(drat.t, phirat - phirat.mean(), lw=0.4, color="C3")
ax[1].set(xlabel="Time [s]", ylabel="diff. phase [cyc]",
          title="Rattly motion (R2, NOSHIMS+torqued): impulsive re-excitations")
fig.tight_layout()
fig.savefig(FIGS / "fig_ringdown.pdf")
plt.close(fig)

m("cleanphistd", num(np.std(phiclean), "{:.2f}"))
m("cleanptp", num(np.ptp(phiclean), "{:.1f}"))
m("rattlyphistd", num(np.std(phirat), "{:.2f}"))
m("rattlyptp", num(np.ptp(phirat), "{:.1f}"))

# ---------------------------------------------------------------------------
# 2) ASD near the tone: the near-tone floor is what matters.
# ---------------------------------------------------------------------------
print("Figure: ASD near the tone ...")
sclean = compute_spectrum(phiclean - phiclean.mean(), dclean.fs, win="kaiser",
                          olap="default")
srat = compute_spectrum(phirat - phirat.mean(), drat.fs, win="kaiser",
                        olap="default")


def band_floor(phi, fs, f0, lo, hi):
    """RMS ASD [cyc/sqrt(Hz)] in [f0-hi,f0-lo] U [f0+lo,f0+hi] via a linear FFT."""
    x = phi - np.mean(phi)
    nfft = x.size
    win = np.hanning(nfft)
    xw = x * win
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    X = np.fft.rfft(xw)
    enbw = fs * np.sum(win ** 2) / np.sum(win) ** 2
    psd = (np.abs(X) ** 2) / (np.sum(win) ** 2 * enbw) * 2.0
    asd = np.sqrt(psd)
    msk = (((f >= f0 - hi) & (f <= f0 - lo)) |
           ((f >= f0 + lo) & (f <= f0 + hi)))
    return float(np.sqrt(np.mean(asd[msk] ** 2)))


fc = band_floor(phiclean, dclean.fs, FMOD, 0.6, 1.5)
fr = band_floor(phirat, drat.fs, FMOD, 0.6, 1.5)
m("floorasdclean", num(fc, "{:.2e}"))
m("floorasdrattly", num(fr, "{:.2e}"))
m("floorasdratio", num(fr / fc, "{:.0f}"))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
ax[0].loglog(sclean.f, sclean.asd, lw=0.7, color="C0", label="clean (R1)")
ax[0].loglog(srat.f, srat.asd, lw=0.7, color="C3", alpha=0.8, label="rattly (R2)")
ax[0].axvline(FMOD, color="0.4", ls="--", lw=1, label=f"{FMOD:.0f} Hz tone")
ax[0].set(xlabel="Frequency [Hz]", ylabel=r"ASD [cyc/$\sqrt{\mathrm{Hz}}$]",
          title="Full differential-phase spectra")
ax[0].legend()
sel_c = (sclean.f > FMOD - 8) & (sclean.f < FMOD + 8)
sel_r = (srat.f > FMOD - 8) & (srat.f < FMOD + 8)
ax[1].semilogy(sclean.f[sel_c], sclean.asd[sel_c], lw=0.9, color="C0",
               label="clean (R1)")
ax[1].semilogy(srat.f[sel_r], srat.asd[sel_r], lw=0.9, color="C3", alpha=0.85,
               label="rattly (R2)")
ax[1].axvspan(FMOD - cfg.demod_bandwidth, FMOD + cfg.demod_bandwidth,
              color="0.6", alpha=0.25, label=r"$\pm$" + f"{cfg.demod_bandwidth} Hz band")
ax[1].axvline(FMOD, color="0.4", ls="--", lw=1)
ax[1].set(xlabel="Frequency [Hz]", ylabel=r"ASD [cyc/$\sqrt{\mathrm{Hz}}$]",
          title=f"Zoom on the demodulation band: R2 floor is "
                f"{fr / fc:.0f}$\\times$ higher")
ax[1].legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_asd.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3) Mechanism: on the rattly record, |z(t)| inflates the incoherent estimate
#    above the coherent baseline; the near-tone floor sets the noise level.
# ---------------------------------------------------------------------------
print("Figure: contamination |z(t)| decomposition ...")
d_c, phi_c = load_diff(RATTLY)
f0 = refine_frequency(phi_c, d_c.fs, FMOD)
dm_c = demodulate(phi_c, d_c.fs, f0)
r_c = demodulate_robust(phi_c, d_c.fs, f0)
opd_t = phase_cycles_to_opd(np.abs(dm_c.z), DNU) * 1e3

fig, ax = plt.subplots(figsize=(8.6, 3.8))
ax.plot(dm_c.t, opd_t, lw=0.5, color="0.6", label=r"$|z(t)|$ (instantaneous)")
ax.axhline(to_opd_mm(dm_c.amplitude), color="C3", lw=1.4,
           label=f"standard incoherent = {to_opd_mm(dm_c.amplitude):.3f} mm")
ax.axhline(to_opd_mm(r_c.amp_robust), color="C0", lw=1.4,
           label=f"robust coherent proj. = {to_opd_mm(r_c.amp_robust):.3f} mm")
ax.axhline(to_opd_mm(r_c.near_tone_floor), color="C2", lw=1.2, ls="--",
           label=f"near-tone floor = {to_opd_mm(r_c.near_tone_floor):.3f} mm")
ax.set(xlabel="Time [s]", ylabel="OPD [mm]",
       title="Rattly record (R2, NOSHIMS+torqued): the incoherent RMS sums "
             "in-band motion power as if it were signal")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGS / "fig_contam.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Synthetic validation with a KNOWN OPD.
# ---------------------------------------------------------------------------
print("Synthetic validation ...")
FS = 596.04644775
rng = np.random.default_rng(11)
OPD0 = 0.30e-3
A0 = opd_to_phase_cycles(OPD0, DNU)
Tsec = 300.0
n = int(Tsec * FS)
tt = np.arange(n) / FS


def lf_motion(t, rms_cyc=4.0):
    comps = [(1, 0.11, 0.4), (0.6, 0.4, 1.1), (0.3, 1.2, 2.0),
             (0.2, 3.0, 0.7), (0.1, 7.7, 1.3)]
    x = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in comps)
    return x / np.std(x) * rms_cyc


def inband(t, fs, f0, level_cyc, bursty):
    nn = t.size
    c = rng.standard_normal(nn) + 1j * rng.standard_normal(nn)
    b, a = butter(2, 3.0 / (fs / 2))
    c = filtfilt(b, a, c)
    if bursty:
        env = np.zeros(nn)
        for c0 in np.linspace(20, t[-1] - 20, 4):
            env += np.exp(-0.5 * ((t - c0) / 7.0) ** 2)
        env = np.clip(env, 0, 1)
    else:
        env = np.ones(nn)
    c = c * env
    c = c / np.sqrt(np.mean(np.abs(c) ** 2)) * level_cyc
    return (c * np.exp(1j * 2 * np.pi * f0 * t)).real


levels = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
lf = lf_motion(tt)
results_syn = {"bursty": {"std": [], "rob": []}, "steady": {"std": [], "rob": []}}
for bursty, key in [(True, "bursty"), (False, "steady")]:
    for lv in levels:
        phi = (A0 * np.cos(2 * np.pi * FMOD * tt + 0.7) + lf
               + (inband(tt, FS, FMOD, lv * A0, bursty) if lv > 0 else 0.0)
               + 1e-4 * rng.standard_normal(n))
        dm = demodulate(phi, FS, FMOD)
        rb = demodulate_robust(phi, FS, FMOD)
        results_syn[key]["std"].append(to_opd_mm(dm.amplitude))
        results_syn[key]["rob"].append(to_opd_mm(rb.amp_robust))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, key, ttl in zip(axes, ["bursty", "steady"],
                        ["Bursty in-band contamination",
                         "Steady in-band contamination"]):
    std = np.array(results_syn[key]["std"])
    rob = np.array(results_syn[key]["rob"])
    ax.axhline(OPD0 * 1e3, color="0.3", ls="--", lw=1.2, label="true OPD (0.300 mm)")
    ax.plot(levels, std, "o-", color="C3", label="standard estimator")
    ax.plot(levels, rob, "s-", color="C0", label="robust estimator")
    ax.set(xlabel=r"in-band contamination level [$\times$ tone amplitude]",
           title=ttl)
    ax.legend()
axes[0].set_ylabel("recovered OPD [mm]")
fig.tight_layout()
fig.savefig(FIGS / "fig_synth.pdf")
plt.close(fig)

# Headline synthetic numbers (worst case, 4x).
i4 = int(np.where(levels == 4.0)[0][0])
m("synopdtrue", num(OPD0 * 1e3, "{:.3f}"))
m("synstdburst", num(results_syn["bursty"]["std"][i4], "{:.3f}"))
m("synrobburst", num(results_syn["bursty"]["rob"][i4], "{:.3f}"))
m("synstdsteady", num(results_syn["steady"]["std"][i4], "{:.3f}"))
m("synrobsteady", num(results_syn["steady"]["rob"][i4], "{:.3f}"))
rob_all = np.array(results_syn["bursty"]["rob"] + results_syn["steady"]["rob"])
m("synrobmaxerr", num(100 * np.max(np.abs(rob_all - OPD0 * 1e3) / (OPD0 * 1e3)),
                      "{:.0f}"))
std_all = np.array(results_syn["bursty"]["std"] + results_syn["steady"]["std"])
m("synstdmaxfac", num(np.max(std_all) / (OPD0 * 1e3), "{:.1f}"))

# ---------------------------------------------------------------------------
# 5) Day09 pre/post table and bar chart.
# ---------------------------------------------------------------------------
print("Day09 pre/post pipeline ...")
RELTHRESH = 3.0
m("relthresh", num(RELTHRESH, "{:.0f}"))
rows = []
for sysname, entries in FILES.items():
    for cfgname, fname in entries:
        rr = estimate_opd_robust(str(DATA / fname), config=cfg, mod_freq=FMOD)
        rows.append((sysname, cfgname, rr))

# floor statistics per system.
_word = {"R1": "Rone", "R2": "Rtwo"}
for sysname in ("R1", "R2"):
    floors = [rr.near_tone_floor * 1e6 for s, c, rr in rows if s == sysname]
    m(f"floor{_word[sysname]}min", num(min(floors), "{:.0f}"))
    m(f"floor{_word[sysname]}max", num(max(floors), "{:.0f}"))

# Named-file macros for the prose (clean / rattly / outlier).
name_map = {CLEAN: "clean", RATTLY: "rattly", OUTLIER: "outlier"}
for s, c, rr in rows:
    base = Path(rr.path).name
    if base in name_map:
        tag = name_map[base]
        m(f"{tag}std", num(rr.opd_standard * 1e3, "{:.3f}"))
        m(f"{tag}rob", num(rr.opd_robust * 1e3, "{:.3f}"))
        m(f"{tag}floor", num(rr.near_tone_floor * 1e6, "{:.0f}"))
        m(f"{tag}relia", num(rr.reliability, "{:.1f}"))
        m(f"{tag}mot", num(rr.motion_rms * 1e6, "{:.0f}"))
        m(f"{tag}mode", rr.standard_mode)

# Build the full LaTeX tabular (input as a block, not inside another tabular).
tbl = [
    r"\begin{tabular}{llrlrrrr}",
    r"\toprule",
    r"Sys & Config & $\mathrm{OPD}_{\text{std}}$ & mode & "
    r"$\mathrm{OPD}_{\text{robust}}$ & floor & $\rho_{\text{rel}}$ & motion \\",
    r" & & [mm] & & [mm] & [\si{\micro m}] & & [\si{\micro m}] \\",
    r"\midrule",
]
for s, c, rr in rows:
    flag = "" if rr.reliability >= RELTHRESH else r"$^{\dagger}$"
    tbl.append(
        f"{s} & {c} & {rr.opd_standard*1e3:.3f} & {rr.standard_mode} & "
        f"{rr.opd_robust*1e3:.3f}{flag} & {rr.near_tone_floor*1e6:.0f} & "
        f"{rr.reliability:.1f} & {rr.motion_rms*1e6:.0f} \\\\"
    )
tbl += [r"\bottomrule", r"\end{tabular}"]
(HERE / "table_day09.tex").write_text("\n".join(tbl) + "\n")

# Bar chart pre/post grouped by system.
fig, ax = plt.subplots(figsize=(9.5, 4.4))
x = np.arange(len(rows))
w = 0.38
std = [rr.opd_standard * 1e3 for _, _, rr in rows]
rob = [rr.opd_robust * 1e3 for _, _, rr in rows]
ax.bar(x - w / 2, std, w, color="C3", alpha=0.85, label="standard")
ax.bar(x + w / 2, rob, w, color="C0", alpha=0.9, label="robust")
for i, (_, _, rr) in enumerate(rows):
    if rr.reliability < RELTHRESH:
        ax.annotate(r"$\dagger$", (i + w / 2, rob[i]), textcoords="offset points",
                    xytext=(0, 2), ha="center", fontsize=11, color="0.2")
labels = [f"{s}\n{c}" for s, c, _ in rows]
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
ax.set_ylabel("OPD [mm]")
ax.set_title(r"Day09 released: standard vs robust OPD "
             r"($\dagger$ = flagged low-reliability)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_prepost.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6) The bonus: test-mass motion read from the low-frequency phase.
# ---------------------------------------------------------------------------
print("Figure: motion readout ...")
rr_rat = estimate_opd_robust(str(DATA / RATTLY), config=cfg, mod_freq=FMOD)
fig, ax = plt.subplots(figsize=(8.6, 3.4))
ax.plot(rr_rat.motion_t, rr_rat.motion * 1e6, lw=0.4, color="C4")
ax.set(xlabel="Time [s]", ylabel=r"$\delta$OPD [$\mu$m]",
       title=f"Test-mass motion from the low-frequency differential phase "
             f"(band {rr_rat.motion_band[0]:.1f}-{rr_rat.motion_band[1]:.0f} Hz), "
             f"RMS {rr_rat.motion_rms*1e6:.0f} um")
fig.tight_layout()
fig.savefig(FIGS / "fig_motion.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Write values.tex
# ---------------------------------------------------------------------------
print("Writing values.tex ...")
lines = ["% Auto-generated by make_figures.py -- do not edit by hand.",
         "% Regenerate with:  make figs   (or  python make_figures.py)"]
for k in sorted(macros):
    lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
(HERE / "values.tex").write_text("\n".join(lines) + "\n")
print("Done. Figures in", FIGS)
