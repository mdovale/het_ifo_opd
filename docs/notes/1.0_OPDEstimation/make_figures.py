#!/usr/bin/env python
"""Generate all figures and numeric macros for the OPD-estimation note.

This script is the reproducible back-end of ``main_OPDEstimation.tex``.  It runs
the ``het_ifo_opd`` pipeline on three deliberately chosen FM1 acquisitions,

    * Case 1  (R1, anchored, air): clean and well behaved, the reference;
    * Case 2  (R1, released, vacuum): coherent but strongly leaky;
    * Case 3  (R2, released, vacuum): phase-incoherent, drift limited,

and writes

    Figs/*.pdf        -- figures included by the LaTeX note;
    values.tex        -- ``\\newcommand`` macros with every number the note quotes,
                         so the prose and tables can never drift from the data.

Run it from anywhere; paths are resolved relative to the repository root.

    python make_figures.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Locate the repository root and make the package importable -------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # docs/notes/1.0_OPDEstimation -> repo root
sys.path.insert(0, str(REPO))

from het_ifo_opd import OPDConfig, estimate_opd, load_phasemeter  # noqa: E402
from het_ifo_opd.estimators import single_bin_amplitude  # noqa: E402
from het_ifo_opd.physics import phase_cycles_to_opd  # noqa: E402
from het_ifo_opd.plotting import plot_diagnostics  # noqa: E402
from speckit import compute_spectrum  # noqa: E402

DATA = REPO / "data" / "FM1"
FIGS = HERE / "Figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
    }
)


@dataclass
class Case:
    tag: str            # LaTeX macro tag: A, B, C
    file: str
    mod_freq: float
    short: str


CASES = [
    Case("A", "FM1Day06_AnchoredAirOPD_EDUbase_R1_20260609_135823.zip", 100.0,
         "R1, anchored, air"),
    Case("B", "FM1Day09_VacuumReleasedOPD_SHIMMED_FLIGHTTORQUED_R1_20260629_140109.zip",
         95.0, "R1, released, vacuum"),
    Case("C", "FM1Day09_VacuumReleasedOPD_SHIMMED_FLIGHTTORQUED_R2_20260629_140110.zip",
         95.0, "R2, released, vacuum"),
]

cfg = OPDConfig()
macros: dict[str, str] = {}


def m(name: str, value: str) -> None:
    macros[name] = value


def num(x: float, fmt: str = "{:.3g}") -> str:
    return fmt.format(x)


# ---------------------------------------------------------------------------
# Run the pipeline on the three cases.
# ---------------------------------------------------------------------------
print("Running the pipeline on the three test cases ...")
results = {}
for c in CASES:
    path = str(DATA / c.file)
    r = estimate_opd(path, config=cfg, mod_freq=c.mod_freq)
    results[c.tag] = r
    t = c.tag
    m(f"case{t}file", c.file.replace("_", r"\_"))
    m(f"case{t}short", c.short)
    m(f"case{t}fmod", num(c.mod_freq, "{:.0f}"))
    m(f"case{t}opd", num(r.opd * 1e3, "{:.4f}"))          # mm
    m(f"case{t}opdum", num(r.opd * 1e6, "{:.1f}"))        # um
    m(f"case{t}noise", num(r.opd_unc * 1e6, "{:.2f}"))    # um
    m(f"case{t}drift", num(r.opd_drift * 1e6, "{:.1f}"))  # um
    m(f"case{t}snr", num(r.tone_snr, "{:.0f}"))
    m(f"case{t}coh", num(r.coherence, "{:.3f}"))
    m(f"case{t}leak", num(r.leakage_ratio, "{:.2f}"))
    m(f"case{t}mode", r.integration_mode)
    m(f"case{t}agree", num(r.method_agreement, "{:.1e}"))
    m(f"case{t}lockinopd", num(phase_cycles_to_opd(r.amp_cycles_lockin,
                                                   cfg.freq_dev_peak) * 1e3, "{:.4f}"))
    m(f"case{t}specopd", num(phase_cycles_to_opd(r.amp_cycles_spectral,
                                                 cfg.freq_dev_peak) * 1e3, "{:.4f}"))
    m(f"case{t}harm", num(r.harmonic_ratio, "{:.2e}"))
    m(f"case{t}freqoff", num(r.tone_freq_offset * 1e3, "{:+.2f}"))  # mHz
    ch = r.channel_opds
    m(f"case{t}chone", num(ch.get("ch1", float("nan")) * 1e3, "{:.1f}"))
    m(f"case{t}chtwo", num(ch.get("ch2", float("nan")) * 1e3, "{:.1f}"))
    print(f"  Case {t}: OPD={r.opd*1e3:.4f} mm  mode={r.integration_mode}  "
          f"coh={r.coherence:.3f}  leak={r.leakage_ratio:.2f}  SNR={r.tone_snr:.0f}")

# Physics macros.
m("freqdevpk", num(cfg.freq_dev_peak / 1e6, "{:.0f}"))
m("actuatortf", num(cfg.actuator_tf / 1e6, "{:.0f}"))
m("modvpp", num(cfg.mod_vpp, "{:.1f}"))
m("demodbw", num(cfg.demod_bandwidth, "{:.1f}"))
m("cohthresh", num(cfg.coherence_threshold, "{:.1f}"))

# ---------------------------------------------------------------------------
# Figure 1: common-mode rejection (Case 1, the clean anchored acquisition).
# ---------------------------------------------------------------------------
print("Figure: common-mode rejection ...")
data = load_phasemeter(str(DATA / CASES[0].file))
a = data.channel_cycles[0]
b = data.channel_cycles[1]
diff = a - b
f0 = CASES[0].mod_freq

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(data.t, a - a.mean(), lw=0.5, label="ch A")
ax[0].plot(data.t, b - b.mean(), lw=0.5, alpha=0.7, label="ch B")
ax[0].plot(data.t, diff - diff.mean(), lw=0.6, color="C2", label="A $-$ B")
ax[0].set(xlabel="Time [s]", ylabel="Phase [cyc]",
          title="Per-channel wander cancels in the difference")
ax[0].legend()

for sig, lab, col in [(a, "ch A", "C0"), (b, "ch B", "C1"),
                      (diff, "A $-$ B", "C2")]:
    s = compute_spectrum(sig - np.mean(sig), data.fs, win="kaiser", olap="default")
    ax[1].loglog(s.f, s.asd, lw=0.8, color=col, label=lab)
ax[1].axvline(f0, color="C3", ls="--", lw=1, label=f"{f0:.0f} Hz tone")
ax[1].set(xlabel="Frequency [Hz]", ylabel=r"ASD [cyc/$\sqrt{\mathrm{Hz}}$]",
          title="Tone is common-mode; residual OPD survives in A$-$B")
ax[1].legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_commonmode.pdf")
plt.close(fig)

# Common-mode numbers.
m("cmstda", num(np.std(a), "{:.3g}"))
m("cmstdb", num(np.std(b), "{:.3g}"))
m("cmstddiff", num(np.std(diff), "{:.3g}"))
m("cmrejection", num(np.std(a) / np.std(diff), "{:.0f}"))
for lab, key in [(a, "cma"), (b, "cmb"), (diff, "cmdiff")]:
    amp = single_bin_amplitude(lab - np.mean(lab), data.fs, f0)
    opd = phase_cycles_to_opd(amp, cfg.freq_dev_peak)
    m(f"{key}tone", num(amp, "{:.3e}"))
    m(f"{key}opd", num(opd * 1e3, "{:.3f}"))

# ---------------------------------------------------------------------------
# Figures 2-4: per-case three-panel diagnostics.
# ---------------------------------------------------------------------------
for c in CASES:
    print(f"Figure: diagnostics case {c.tag} ...")
    fig = plot_diagnostics(results[c.tag], config=cfg)
    fig.savefig(FIGS / f"fig_case{c.tag}_diag.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: estimator-comparison / leakage bar chart across the three cases.
# ---------------------------------------------------------------------------
print("Figure: estimator comparison (leakage) ...")
fig, ax = plt.subplots(figsize=(8.2, 4.2))
labels = [f"Case {c.tag}\n{c.short}" for c in CASES]
x = np.arange(len(CASES))
w = 0.26
demod = [results[c.tag].opd * 1e3 for c in CASES]
lockin = [phase_cycles_to_opd(results[c.tag].amp_cycles_lockin,
                              cfg.freq_dev_peak) * 1e3 for c in CASES]
spec = [phase_cycles_to_opd(results[c.tag].amp_cycles_spectral,
                            cfg.freq_dev_peak) * 1e3 for c in CASES]
ax.bar(x - w, demod, w, label="demodulator (headline)", color="C0")
ax.bar(x, lockin, w, label="rectangular lock-in", color="C3", alpha=0.85)
ax.bar(x + w, spec, w, label="windowed single-bin", color="C2", alpha=0.85)
for i, c in enumerate(CASES):
    r = results[c.tag]
    ax.annotate(f"leak\n$\\times${r.leakage_ratio:.2f}",
                (i, max(demod[i], lockin[i], spec[i])),
                textcoords="offset points", xytext=(0, 4),
                ha="center", va="bottom", fontsize=8, color="0.25")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Residual OPD [mm]")
ax.set_title("Three amplitude estimators: the lock-in is biased by leakage")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_leakage.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6: OPD(t) of the three cases stacked (stability vs drift vs incoherence).
# ---------------------------------------------------------------------------
print("Figure: OPD(t) comparison ...")
fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=False)
for ax, c in zip(axes, CASES):
    r = results[c.tag]
    ax.plot(r.t_bb, r.opd_t * 1e3, lw=0.7, color="C0", alpha=0.85, label="OPD(t)")
    ax.axhline(r.opd * 1e3, color="C3", lw=1.2,
               label=f"headline ({r.integration_mode})")
    ax.axhspan((r.opd - r.opd_drift) * 1e3, (r.opd + r.opd_drift) * 1e3,
               color="C3", alpha=0.15)
    ax.set(ylabel="OPD [mm]",
           title=f"Case {c.tag} ({c.short}): OPD={r.opd*1e3:.4f} mm, "
                 f"drift $\\pm${r.opd_drift*1e6:.0f} um, "
                 f"noise $\\pm${r.opd_unc*1e6:.2f} um, "
                 f"coh {r.coherence:.2f}, leak $\\times${r.leakage_ratio:.2f}")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Time [s]")
fig.tight_layout()
fig.savefig(FIGS / "fig_opd_t.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Write the LaTeX macro file.
# ---------------------------------------------------------------------------
print("Writing values.tex ...")
lines = ["% Auto-generated by make_figures.py -- do not edit by hand.",
         "% Regenerate with:  make figs   (or  python make_figures.py)"]
for k in sorted(macros):
    lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
(HERE / "values.tex").write_text("\n".join(lines) + "\n")

print("Done. Figures in", FIGS, "and values.tex written.")
