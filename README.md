# het_ifo_opd

**Optimal estimation of interferometer residual optical path-length difference
(OPD) from frequency-modulated Moku:Pro Phasemeter data.**

A known single-tone modulation of the laser frequency imprints a phase tone on
the differential interferometer phase whose amplitude is directly proportional
to the residual OPD. `het_ifo_opd` ingests Moku:Pro Phasemeter files, estimates
that tone amplitude with a leakage-immune complex-baseband demodulator,
cross-checks it spectrally, and converts it into an OPD (and an OPD *time
series*) with a rigorous, statistically-calibrated uncertainty.

---

## 1. Measurement principle

A two-beam interferometer with optical path-length difference `OPD` converts the
instantaneous laser frequency `ν(t)` into an interferometric phase

```
φ(t) = ν(t) · OPD / c          [cycles]
```

so a small laser-frequency modulation `δν(t)` produces a phase modulation

```
δφ(t) = (OPD / c) · δν(t).
```

Driving the laser-frequency actuator with a single tone at `f_mod` therefore
imprints a phase tone of amplitude

```
A_φ = (OPD / c) · δν_pk      ⇒      OPD = A_φ · c / δν_pk ,
```

where the **peak** frequency deviation is

```
δν_pk = (dν/dV) · V_pp / 2 = 376 MHz/V · (1 V_pp / 2) = 188 MHz.
```

Key consequences exploited by the package:

* **Wavelength-independent.** The OPD follows from the *frequency* modulation
  alone; the 1550 nm carrier is needed only for context, not for the estimate.
* **Linear.** Each Fourier component of `δν` maps to the same component of `φ`,
  so the OPD is read from the **fundamental** tone; harmonics (from drive or
  actuator distortion) are nuisances, not signal.
* **Differential observable.** The two phasemeter channels share the laser, so
  their **difference** `φ = ch1 − ch2` rejects the common laser frequency noise
  (~1700× in this data) and isolates the *residual* differential OPD, exactly
  the quantity of interest. Per-channel absolute OPDs (~4–5 cm here) are
  retained as diagnostics.

## 2. Estimation method

For each file the pipeline:

1. **Uses the known modulation frequency** supplied by the user (per file or via
   a filename map), optionally refining it on a fine periodogram grid to absorb
   drive/digitizer clock offsets.
2. **Estimates the amplitude with a complex-baseband demodulator** (primary).
   The differential phase is heterodyned to DC (`φ·e^{−2πi f₀ t}`), low-passed
   to a narrow one-sided bandwidth `B` around the tone, and decimated, giving
   the **instantaneous tone phasor** `z(t)` with `|z(t)|` the zero-to-peak
   amplitude. The narrow low-pass makes the estimate **immune to spectral
   leakage** from the very large low-frequency differential-phase wander (the
   `ν₀·OPD(t)/c` term, tens of cycles RMS) that otherwise leaks into the tone
   bin through the `1/f` sidelobes of a rectangular-window fit — a bias that
   inflated the naive lock-in by up to ~2× and the short-segment estimates by
   ~10× on the released/torqued acquisitions.

   The headline amplitude is chosen from a **measured coherence**
   `ρ = |⟨z⟩| / √⟨|z|²⟩` (constant frequency-offset removed):
   * `ρ ≥ coherence_threshold` → **coherent** integration `|⟨z⟩|` (optimal SNR);
   * otherwise → **incoherent**, noise-debiased `√(⟨|z|²⟩ − ⟨|z_off|²⟩)`, the
     correct estimator for a genuinely phase-wandering tone.

   The instantaneous `OPD(t) = |z(t)|·c/δν_pk` is retained as the deliverable;
   for acquisitions whose OPD physically evolves (released/torqued) it is
   summarised as `mean ± drift`.
3. **Cross-checks** with the rectangular-window least-squares lock-in (now a
   **leakage diagnostic**, reported as `leakage_ratio = A_lockin / A_demod`) and
   an independent windowed LPSD single-bin estimate (`speckit`). Since the
   demodulator and the windowed single-bin are both leakage-suppressed, their
   `method_agreement` is small (≲1 %) when the tone is well-behaved and flags a
   real problem otherwise. `tone_snr = A/σ_A = amp_cycles / amp_unc_cycles`.
4. **Quantifies uncertainty:**
   * *Coherent noise floor:* `σ_A = √(⟨|z_off|²⟩ / N_eff)` with `N_eff = T·2B`
     independent looks, measured from an **off-tone** demodulation — equal to
     the Cramér–Rao bound `√(S_n(f₀)/T)`. *(Verified calibrated by Monte-Carlo —
     see `tests/`.)*
   * *Physical drift:* the standard deviation of `OPD(t)` (real non-stationarity,
     distinct from the measurement noise floor).
   * *Empirical (per-segment):* retained as a diagnostic (note: leakage-prone).
5. **Converts to OPD** via `OPD = A · c / δν_pk`.

## 3. Installation

```bash
pip install -e .          # uses the dependencies in pyproject.toml / requirements.txt
```

Requires `numpy`, `scipy`, `pandas`, `matplotlib`, and the Liquid Instruments
helpers `speckit` and `mokutools` (already present in the project `.venv`).

## 4. Usage

### Python API

```python
from het_ifo_opd import estimate_opd, OPDConfig

# Known modulation frequency for this file:
r = estimate_opd("FM1/FM1Day06_AnchoredAirOPD_EDUbase_R1_20260609_135823.zip",
                 mod_freq=100.0)
print(f"OPD = {r.opd*1e3:.4f} mm  ± {r.opd_unc*1e6:.2f} µm (noise floor)")
print(f"          drift ± {r.opd_drift*1e6:.1f} µm  [{r.integration_mode}, "
      f"coherence {r.coherence:.2f}]")
print(f"tone {r.tone_freq:.3f} Hz (offset {r.tone_freq_offset*1e3:+.2f} mHz), "
      f"SNR {r.tone_snr:.3g}, leakage×{r.leakage_ratio:.2f}, "
      f"method agreement {r.method_agreement:.1e}")

# The instantaneous OPD time series (baseband) is the deliverable:
#   r.t_bb  [s],  r.opd_t  [m]

```

A whole dataset, assigning the frequency per acquisition day:

```python
from het_ifo_opd import estimate_opd_dataset

def freq(path):
    return 95.0 if "Day09" in path else 100.0

ds = estimate_opd_dataset("FM1/", freq_resolver=freq)
print(ds.to_dataframe())
```

### Command line

```bash
# Mixed dataset, assigning the frequency by filename substring:
python -m het_ifo_opd FM1/ --freq-map Day09=95 Day06=100 --out results --plots
```

### Reproduce the FM1 analysis

```bash
python scripts/analyze_fm1.py FM1 results
```

writes `results/opd_results.csv`, a per-file 3-panel diagnostic plot (ASD with
the tone marked · phase-folded synchronous-averaged tone · instantaneous
`OPD(t)` with the headline mode-selected OPD, drift band, and the leakage-prone
per-segment estimates for comparison), and a `summary.png` bar chart.

## 5. Package layout

```
het_ifo_opd/
  config.py       OPDConfig: physics + analysis parameters (FM1 defaults)
  physics.py      phase-amplitude  <->  OPD conversions
  io.py           load Moku:Pro Phasemeter files -> differential phase
  estimators.py   frequency refinement, complex-baseband demodulator (primary),
                  least-squares lock-in, spectral single-bin, local noise PSD,
                  per-segment amplitudes
  pipeline.py     estimate_opd / estimate_opd_dataset -> OPDResult / DatasetResult
  plotting.py     per-file diagnostics and dataset summary
  cli.py          `python -m het_ifo_opd` command-line interface
scripts/analyze_fm1.py   authoritative FM1 run
tests/test_validation.py synthetic accuracy & uncertainty-calibration tests
```

## 6. Validation

`python tests/test_validation.py` (or `pytest -q`) verifies on synthetic data
that the estimator recovers an injected OPD to <1 % on realistic low-frequency-
dominated noise, that the reported coherent uncertainty matches the Monte-Carlo
scatter (calibrated, near the Cramér–Rao bound), and specifically that the
demodulator is **leakage-immune** — under a 16-cycle low-frequency wander the
rectangular lock-in is biased >2× high while the demodulator recovers the truth
to a few percent — and that it correctly falls back to **incoherent** integration
for a genuinely wandering (chirped) tone.
