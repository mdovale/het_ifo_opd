# het_ifo_opd

**Optimal estimation of interferometer residual optical path-length difference
(OPD) from frequency-modulated Moku:Pro Phasemeter data.**

A known single-tone modulation of the laser frequency imprints a phase tone on
the differential interferometer phase whose amplitude is directly proportional
to the residual OPD. `het_ifo_opd` ingests Moku:Pro Phasemeter files, estimates
that tone amplitude with an optimal (maximum-likelihood) least-squares lock-in,
cross-checks it spectrally, and converts it into an OPD with a rigorous,
statistically-calibrated uncertainty.

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
2. **Estimates the amplitude with a least-squares lock-in** (primary). It fits

   ```
   φ(t) = Σ_k [a_k cos + b_k sin](2π k f₀ t)  +  Σ_p c_p t^p
   ```

   The fundamental amplitude `√(a₁² + b₁²)` is the OPD signal; harmonics and a
   polynomial trend (slow laser-noise drift) are fitted as nuisances. For a
   single tone of known frequency this is the maximum-likelihood / minimum-
   variance estimator.
3. **Cross-checks spectrally** with an independent LPSD single-bin estimate
   (`speckit`), where `A = √(2·PS)`. Agreement of the two methods is reported as
   a quality metric (≲0.1 % on the clean, high-SNR acquisitions).
   The reported `tone_snr` is the coherent amplitude SNR `A/σ_A`, equal to
   `amp_cycles / amp_unc_cycles`.
4. **Quantifies uncertainty two ways:**
   * *Coherent (analytic):* `σ_A = √(S_n(f₀)/T)`, with the local background
     noise PSD `S_n` measured **on the lock-in residual** (the tone removed in
     the time domain, so its spectral leakage cannot inflate the floor). This
     equals the Cramér–Rao bound `σ·√(2/N)` for white noise. *(Verified
     calibrated by Monte-Carlo — see `tests/`.)*
   * *Empirical (per-segment):* the scatter of the estimate across contiguous
     segments, which additionally captures genuine slow OPD drift / non-
     stationarity.
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
print(f"OPD = {r.opd*1e3:.4f} mm  ± {r.opd_unc*1e6:.2f} µm (coherent)")
print(f"            ± {r.opd_unc_empirical*1e6:.1f} µm (per-segment drift)")
print(f"tone {r.tone_freq:.3f} Hz, SNR {r.tone_snr:.3g}, "
      f"method agreement {r.method_agreement:.1e}")
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
the tone marked · phase-folded synchronous-averaged tone · per-segment OPD
stability), and a `summary.png` bar chart.

## 5. Package layout

```
het_ifo_opd/
  config.py       OPDConfig: physics + analysis parameters (FM1 defaults)
  physics.py      phase-amplitude  <->  OPD conversions
  io.py           load Moku:Pro Phasemeter files -> differential phase
  estimators.py   frequency refinement, least-squares lock-in, spectral single-bin,
                  local noise PSD, per-segment amplitudes
  pipeline.py     estimate_opd / estimate_opd_dataset -> OPDResult / DatasetResult
  plotting.py     per-file diagnostics and dataset summary
  cli.py          `python -m het_ifo_opd` command-line interface
scripts/analyze_fm1.py   authoritative FM1 run
tests/test_validation.py synthetic accuracy & uncertainty-calibration tests
```

## 6. Validation

`python tests/test_validation.py` (or `pytest -q`) verifies on synthetic data
that the estimator recovers an injected OPD to <1 % on realistic low-frequency-
dominated noise, that the lock-in and spectral methods agree, and that the
reported coherent uncertainty matches the Monte-Carlo scatter (i.e. it is
statistically calibrated and near the Cramér–Rao bound).
