#!/usr/bin/env python
"""Authoritative analysis of the FM1 data set.

Runs the OPD estimator over every FM1 acquisition, assigning the correct
modulation frequency per acquisition day (Day06 = 100 Hz, Day09 = 95 Hz), and
writes a CSV summary plus per-file and dataset-level diagnostic plots.

Usage
-----
    python scripts/analyze_fm1.py [FM1_DIR] [OUT_DIR]

Defaults: FM1_DIR=FM1, OUT_DIR=results.
"""
from __future__ import annotations

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

from het_ifo_opd import OPDConfig, estimate_opd_dataset
from het_ifo_opd.plotting import plot_dataset_summary, plot_diagnostics


def fm1_freq_resolver(path: str):
    """Modulation frequency [Hz] by acquisition day for the FM1 set."""
    base = os.path.basename(path)
    if "Day09" in base:
        return 95.0
    if "Day06" in base:
        return 100.0
    return None  # fall back to config candidates/auto-detection


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    fm1_dir = argv[0] if len(argv) > 0 else "FM1"
    out_dir = argv[1] if len(argv) > 1 else "results"
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("analyze_fm1")

    # Physics defaults already match FM1 (1550 nm, 376 MHz/V, 1 Vpp). The
    # per-file frequency is supplied by the resolver below.
    config = OPDConfig()

    dataset = estimate_opd_dataset(
        fm1_dir, config=config, logger=logger, freq_resolver=fm1_freq_resolver
    )

    df = dataset.to_dataframe().sort_values("name").reset_index(drop=True)
    csv_path = os.path.join(out_dir, "opd_results.csv")
    df.to_csv(csv_path, index=False)

    import pandas as pd

    with pd.option_context("display.max_columns", None, "display.width", 220,
                           "display.float_format", lambda x: f"{x:.4g}"):
        print("\n" + df.to_string(index=False))

    for r in dataset:
        fig = plot_diagnostics(r, config=config)
        fig.savefig(os.path.join(out_dir, f"{r.name}.png"), dpi=130,
                    bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

    fig = plot_dataset_summary(dataset)
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=130, bbox_inches="tight")

    print(f"\nWrote {csv_path} and diagnostic plots to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
