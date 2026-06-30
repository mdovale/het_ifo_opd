"""Command-line interface for het_ifo_opd.

Examples
--------
Analyse a whole folder and write a CSV summary + diagnostic plots::

    python -m het_ifo_opd FM1/ --out results --plots

Analyse selected files with a custom modulation set-up::

    python -m het_ifo_opd a.zip b.zip --actuator-tf 376e6 --mod-vpp 1.0 --mod-freq 100
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

from .config import OPDConfig
from .pipeline import estimate_opd_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="het_ifo_opd",
        description="Estimate residual interferometer OPD from frequency-modulated "
        "Moku:Pro Phasemeter data.",
    )
    p.add_argument("paths", nargs="+", help="Files, a directory, or glob pattern(s).")
    p.add_argument("--out", default=None, help="Output directory for CSV/plots.")
    p.add_argument("--plots", action="store_true", help="Save per-file diagnostic plots.")

    g = p.add_argument_group("physics")
    g.add_argument("--wavelength", type=float, default=1550e-9, help="Laser wavelength [m].")
    g.add_argument("--actuator-tf", type=float, default=376e6, help="Actuator TF [Hz/V].")
    g.add_argument("--mod-vpp", type=float, default=1.0, help="Modulation amplitude [Vpp].")
    g.add_argument("--mod-freq", type=float, default=100.0,
                   help="Modulation frequency [Hz] (default for all files).")
    g.add_argument("--freq-map", nargs="+", default=None, metavar="SUBSTR=FREQ",
                   help="Assign the modulation frequency by filename "
                   "substring, e.g. 'Day09=95 Day06=100'. Overrides --mod-freq "
                   "for matching files.")

    g = p.add_argument_group("analysis")
    g.add_argument("--channels", type=int, nargs=2, default=None,
                   help="1-indexed channels for the differential phase (e.g. 1 2).")
    g.add_argument("--no-refine", action="store_true", help="Do not refine tone frequency.")
    g.add_argument("--harmonics", type=int, default=3, help="Number of fitted harmonics.")
    g.add_argument("--detrend-order", type=int, default=3, help="Polynomial detrend order.")
    g.add_argument("--segments", type=int, default=10, help="Stability segments.")
    g.add_argument("--start-time", type=float, default=None, help="Analysis start [s].")
    g.add_argument("--duration", type=float, default=None, help="Analysis duration [s].")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    logger = logging.getLogger("het_ifo_opd")

    config = OPDConfig(
        laser_wavelength=args.wavelength,
        actuator_tf=args.actuator_tf,
        mod_vpp=args.mod_vpp,
        mod_freq=args.mod_freq,
        channels=tuple(args.channels) if args.channels else None,
        refine_frequency=not args.no_refine,
        n_harmonics=args.harmonics,
        detrend_order=args.detrend_order,
        n_stability_segments=args.segments,
        start_time=args.start_time,
        duration=args.duration,
    )

    freq_resolver = _build_freq_resolver(args.freq_map)
    dataset = estimate_opd_dataset(
        args.paths, config=config, logger=logger, freq_resolver=freq_resolver
    )
    df = dataset.to_dataframe()

    # Pretty console table.
    with_pd_display(df)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        csv_path = os.path.join(args.out, "opd_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nWrote {csv_path}")

        if args.plots:
            from .plotting import plot_dataset_summary, plot_diagnostics
            import matplotlib
            matplotlib.use("Agg")
            for r in dataset:
                fig = plot_diagnostics(r, config=config)
                fpath = os.path.join(args.out, f"{r.name}.png")
                fig.savefig(fpath, dpi=130, bbox_inches="tight")
            fig = plot_dataset_summary(dataset)
            fig.savefig(os.path.join(args.out, "summary.png"), dpi=130,
                        bbox_inches="tight")
            print(f"Wrote diagnostic plots to {args.out}/")

    return 0


def _build_freq_resolver(freq_map):
    """Turn ``['Day09=95', 'Day06=100']`` into a ``path -> freq`` resolver."""
    if not freq_map:
        return None
    rules = []
    for entry in freq_map:
        if "=" not in entry:
            raise ValueError(f"--freq-map entry must be SUBSTR=FREQ, got {entry!r}")
        substr, freq = entry.split("=", 1)
        rules.append((substr, float(freq)))

    def resolver(path):
        base = os.path.basename(path)
        for substr, freq in rules:
            if substr in base:
                return freq
        return None

    return resolver


def with_pd_display(df):
    import pandas as pd
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda x: f"{x:.4g}",
    ):
        print(df.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
