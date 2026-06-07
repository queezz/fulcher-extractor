"""Summarize clean-line sigma statistics from a Fulcher fit report."""

from __future__ import annotations

import argparse
from pathlib import Path

from fulcher_extractor.sigma_stats import write_sigma_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.fit_report.parent
    clean_path, summary_path, stats = write_sigma_stats(
        args.fit_report,
        output_dir=output_dir,
    )

    print(f"clean lines: {clean_path}")
    print(f"summary: {summary_path}")
    print(
        "sigma_nm: "
        f"count={stats.count} "
        f"median={stats.median:.6f} "
        f"q10={stats.q10:.6f} "
        f"q90={stats.q90:.6f} "
        f"min={stats.min:.6f} "
        f"max={stats.max:.6f}"
    )


if __name__ == "__main__":
    main()
