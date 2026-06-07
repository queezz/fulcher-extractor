"""Run one policy-aware H2 Fulcher extraction frame for development QC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from fulcher_extractor.extract import extract_lines
from fulcher_extractor.fit import FitConfig
from fulcher_extractor.line_database import load_lines
from fulcher_extractor.line_policy import load_line_policies
from fulcher_extractor.output import results_to_dataframe, write_fulcheranalyzer_csvs
from fulcher_extractor.qc import plot_region, write_line_fit_qc
from fulcher_extractor.spectrocube_io import load_spectrum


DEFAULT_CUBE_DIR = Path(
    r"C:\Users\queezz\Dropbox\Experiments\2025-LHD-BH\Echelle\20250926-spectrocubes"
)
DEFAULT_SHOT = 193809
DEFAULT_FRAME = 9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--cube-dir", type=Path, default=DEFAULT_CUBE_DIR)
    parser.add_argument("--engine", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local/runs/policy_single_frame_193809_fr9"),
    )
    args = parser.parse_args()

    cube_path = args.cube_dir / f"{args.shot}_Echelle_spectrocube_wmsr_403nm.nc"
    spectrum = load_spectrum(cube_path, frame=args.frame, engine=args.engine)
    lines = load_lines()
    policies = load_line_policies()
    used_line_ids = {
        line_id
        for line_id, policy in policies.items()
        if policy.line_scale_role == "used"
    }
    label_lines = [line for line in lines if line.line_id in used_line_ids]

    config = FitConfig(
        line_left_width_nm=0.24,
        line_right_width_nm=0.18,
        center_left_offset_nm=0.16,
        center_right_offset_nm=0.08,
        instrument_sigma_nm=0.0273,
        instrument_sigma_leeway_nm=0.015,
        close_neighbor_threshold_nm=0.15,
    )
    results = extract_lines(spectrum, lines=lines, config=config)

    output_dir = args.output_dir
    intensity_dir = output_dir / "intensities"
    fit_report_dir = output_dir / "fit_reports"
    qc_region_dir = output_dir / "qc_region_figures"
    qc_line_dir = output_dir / "qc_line_figures"
    for path in (intensity_dir, fit_report_dir, qc_region_dir, qc_line_dir):
        path.mkdir(parents=True, exist_ok=True)

    _, _, fit_report_path = write_fulcheranalyzer_csvs(
        results,
        output_dir=intensity_dir,
        shot=args.shot,
        frame=args.frame,
        metadata={"policy_layer": "line_policies.toml"},
    )
    fit_table = results_to_dataframe(results)
    fit_table.to_csv(fit_report_dir / fit_report_path.name, index=False)

    summary = (
        fit_table.groupby(["legacy_policy", "legacy_matrix_action"], dropna=False)
        .size()
        .rename("n_lines")
        .reset_index()
    )
    summary.to_csv(output_dir / "policy_summary.csv", index=False)

    region_fig = plot_region(
        spectrum,
        lines=lines,
        label_lines=label_lines,
        guide_lines=label_lines,
        output_path=qc_region_dir / f"{args.shot}_fr_{args.frame}_600_630.png",
    )
    plt.close(region_fig)
    line_paths = write_line_fit_qc(
        spectrum,
        results,
        pdf_path=qc_line_dir / f"{args.shot}_fr_{args.frame}_line_fits.pdf",
        columns=5,
    )

    print(f"cube: {cube_path}")
    print(f"fit report: {fit_report_dir / fit_report_path.name}")
    print(f"policy summary: {output_dir / 'policy_summary.csv'}")
    print(f"line QC: {line_paths[0] if line_paths else ''}")


if __name__ == "__main__":
    main()
