"""Run one H2 frame from extraction through Boltzmann and coronal fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT.parent
ANALYZER_ROOT = CODE_ROOT / "fulcheranalyzer"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(ANALYZER_ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from fulcher_analyzer import (
    BoltzmannPlot,
    CoronaModel,
    apply_boltzmann_qc_mask,
    boltzmann_qc_points,
    plot_boltzmann_qc,
    read_intensities,
)
from fulcher_extractor.extract import extract_lines
from fulcher_extractor.fit import FitConfig
from fulcher_extractor.line_database import load_lines
from fulcher_extractor.line_policy import load_line_policy_set, overview_qc_lines
from fulcher_extractor.output import results_to_dataframe, write_fulcheranalyzer_csvs
from fulcher_extractor.qc import plot_region, write_line_fit_qc
from fulcher_extractor.spectrocube_io import load_spectrum


DEFAULT_CUBE_DIR = Path(
    "/Users/queezz/Dropbox/Experiments/2025-LHD-BH/Echelle/20250926-spectrocubes"
)
DEFAULT_SHOT = 193809
DEFAULT_FRAME = 9
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local/runs/20260610-h2-e2e-193809-fr9"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--cube-dir", type=Path, default=DEFAULT_CUBE_DIR)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-fit-relerr", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    intensity_dir = output_dir / "intensities"
    fit_report_dir = output_dir / "fit_reports"
    qc_region_dir = output_dir / "qc_region_figures"
    qc_line_dir = output_dir / "qc_line_figures"
    boltzmann_dir = output_dir / "boltzmann"
    coronal_dir = output_dir / "coronal"
    for path in (
        intensity_dir,
        fit_report_dir,
        qc_region_dir,
        qc_line_dir,
        boltzmann_dir,
        coronal_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    cube_path = args.cube_dir / f"{args.shot}_Echelle_spectrocube_wmsr_403nm.nc"
    spectrum = load_spectrum(cube_path, frame=args.frame, engine=args.engine)
    lines = load_lines()
    policy_set = load_line_policy_set()
    label_lines = overview_qc_lines(lines, policy_set=policy_set)

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

    _, _, fit_report_path = write_fulcheranalyzer_csvs(
        results,
        output_dir=intensity_dir,
        shot=args.shot,
        frame=args.frame,
        metadata={"policy_layer": "line_policies.toml"},
    )
    fit_table = results_to_dataframe(results)
    fit_report_copy = fit_report_dir / fit_report_path.name
    fit_table.to_csv(fit_report_copy, index=False)

    policy_summary = (
        fit_table.groupby(["legacy_policy", "legacy_matrix_action"], dropna=False)
        .size()
        .rename("n_lines")
        .reset_index()
    )
    policy_summary.to_csv(output_dir / "policy_summary.csv", index=False)

    region_fig = plot_region(
        spectrum,
        lines=lines,
        label_lines=label_lines,
        guide_lines=label_lines,
        output_path=qc_region_dir / f"{args.shot}_fr_{args.frame}_600_630.png",
    )
    plt.close(region_fig)
    write_line_fit_qc(
        spectrum,
        results,
        pdf_path=qc_line_dir / f"{args.shot}_fr_{args.frame}_line_fits.pdf",
        columns=5,
    )

    intensities = read_intensities(args.shot, args.frame, data_folder=intensity_dir)
    bp = BoltzmannPlot(intensities, "h")
    points = boltzmann_qc_points(
        bp,
        max_fit_relerr=args.max_fit_relerr,
        fit_report=fit_report_copy,
    )
    apply_boltzmann_qc_mask(bp, points)
    bp.autofit()

    points = boltzmann_qc_points(
        bp,
        max_fit_relerr=args.max_fit_relerr,
        fit_report=fit_report_copy,
    )
    points.to_csv(
        boltzmann_dir / f"{args.shot}_fr_{args.frame}_boltzmann_qc_points.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "parameter": ["alpha", "beta", "Trot1", "Trot2"],
            "value": [bp.alpha, bp.beta, bp.trot1, bp.trot2],
            "stderr": [bp.err[0], bp.err[1], bp.err[2], bp.err[3]],
        }
    ).to_csv(
        boltzmann_dir / f"{args.shot}_fr_{args.frame}_boltzmann_qc_summary.csv",
        index=False,
    )
    bp.trotall.to_csv(
        boltzmann_dir / f"{args.shot}_fr_{args.frame}_boltzmann_qc_trot.csv",
        index=False,
    )
    pd.DataFrame({"band": list(bp.nd.columns), "nd_vibrofit": bp.nd_vibrofit}).to_csv(
        boltzmann_dir / f"{args.shot}_fr_{args.frame}_boltzmann_qc_nd_vibrofit.csv",
        index=False,
    )

    fig = plot_boltzmann_qc(
        bp,
        points,
        title=f"H2 {args.shot} frame {args.frame}: d-state Boltzmann fit",
    )
    fig.savefig(
        boltzmann_dir / f"{args.shot}_fr_{args.frame}_boltzmann_qc.png",
        dpi=200,
    )
    plt.close(fig)

    cm = CoronaModel(bp)
    cm.coronal_autofit()
    pd.DataFrame(
        {
            "parameter": ["Tvib"],
            "value": [cm.tvib],
            "stderr": [cm.tviberr],
        }
    ).to_csv(coronal_dir / f"{args.shot}_fr_{args.frame}_coronal_summary.csv", index=False)

    _save_current_figure(
        coronal_dir / f"{args.shot}_fr_{args.frame}_coronal_result.png",
        cm.plot_coronal_result,
        dpi=200,
    )
    _save_current_figure(
        coronal_dir / f"{args.shot}_fr_{args.frame}_coronal_compare.png",
        cm.plot_paper_compare,
        dpi=200,
    )
    _save_current_figure(
        coronal_dir / f"{args.shot}_fr_{args.frame}_x_d_populations.png",
        cm.plot_xd,
        dpi=200,
    )
    _save_current_figure(
        coronal_dir / f"{args.shot}_fr_{args.frame}_rmatrix.png",
        cm.plot_R,
        dpi=180,
    )
    _save_current_figure(
        coronal_dir / f"{args.shot}_fr_{args.frame}_x_v_contribution.png",
        cm.plot_contribution,
        dpi=200,
    )

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                f"# H2 end-to-end run: {args.shot} frame {args.frame}",
                "",
                "Stages:",
                "",
                "1. Policy-aware Fulcher-alpha intensity extraction.",
                "2. New-style two-temperature d-state Boltzmann QC fit.",
                "3. Current coronal-model Tvib fit and confirmation plots.",
                "",
                f"Input cube: `{cube_path}`",
                f"Output directory: `{output_dir}`",
                "",
                "Fit summary:",
                "",
                f"- alpha = {bp.alpha:.6g} +/- {bp.err[0]:.3g}",
                f"- beta = {bp.beta:.6g} +/- {bp.err[1]:.3g}",
                f"- Trot1 = {bp.trot1:.3f} +/- {bp.err[2]:.3f} K",
                f"- Trot2 = {bp.trot2:.3f} +/- {bp.err[3]:.3f} K",
                f"- Tvib = {cm.tvib:.3f} +/- {cm.tviberr:.3f} K",
                "",
                "Key plots:",
                "",
                f"- `{qc_region_dir.name}/{args.shot}_fr_{args.frame}_600_630.png`",
                f"- `{qc_line_dir.name}/{args.shot}_fr_{args.frame}_line_fits.pdf`",
                f"- `{boltzmann_dir.name}/{args.shot}_fr_{args.frame}_boltzmann_qc.png`",
                f"- `{coronal_dir.name}/{args.shot}_fr_{args.frame}_coronal_result.png`",
                f"- `{coronal_dir.name}/{args.shot}_fr_{args.frame}_coronal_compare.png`",
                f"- `{coronal_dir.name}/{args.shot}_fr_{args.frame}_x_d_populations.png`",
                f"- `{coronal_dir.name}/{args.shot}_fr_{args.frame}_rmatrix.png`",
                f"- `{coronal_dir.name}/{args.shot}_fr_{args.frame}_x_v_contribution.png`",
                "",
            ]
        )
    )

    print(f"run directory: {output_dir}")
    print(f"Boltzmann: Trot1={bp.trot1:.3f} K, Trot2={bp.trot2:.3f} K")
    print(f"Coronal: Tvib={cm.tvib:.3f} K +/- {cm.tviberr:.3f} K")


def _save_current_figure(path: Path, plotter, *, dpi: int) -> None:
    plt.figure()
    plotter()
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
