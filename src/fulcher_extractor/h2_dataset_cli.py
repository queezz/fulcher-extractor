"""Run H2 Fulcher extraction plus Boltzmann/coronal summaries for a cube dataset."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

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
from fulcher_extractor.spectrocube_io import load_spectrum, parse_shot_id


DEFAULT_CUBE_GLOB = (
    "C:/Users/queez/Dropbox/Experiments/2025-LHD-BH/Echelle/"
    "20250926-spectrocubes/*_spectrocube_wmsr_403nm.nc"
)
DEFAULT_OUTPUT_DIR = Path("local/runs/20260611-h2-dataset-20250926")
DEFAULT_SHOTS_TOML_CANDIDATES = (
    Path(
        "C:/Users/queez/Dropbox/10-Research/50-Conferences/"
        "2026-05-17-PSI/analysis/notebooks/shots.toml"
    ),
    Path("local/shots.toml"),
)
DEFAULT_SHOTS_TOML = next(
    (path for path in DEFAULT_SHOTS_TOML_CANDIDATES if path.is_file()),
    DEFAULT_SHOTS_TOML_CANDIDATES[0],
)
DEFAULT_SIGNAL_MIN_NM = 600.0
DEFAULT_SIGNAL_MAX_NM = 630.0
PLAN_PATH_KEYS = {"output_dir", "manifest", "shots_toml"}
PLAN_LIST_AS_CSV_KEYS = {"shot_groups"}
DEFAULT_SCAN_THRESHOLDS = (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06)


def _provided_destinations(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    provided: set[str] = set()
    actions = [action for action in parser._actions if action.dest != argparse.SUPPRESS]
    for token in argv:
        if token in {"scan", "run"}:
            provided.add("command")
        for action in actions:
            for option in action.option_strings:
                if token == option or token.startswith(f"{option}="):
                    provided.add(action.dest)
    return provided


def _plan_path(value: object, plan_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return plan_path.parent / path


def _apply_plan(args: argparse.Namespace, plan_path: Path, provided: set[str]) -> None:
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        raise SystemExit(f"Plan file not found: {plan_path}")
    with plan_path.open("rb") as fh:
        raw = tomllib.load(fh)

    sections = []
    for section_name in ("common", args.command):
        section = raw.get(section_name, {})
        if section:
            if not isinstance(section, dict):
                raise SystemExit(f"Plan section [{section_name}] must be a TOML table.")
            sections.append((section_name, section))

    for section_name, section in sections:
        for key, value in section.items():
            if key in provided:
                continue
            if not hasattr(args, key):
                raise SystemExit(f"Unknown plan key [{section_name}].{key}")
            if key in PLAN_PATH_KEYS and value not in (None, ""):
                value = _plan_path(value, plan_path)
            elif key in PLAN_LIST_AS_CSV_KEYS and isinstance(value, list):
                value = ",".join(str(item) for item in value)
            setattr(args, key, value)


def _progress(
    iterable,
    *,
    total: int,
    desc: str,
    every: int,
    enabled: bool = True,
):
    """Yield items with tqdm when available, otherwise a compact text bar."""
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm import tqdm
    except ImportError:
        pass
    else:
        yield from tqdm(iterable, total=total, desc=desc, unit="item")
        return

    start = time.monotonic()
    last_len = 0
    stream = sys.stderr
    interactive = hasattr(stream, "isatty") and stream.isatty()

    def write(index: int, *, final: bool = False) -> None:
        nonlocal last_len
        elapsed = max(time.monotonic() - start, 1e-9)
        rate = index / elapsed
        fraction = index / total if total else 1.0
        width = 28
        filled = min(width, int(math.floor(width * fraction)))
        bar = "#" * filled + "-" * (width - filled)
        text = f"{desc} [{bar}] {index}/{total} {fraction:5.1%} {rate:5.2f}/s elapsed {elapsed:5.1f}s"
        if interactive:
            stream.write("\r" + text + " " * max(0, last_len - len(text)))
            if final:
                stream.write("\n")
            stream.flush()
            last_len = len(text)
        elif final or index == 1 or index % every == 0:
            print(text, file=stream, flush=True)

    for index, item in enumerate(iterable, start=1):
        yield item
        if index == total or index == 1 or index % max(every, 1) == 0:
            write(index, final=index == total)


def _progress_updater(
    *,
    total: int | None,
    desc: str,
    every: int,
    enabled: bool = True,
):
    """Return ``(update, close)`` for frame-level progress."""
    if not enabled:
        return lambda _index, _label="": None, lambda: None

    if total:
        try:
            from tqdm import tqdm
        except ImportError:
            pass
        else:
            bar = tqdm(
                total=total,
                desc=desc,
                unit="frame",
                ncols=96,
                ascii=True,
                dynamic_ncols=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar:14}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            )

            def update_tqdm(_index: int, label: str = "") -> None:
                if label:
                    bar.set_postfix_str(label, refresh=False)
                bar.update(1)

            return update_tqdm, bar.close

    start = time.monotonic()
    stream = sys.stderr
    interactive = hasattr(stream, "isatty") and stream.isatty()
    last_len = 0

    def update_text(index: int, label: str = "") -> None:
        nonlocal last_len
        if index != 1 and index % max(every, 1) != 0 and not (total and index == total):
            return
        elapsed = max(time.monotonic() - start, 1e-9)
        rate = index / elapsed
        if total:
            fraction = index / total
            width = 28
            filled = min(width, int(math.floor(width * fraction)))
            bar = "#" * filled + "-" * (width - filled)
            text = (
                f"{desc} [{bar}] {index}/{total} {fraction:5.1%} "
                f"{rate:5.2f}/s elapsed {elapsed:5.1f}s"
            )
        else:
            text = f"{desc} {index} frames {rate:5.2f}/s elapsed {elapsed:5.1f}s"
        if label:
            text = f"{text} {label}"
        if interactive:
            stream.write("\r" + text + " " * max(0, last_len - len(text)))
            if total and index == total:
                stream.write("\n")
            stream.flush()
            last_len = len(text)
        else:
            print(text, file=stream, flush=True)

    def close_text() -> None:
        if interactive:
            stream.write("\n")
            stream.flush()

    return update_text, close_text


def _cube_paths(cube_glob: str, max_cubes: int | None = None) -> list[Path]:
    pattern = Path(cube_glob)
    if pattern.is_absolute():
        paths = sorted(pattern.parent.glob(pattern.name))
    else:
        paths = sorted(Path().glob(cube_glob))
    if max_cubes is not None:
        paths = paths[:max_cubes]
    return paths


def _parse_frames(value: str | None) -> list[int] | None:
    if value in (None, "", "all"):
        return None
    frames: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(piece.strip()) for piece in part.split("-", maxsplit=1)]
            frames.extend(range(start, end + 1))
        else:
            frames.append(int(part))
    return list(dict.fromkeys(frames))


def _expand_shot_ranges(values: list[object]) -> list[str]:
    shots: list[str] = []
    for value in values:
        if isinstance(value, int):
            shots.append(str(value))
            continue
        text = str(value).strip()
        if "-" in text:
            start, end = [int(piece.strip()) for piece in text.split("-", maxsplit=1)]
            shots.extend(str(shot) for shot in range(start, end + 1))
        elif text:
            shots.append(text)
    return shots


def _iter_shot_groups(groups: dict, prefix: str = "", inherited: dict | None = None):
    inherited = dict(inherited or {})
    for name, payload in groups.items():
        if not isinstance(payload, dict):
            continue
        path = f"{prefix}.{name}" if prefix else name
        metadata = dict(inherited)
        metadata.update(
            {
                key: value
                for key, value in payload.items()
                if key != "shots" and not isinstance(value, dict)
            }
        )
        shots = _expand_shot_ranges(payload.get("shots", []))
        if shots:
            yield path, shots, metadata
        nested = {key: value for key, value in payload.items() if isinstance(value, dict)}
        yield from _iter_shot_groups(nested, path, metadata)


def _load_shot_filter(
    shots_toml: Path | None,
    shot_groups: str | None,
) -> tuple[set[str] | None, dict[str, dict[str, object]]]:
    if not shots_toml or not shots_toml.is_file():
        return None, {}
    with shots_toml.open("rb") as fh:
        raw = tomllib.load(fh)
    requested = {
        group.strip()
        for group in (shot_groups or "").split(",")
        if group.strip()
    }
    selected: set[str] = set()
    metadata_by_shot: dict[str, dict[str, object]] = {}
    for group_path, shots, metadata in _iter_shot_groups(raw.get("groups", {})):
        if requested and group_path not in requested:
            continue
        for shot in shots:
            selected.add(shot)
            shot_metadata = metadata_by_shot.setdefault(shot, {})
            shot_metadata.update(metadata)
            shot_metadata["shot_groups"] = ",".join(
                sorted(
                    set(str(shot_metadata.get("shot_groups", "")).split(","))
                    - {""}
                    | {group_path}
                )
            )
    if requested:
        return selected, metadata_by_shot
    return None, metadata_by_shot


def _cube_frames(cube_path: Path, requested: list[int] | None, *, engine: str | None) -> tuple[str, list[int], dict]:
    ds = xr.load_dataset(cube_path, engine=engine)
    attrs = dict(ds.attrs)
    shot = parse_shot_id(cube_path, attrs) or cube_path.stem.split("_", maxsplit=1)[0]
    if "frame" in ds.sizes:
        available = list(range(int(ds.sizes["frame"])))
    else:
        available = [0]
    ds.close()
    if requested is None:
        return shot, available, attrs
    selected = [frame for frame in requested if frame in available]
    missing = [frame for frame in requested if frame not in available]
    if missing:
        print(f"warning: {cube_path.name} missing requested frame(s): {missing}", file=sys.stderr)
    return shot, selected, attrs


def _load_manifest(path: Path) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = defaultdict(set)
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            selected[str(row["cube"])].add(int(row["frame"]))
    return dict(selected)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scan_threshold_rows(scored: pd.DataFrame) -> list[dict]:
    if scored.empty:
        return []
    rows: list[dict] = []
    for threshold in DEFAULT_SCAN_THRESHOLDS:
        selected = scored[scored["signal_metric"] >= threshold]
        row: dict[str, object] = {
            "signal_threshold": threshold,
            "selected_frames": int(len(selected)),
        }
        if "rax" in selected:
            for rax, count in selected.groupby("rax").size().items():
                label = str(rax).replace(".", "_")
                row[f"rax_{label}_frames"] = int(count)
        rows.append(row)
    return rows


def _suggest_threshold(threshold_rows: list[dict], target: int = 350) -> dict | None:
    candidates = [row for row in threshold_rows if int(row["selected_frames"]) > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(int(row["selected_frames"]) - target))


def _format_temperature(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.0f}K"


def _run_progress_label(stem: str, *, trot1: object = None, trot2: object = None, tvib: object = None) -> str:
    return (
        f"{stem} "
        f"T1={_format_temperature(trot1)} "
        f"T2={_format_temperature(trot2)} "
        f"Tv={_format_temperature(tvib)}"
    )


def _write_scan_report(
    path: Path,
    *,
    scored: pd.DataFrame,
    selected: pd.DataFrame,
    threshold_rows: list[dict],
    suggested: dict | None,
) -> None:
    lines = [
        "# H2 Fulcher Scan Summary",
        "",
        f"Scored frames: {len(scored)}",
        f"Selected frames: {len(selected)}",
        "",
    ]
    if not scored.empty:
        quantiles = scored["signal_metric"].quantile([0.5, 0.75, 0.9, 0.95, 0.98, 0.99])
        lines.extend(["## Signal Metric Quantiles", ""])
        for quantile, value in quantiles.items():
            lines.append(f"- p{quantile * 100:g}: {value:.6g}")
        lines.append("")
    if threshold_rows:
        lines.extend(["## Candidate Gates", "", "| threshold | frames | Rax 3.6 | Rax 3.9 |", "|---:|---:|---:|---:|"])
        for row in threshold_rows:
            lines.append(
                "| "
                f"{float(row['signal_threshold']):.3f} | "
                f"{int(row['selected_frames'])} | "
                f"{int(row.get('rax_3_6_frames', 0))} | "
                f"{int(row.get('rax_3_9_frames', 0))} |"
            )
        lines.append("")
    if suggested:
        threshold = float(suggested["signal_threshold"])
        lines.extend(
            [
                "## Suggested Next Command",
                "",
                "```powershell",
                f"fulcher-h2-dataset --plan h2_dataset_plan.toml scan --signal-threshold {threshold:.3f}",
                "fulcher-h2-dataset --plan h2_dataset_plan.toml run",
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def scan_signal_frames(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cube_paths = _cube_paths(args.cube_glob, args.max_cubes)
    if not cube_paths:
        raise SystemExit(f"No cubes matched {args.cube_glob!r}")
    shot_filter, shot_metadata = _load_shot_filter(args.shots_toml, args.shot_groups)

    rows: list[dict] = []
    start = time.monotonic()
    for cube_index, cube_path in enumerate(
        _progress(
            cube_paths,
            total=len(cube_paths),
            desc="scan cubes",
            every=args.progress_every,
            enabled=not args.no_progress,
        ),
        start=1,
    ):
        ds = xr.load_dataset(cube_path, engine=args.engine)
        attrs = dict(ds.attrs)
        shot = parse_shot_id(cube_path, attrs) or cube_path.stem.split("_", maxsplit=1)[0]
        if shot_filter is not None and shot not in shot_filter:
            ds.close()
            if args.no_progress and (
                cube_index % args.progress_every == 0 or cube_index == len(cube_paths)
            ):
                print(
                    f"scan {cube_index}/{len(cube_paths)} cubes, "
                    f"{len(rows)} frames scored, skipped shot {shot}",
                    flush=True,
                )
            continue
        metadata = shot_metadata.get(shot, {})
        wavelength = ds["wavelength"].values
        window_mask = (wavelength >= args.signal_min_nm) & (wavelength <= args.signal_max_nm)
        intensity = ds["intensity"].sel(wavelength=window_mask)
        if "frame" not in intensity.dims:
            frame_values = [0]
            data = intensity.values[None, :]
        else:
            frame_values = [int(v) for v in intensity["frame"].values]
            data = intensity.transpose("frame", "wavelength").values
        for frame, spectrum in zip(frame_values, data):
            finite = spectrum[np.isfinite(spectrum)]
            if finite.size:
                median = float(pd.Series(finite).median())
                p95 = float(pd.Series(finite).quantile(0.95))
                p99 = float(pd.Series(finite).quantile(0.99))
                maximum = float(finite.max())
                signal_metric = p99 - median
                p95_signal = p95 - median
            else:
                median = p95 = p99 = maximum = signal_metric = p95_signal = float("nan")
            rows.append(
                {
                    "shot": shot,
                    "frame": frame,
                    "cube": str(cube_path),
                    "signal_min_nm": args.signal_min_nm,
                    "signal_max_nm": args.signal_max_nm,
                    "window_median": median,
                    "window_p95": p95,
                    "window_p99": p99,
                    "window_max": maximum,
                    "signal_metric": signal_metric,
                    "p95_signal": p95_signal,
                    "n_finite": int(finite.size),
                    "shot_groups": metadata.get("shot_groups", ""),
                    "rax": metadata.get("rax", ""),
                    "config": metadata.get("config", ""),
                    "label": metadata.get("label", ""),
                }
            )
        ds.close()
        if args.no_progress and (
            cube_index % args.progress_every == 0 or cube_index == len(cube_paths)
        ):
            elapsed = time.monotonic() - start
            print(
                f"scan {cube_index}/{len(cube_paths)} cubes, "
                f"{len(rows)} frames scored, elapsed {elapsed:.1f}s",
                flush=True,
            )

    if rows:
        scored = pd.DataFrame(rows).sort_values("signal_metric", ascending=False)
        scored["signal_rank"] = range(1, len(scored) + 1)
    else:
        scored = pd.DataFrame(
            columns=[
                "shot",
                "frame",
                "cube",
                "signal_min_nm",
                "signal_max_nm",
                "window_median",
                "window_p95",
                "window_p99",
                "window_max",
                "signal_metric",
                "p95_signal",
                "n_finite",
                "shot_groups",
                "rax",
                "config",
                "label",
                "signal_rank",
            ]
        )
    scored_path = output_dir / "frame_signal_scores.csv"
    scored.to_csv(scored_path, index=False)

    threshold_rows = _scan_threshold_rows(scored)
    threshold_path = output_dir / "threshold_summary.csv"
    _write_csv(threshold_path, threshold_rows)
    suggested = _suggest_threshold(threshold_rows)

    selected = scored.copy()
    if args.signal_threshold is None and args.select_top is None:
        selected = scored.head(0)
    if args.signal_threshold is not None:
        selected = selected[selected["signal_metric"] >= args.signal_threshold]
    if args.select_top is not None:
        selected = selected.head(args.select_top)
    selected_path = output_dir / "selected_frames.csv"
    selected.to_csv(selected_path, index=False)
    report_path = output_dir / "scan_summary.md"
    _write_scan_report(
        report_path,
        scored=scored,
        selected=selected,
        threshold_rows=threshold_rows,
        suggested=suggested,
    )

    print(f"signal scores: {scored_path}")
    print(f"threshold summary: {threshold_path}")
    print(f"scan summary: {report_path}")
    print(f"selected frames: {selected_path}")
    print(f"scored frames: {len(scored)}")
    print(f"selected frames: {len(selected)}")
    if args.signal_threshold is None and args.select_top is None:
        print("selection gate: none; choose a gate below, then rerun scan with --signal-threshold or --select-top")
    if args.shot_groups:
        print(f"shot groups: {args.shot_groups}")
    if threshold_rows:
        print("candidate gates:")
        for row in threshold_rows:
            print(
                "  "
                f"{float(row['signal_threshold']):.3f}: "
                f"{int(row['selected_frames'])} frames "
                f"(Rax 3.6: {int(row.get('rax_3_6_frames', 0))}, "
                f"Rax 3.9: {int(row.get('rax_3_9_frames', 0))})"
            )
    if args.signal_threshold is None and args.select_top is None and suggested:
        threshold = float(suggested["signal_threshold"])
        print(
            "suggested first pass: "
            f"fulcher-h2-dataset --plan h2_dataset_plan.toml scan --signal-threshold {threshold:.3f}"
        )
    if not selected.empty:
        print(
            "selected signal range: "
            f"{selected['signal_metric'].min():.6g} .. {selected['signal_metric'].max():.6g}"
        )


def _fit_config() -> FitConfig:
    return FitConfig(
        line_left_width_nm=0.24,
        line_right_width_nm=0.18,
        center_left_offset_nm=0.16,
        center_right_offset_nm=0.08,
        instrument_sigma_nm=0.0273,
        instrument_sigma_leeway_nm=0.015,
        close_neighbor_threshold_nm=0.15,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=None, help="TOML run plan with [common], [scan], and [run] sections.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["scan", "run"],
        default="run",
        help="scan ranks signal frames; run processes selected frames.",
    )
    parser.add_argument("--cube-glob", default=DEFAULT_CUBE_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", default="all", help="all, a comma list, or ranges like 0-10,20")
    parser.add_argument("--manifest", type=Path, default=None, help="CSV from scan, usually selected_frames.csv.")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--max-cubes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-fit-relerr", type=float, default=1.0)
    parser.add_argument("--shots-toml", type=Path, default=DEFAULT_SHOTS_TOML)
    parser.add_argument("--shot-groups", default="", help="scan: comma-separated groups from shots.toml, e.g. rax36,rax39.")
    parser.add_argument("--select-top", type=int, default=None, help="scan: optional top N frames by signal metric.")
    parser.add_argument("--signal-threshold", type=float, default=None, help="scan: optional minimum signal metric.")
    parser.add_argument("--signal-min-nm", type=float, default=DEFAULT_SIGNAL_MIN_NM)
    parser.add_argument("--signal-max-nm", type=float, default=DEFAULT_SIGNAL_MAX_NM)
    parser.add_argument("--progress-every", type=int, default=5, help="Print progress every N cubes/frames.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and use periodic log lines.")
    parser.add_argument("--show-model-output", action="store_true", help="Show verbose per-frame output from downstream model code.")
    parser.add_argument("--qc-every", type=int, default=0, help="Write extraction/Boltzmann plots every N frames; 0 disables.")
    parser.add_argument("--line-fit-qc", action="store_true", help="Also write per-frame line-fit PDFs when --qc-every selects a frame.")
    args = parser.parse_args()
    provided = _provided_destinations(parser, sys.argv[1:])
    if args.plan:
        _apply_plan(args, args.plan, provided)

    if args.command == "scan":
        scan_signal_frames(args)
        return

    output_dir = args.output_dir
    intensity_dir = output_dir / "intensities"
    fit_report_dir = output_dir / "fit_reports"
    qc_region_dir = output_dir / "plots" / "extraction_region"
    qc_line_dir = output_dir / "plots" / "extraction_lines"
    boltzmann_plot_dir = output_dir / "plots" / "boltzmann"
    tables_dir = output_dir / "tables"
    for path in (intensity_dir, fit_report_dir, qc_region_dir, qc_line_dir, boltzmann_plot_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_frames = _load_manifest(args.manifest) if args.manifest else None
    requested_frames = _parse_frames(args.frames)
    cube_paths = _cube_paths(args.cube_glob, args.max_cubes)
    if not cube_paths:
        raise SystemExit(f"No cubes matched {args.cube_glob!r}")

    lines = load_lines()
    policy_set = load_line_policy_set()
    label_lines = overview_qc_lines(lines, policy_set=policy_set)
    config = _fit_config()

    run_rows: list[dict] = []
    boltzmann_rows: list[dict] = []
    coronal_rows: list[dict] = []
    calibration_rows: list[dict] = []
    policy_rows: list[dict] = []
    processed = 0
    start = time.monotonic()

    total_frames = 0
    manifest_by_cube = manifest_frames or {}
    if manifest_frames is not None:
        total_frames = sum(len(frames) for frames in manifest_by_cube.values())
    elif requested_frames is not None:
        total_frames = len(cube_paths) * len(requested_frames)
    if args.max_frames is not None and total_frames:
        total_frames = min(total_frames, args.max_frames)
    update_progress, close_progress = _progress_updater(
        total=total_frames or None,
        desc="run frames",
        every=args.progress_every,
        enabled=not args.no_progress,
    )

    try:
        for cube_index, cube_path in enumerate(cube_paths, start=1):
            shot, frames, cube_attrs = _cube_frames(cube_path, requested_frames, engine=args.engine)
            if manifest_frames is not None:
                frames = sorted(manifest_frames.get(str(cube_path), set()))
                if not frames:
                    continue
            calibration_rows.append(
                {
                    "shot": shot,
                    "cube": str(cube_path),
                    "wavelength_calibration_file": cube_attrs.get("wavelength_calibration_file", ""),
                    "calibration_order_pattern_file": cube_attrs.get("calibration_order_pattern_file", ""),
                    "calibration_file_digests_json": cube_attrs.get("calibration_file_digests_json", ""),
                    "order_border_pixel_ranges_json": cube_attrs.get("order_border_pixel_ranges_json", ""),
                    "order_wavelength_ranges_nm_json": cube_attrs.get("order_wavelength_ranges_nm_json", ""),
                }
            )
            for frame in frames:
                if args.max_frames is not None and processed >= args.max_frames:
                    break
                processed += 1
                stem = f"{shot}_fr_{frame}"
                if args.no_progress and (
                    processed == 1 or processed % args.progress_every == 0
                ):
                    elapsed = time.monotonic() - start
                    print(
                        f"run frame {processed}: cube {cube_index}/{len(cube_paths)} "
                        f"{stem}, elapsed {elapsed:.1f}s",
                        flush=True,
                )
                row = {"shot": shot, "frame": frame, "cube": str(cube_path), "status": "ok"}
                progress_label = stem
                model_stdout = io.StringIO()
                stdout_context = (
                    contextlib.nullcontext()
                    if args.show_model_output
                    else contextlib.redirect_stdout(model_stdout)
                )
                try:
                    with stdout_context:
                        spectrum = load_spectrum(cube_path, frame=frame, engine=args.engine)
                        results = extract_lines(spectrum, lines=lines, config=config)
                        metadata = {
                            "policy_layer": "line_policies.toml",
                            "source_cube": str(cube_path),
                            "wavelength_calibration_file": spectrum.metadata.get("wavelength_calibration_file", ""),
                            "calibration_file_digests_json": spectrum.metadata.get("calibration_file_digests_json", ""),
                        }
                        _, _, fit_report_path = write_fulcheranalyzer_csvs(
                            results,
                            output_dir=intensity_dir,
                            shot=shot,
                            frame=frame,
                            metadata=metadata,
                        )
                        fit_table = results_to_dataframe(results)
                        fit_report_copy = fit_report_dir / fit_report_path.name
                        fit_table.to_csv(fit_report_copy, index=False)
        
                        summary = (
                            fit_table.groupby(["legacy_policy", "legacy_matrix_action"], dropna=False)
                            .size()
                            .rename("n_lines")
                            .reset_index()
                        )
                        for _, policy_row in summary.iterrows():
                            policy_rows.append(
                                {
                                    "shot": shot,
                                    "frame": frame,
                                    "legacy_policy": policy_row["legacy_policy"],
                                    "legacy_matrix_action": policy_row["legacy_matrix_action"],
                                    "n_lines": int(policy_row["n_lines"]),
                                }
                            )
        
                        intensities = read_intensities(shot, frame, data_folder=intensity_dir)
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
                        points.to_csv(tables_dir / f"{stem}_boltzmann_qc_points.csv", index=False)
                        boltzmann_rows.append(
                            {
                                "shot": shot,
                                "frame": frame,
                                "alpha": bp.alpha,
                                "beta": bp.beta,
                                "Trot1": bp.trot1,
                                "Trot2": bp.trot2,
                                "alpha_stderr": bp.err[0],
                                "beta_stderr": bp.err[1],
                                "Trot1_stderr": bp.err[2],
                                "Trot2_stderr": bp.err[3],
                                "n_boltzmann_points": int(points["fit_mask"].sum()) if "fit_mask" in points else "",
                                "status": "ok",
                            }
                        )
        
                        cm = CoronaModel(bp)
                        cm.coronal_autofit()
                        progress_label = _run_progress_label(
                            stem,
                            trot1=bp.trot1,
                            trot2=bp.trot2,
                            tvib=cm.tvib,
                        )
                        coronal_rows.append(
                            {
                                "shot": shot,
                                "frame": frame,
                                "Tvib": cm.tvib,
                                "Tvib_stderr": cm.tviberr,
                                "status": "ok",
                            }
                        )
        
                        if args.qc_every and processed % args.qc_every == 0:
                            fig = plot_region(
                                spectrum,
                                lines=lines,
                                label_lines=label_lines,
                                guide_lines=label_lines,
                                output_path=qc_region_dir / f"{stem}_600_630.png",
                            )
                            plt.close(fig)
                            fig = plot_boltzmann_qc(
                                bp,
                                points,
                                title=f"H2 {shot} frame {frame}: d-state Boltzmann fit",
                            )
                            fig.savefig(boltzmann_plot_dir / f"{stem}_boltzmann_qc.png", dpi=180)
                            plt.close(fig)
                            if args.line_fit_qc:
                                write_line_fit_qc(
                                    spectrum,
                                    results,
                                    pdf_path=qc_line_dir / f"{stem}_line_fits.pdf",
                                    columns=5,
                                )
                        row["n_lines"] = len(results)
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                    row["traceback"] = traceback.format_exc()
                    if not args.show_model_output and model_stdout.getvalue():
                        row["captured_stdout"] = model_stdout.getvalue()
                    progress_label = f"{stem} failed"
                    boltzmann_rows.append({"shot": shot, "frame": frame, "status": "failed", "error": repr(exc)})
                    coronal_rows.append({"shot": shot, "frame": frame, "status": "failed", "error": repr(exc)})
                    print(f"failed {stem}: {exc!r}", file=sys.stderr, flush=True)
                run_rows.append(row)
                update_progress(processed, progress_label)
            if args.max_frames is not None and processed >= args.max_frames:
                break
    finally:
        close_progress()

    _write_csv(output_dir / "run_summary.csv", run_rows)
    _write_csv(output_dir / "boltzmann_summary.csv", boltzmann_rows)
    _write_csv(output_dir / "coronal_summary.csv", coronal_rows)
    _write_csv(output_dir / "policy_summary.csv", policy_rows)
    _write_csv(output_dir / "calibration_manifest.csv", calibration_rows)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# H2 Fulcher Dataset Run",
                "",
                f"Cube glob: `{args.cube_glob}`",
                f"Plan: `{args.plan or ''}`",
                f"Manifest: `{args.manifest or ''}`",
                f"Frames: `{args.frames}`",
                f"Processed frames: `{processed}`",
                "",
                "Primary outputs:",
                "",
                "- `intensities/`: fulcheranalyzer-compatible intensity/error matrices",
                "- `fit_reports/`: long-form extraction audit tables",
                "- `boltzmann_summary.csv`: per-frame two-temperature fit summary",
                "- `coronal_summary.csv`: per-frame coronal Tvib summary",
                "- `calibration_manifest.csv`: cube calibration metadata and digests",
                "- `policy_summary.csv`: per-frame extraction policy counts",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"processed frames: {processed}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
