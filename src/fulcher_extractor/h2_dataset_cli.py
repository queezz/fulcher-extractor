"""Run H2 Fulcher dataset scanning and SpectroCube intensity extraction."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

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

from fulcher_extractor.extract import extract_lines
from fulcher_extractor.fit import FitConfig
from fulcher_extractor.line_database import load_lines
from fulcher_extractor.line_policy import load_line_policy_set, overview_qc_lines
from fulcher_extractor.output import results_to_dataframe, write_fulcheranalyzer_csvs
from fulcher_extractor.qc import plot_line_fit_page, plot_region
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
DEFAULT_FULCHER_MIN_NM = 600.0
DEFAULT_FULCHER_MAX_NM = 630.0
DEFAULT_HALPHA_MIN_NM = 655.5
DEFAULT_HALPHA_MAX_NM = 657.1
HALPHA_SCAN_WEIGHT = 0.25
PLAN_PATH_KEYS = {"output_dir", "manifest", "shots_toml"}
PLAN_LIST_AS_CSV_KEYS = {"shot_groups"}
DEFAULT_SCAN_QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99)


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    setting = os.environ.get("FULCHER_COLOR", "").lower()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False
    return sys.stdout.isatty()


USE_COLOR = _color_enabled()


def _c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _c(text, "1")


def _dim(text: str) -> str:
    return _c(text, "2")


def _green(text: str) -> str:
    return _c(text, "32")


def _cyan(text: str) -> str:
    return _c(text, "36")


def _yellow(text: str) -> str:
    return _c(text, "33")


def _red(text: str) -> str:
    return _c(text, "31")


def _provided_destinations(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    provided: set[str] = set()
    actions = [action for action in parser._actions if action.dest != argparse.SUPPRESS]
    for token in argv:
        if token in {"scan", "extract"}:
            provided.add("command")
        for action in actions:
            for option in action.option_strings:
                if token == option or token.startswith(f"{option}="):
                    provided.add(action.dest)
    return provided


def _plan_path(value: object, plan_path: Path) -> Path:
    path = Path(str(value)).expanduser()
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


def _relative_display_path(path: Path, base: Path) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_base = base.expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_base).as_posix()
    except ValueError:
        return str(resolved_path)


def _print_artifact_summary(
    *,
    title: str,
    output_dir: Path,
    artifacts: list[tuple[str, Path]],
    workdir: Path | None = None,
) -> None:
    workdir = workdir or Path.cwd()
    print()
    print(_bold(f"=== {title} ==="))
    print(f"🏠 workdir   : {_dim(str(workdir.resolve()))}")
    print(f"📁 output   : {_cyan(_relative_display_path(output_dir, workdir))}")
    print("📄 artifacts:")
    for label, path in artifacts:
        print(f"  {_green('WRITE')} {label:<12} {_relative_display_path(path, output_dir)}")


def _print_metric(label: str, value: object, *, icon: str = "•", color=None) -> None:
    style = color or (lambda text: text)
    print(f"{icon} {label:<17} {style(str(value))}")


def _empty_window_stats(prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_median": float("nan"),
        f"{prefix}_p95": float("nan"),
        f"{prefix}_p99": float("nan"),
        f"{prefix}_max": float("nan"),
        f"{prefix}_signal": float("nan"),
        f"{prefix}_p95_signal": float("nan"),
        f"{prefix}_peak_signal": float("nan"),
        f"{prefix}_peak_count": 0,
        f"{prefix}_n_finite": 0,
    }


def _window_stats(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    *,
    min_nm: float,
    max_nm: float,
    prefix: str,
) -> dict[str, object]:
    mask = (wavelength >= min_nm) & (wavelength <= max_nm)
    x = wavelength[mask]
    y = spectrum[mask]
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not finite_mask.any():
        return _empty_window_stats(prefix)

    finite = y[finite_mask]
    median = float(np.median(finite))
    p95 = float(np.quantile(finite, 0.95))
    p99 = float(np.quantile(finite, 0.99))
    maximum = float(finite.max())
    signal = p99 - median
    p95_signal = p95 - median

    peak_signal = float("nan")
    peak_count = 0
    if finite.size >= 3:
        centered = finite - median
        local_maxima = centered[1:-1][
            (finite[1:-1] > finite[:-2])
            & (finite[1:-1] >= finite[2:])
            & (centered[1:-1] > 0)
        ]
        mad = float(np.median(np.abs(finite - median)))
        noise_floor = 3.0 * 1.4826 * mad
        if local_maxima.size:
            peak_signal = float(local_maxima.max())
            if noise_floor > 0:
                peak_count = int(np.count_nonzero(local_maxima >= noise_floor))
            else:
                peak_count = int(local_maxima.size)

    return {
        f"{prefix}_median": median,
        f"{prefix}_p95": p95,
        f"{prefix}_p99": p99,
        f"{prefix}_max": maximum,
        f"{prefix}_signal": signal,
        f"{prefix}_p95_signal": p95_signal,
        f"{prefix}_peak_signal": peak_signal,
        f"{prefix}_peak_count": peak_count,
        f"{prefix}_n_finite": int(finite.size),
    }


def _nanmax(*values: object) -> float:
    finite = [float(value) for value in values if pd.notna(value) and np.isfinite(float(value))]
    if not finite:
        return float("nan")
    return max(finite)


def _score_scan_frame(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    *,
    fulcher_min_nm: float,
    fulcher_max_nm: float,
    halpha_min_nm: float = DEFAULT_HALPHA_MIN_NM,
    halpha_max_nm: float = DEFAULT_HALPHA_MAX_NM,
) -> dict[str, object]:
    fulcher = _window_stats(
        wavelength,
        spectrum,
        min_nm=fulcher_min_nm,
        max_nm=fulcher_max_nm,
        prefix="fulcher",
    )
    halpha = _window_stats(
        wavelength,
        spectrum,
        min_nm=halpha_min_nm,
        max_nm=halpha_max_nm,
        prefix="halpha",
    )
    fulcher_window_signal = float(fulcher["fulcher_signal"])
    fulcher_peak_signal = float(fulcher["fulcher_peak_signal"])
    halpha_signal = float(halpha["halpha_signal"])

    fulcher_evidence = _nanmax(fulcher_window_signal, fulcher_peak_signal)
    halpha_support = np.isfinite(halpha_signal) and halpha_signal > 0
    halpha_boost = (
        HALPHA_SCAN_WEIGHT * fulcher_evidence
        if halpha_support and np.isfinite(fulcher_evidence) and fulcher_evidence > 0
        else 0.0
    )
    scan_score = fulcher_evidence + halpha_boost if np.isfinite(fulcher_evidence) else halpha_boost

    reason = "no_signal"
    if int(fulcher["fulcher_peak_count"]) > 0:
        reason = "fulcher_peaks"
    elif np.isfinite(fulcher_window_signal) and fulcher_window_signal > 0:
        reason = "fulcher_window"
    elif halpha_support:
        reason = "halpha_only"
    if halpha_boost > 0 and reason.startswith("fulcher"):
        reason = f"{reason}+halpha"

    return {
        "scan_score": scan_score,
        "scan_score_reason": reason,
        "fulcher_evidence": fulcher_evidence,
        "halpha_boost": halpha_boost,
        "fulcher_min_nm": fulcher_min_nm,
        "fulcher_max_nm": fulcher_max_nm,
        "halpha_min_nm": halpha_min_nm,
        "halpha_max_nm": halpha_max_nm,
        **fulcher,
        **halpha,
    }


def _scan_thresholds(scored: pd.DataFrame) -> list[float]:
    if scored.empty:
        return []
    finite = scored["scan_score"].replace([np.inf, -np.inf], np.nan).dropna()
    finite = finite[finite > 0]
    if finite.empty:
        return []
    thresholds = sorted({float(value) for value in finite.quantile(DEFAULT_SCAN_QUANTILES)})
    if len(thresholds) == 1:
        return thresholds
    cleaned: list[float] = []
    for threshold in thresholds:
        if not cleaned or not math.isclose(threshold, cleaned[-1], rel_tol=1e-9, abs_tol=1e-12):
            cleaned.append(threshold)
    return cleaned


def _scan_threshold_rows(scored: pd.DataFrame) -> list[dict]:
    if scored.empty:
        return []
    rows: list[dict] = []
    for threshold in _scan_thresholds(scored):
        selected = scored[scored["scan_score"] >= threshold]
        fulcher_threshold = threshold / (1.0 + HALPHA_SCAN_WEIGHT)
        row: dict[str, object] = {
            "scan_threshold": threshold,
            "fulcher_threshold": fulcher_threshold,
            "scan_quantile": float((scored["scan_score"] <= threshold).mean()),
            "gate_metric": "scan_score",
            "selected_frames": int(len(selected)),
            "fulcher_evidence_frames": int((selected["fulcher_evidence"] >= fulcher_threshold).sum()),
            "fulcher_peak_frames": int((selected["fulcher_peak_signal"] >= fulcher_threshold).sum()),
            "halpha_boost_frames": int((selected["halpha_boost"] >= threshold).sum()),
            "halpha_support_frames": int((selected["halpha_boost"] > 0).sum()),
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
        "## What The Score Means",
        "",
        "`scan_score` is a ranking score in the cube intensity units, not a calibrated plasma quantity.",
        "",
        "```text",
        "fulcher_evidence = max(fulcher_signal, fulcher_peak_signal)",
        f"scan_score = fulcher_evidence + {HALPHA_SCAN_WEIGHT:g} * fulcher_evidence when H-alpha is present",
        "```",
        "",
        "- `fulcher_signal`: p99 minus median intensity in the Fulcher review window.",
        "- `fulcher_peak_signal`: strongest local peak above the Fulcher-window median.",
        "- `halpha_signal`: p99 minus median intensity around H-alpha.",
        "- H-alpha is a support flag: it can boost Fulcher evidence, but it cannot dominate the score by itself.",
        "",
    ]
    if not scored.empty:
        quantiles = scored["scan_score"].quantile([0.5, 0.75, 0.9, 0.95, 0.98, 0.99])
        lines.extend(["## Scan Score Quantiles", ""])
        for quantile, value in quantiles.items():
            lines.append(f"- p{quantile * 100:g}: {value:.6g}")
        lines.append("")
    if threshold_rows:
        lines.extend(
            [
                "## Candidate Gates",
                "",
                "| quantile | scan score | frames | Fulcher evidence | Fulcher peaks | H-alpha support | Rax 3.6 | Rax 3.9 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in threshold_rows:
            lines.append(
                "| "
                f"p{100 * float(row['scan_quantile']):.0f} | "
                f"{float(row['scan_threshold']):.3f} | "
                f"{int(row['selected_frames'])} | "
                f"{int(row.get('fulcher_evidence_frames', 0))} | "
                f"{int(row.get('fulcher_peak_frames', 0))} | "
                f"{int(row.get('halpha_support_frames', 0))} | "
                f"{int(row.get('rax_3_6_frames', 0))} | "
                f"{int(row.get('rax_3_9_frames', 0))} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Scan Plot",
            "",
            "Open `scan_gate_summary.png` for the score distribution and candidate-gate curves.",
            "",
        ]
    )
    if suggested:
        threshold = float(suggested["scan_threshold"])
        lines.extend(
            [
                "## Suggested Next Command",
                "",
                "```powershell",
                f"fulcher-h2-dataset --plan h2_dataset_plan.toml scan --scan-threshold {threshold:.3f}",
                "fulcher-h2-dataset --plan h2_dataset_plan.toml extract",
                "fulcher-analyze-batch --plan h2_dataset_plan.toml",
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_scan_plot(
    path: Path,
    *,
    scored: pd.DataFrame,
    threshold_rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axes[0]
    if scored.empty:
        ax.text(0.5, 0.5, "No scored frames", ha="center", va="center", transform=ax.transAxes)
    else:
        by_rax = scored.groupby("rax", dropna=False) if "rax" in scored else [(None, scored)]
        for rax, group in by_rax:
            label = f"Rax {rax}" if str(rax).strip() else "unlabelled"
            ax.hist(
                group["scan_score"].dropna(),
                bins=40,
                histtype="step",
                linewidth=1.4,
                label=label,
            )
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Scan score distribution", loc="left")
    ax.set_xlabel("scan score")
    ax.set_ylabel("frames")

    ax = axes[1]
    if threshold_rows:
        thresholds = [float(row["scan_threshold"]) for row in threshold_rows]
        selected = [int(row["selected_frames"]) for row in threshold_rows]
        rax36 = [int(row.get("rax_3_6_frames", 0)) for row in threshold_rows]
        rax39 = [int(row.get("rax_3_9_frames", 0)) for row in threshold_rows]
        ax.plot(thresholds, selected, marker="o", label="selected")
        ax.plot(thresholds, rax36, marker="s", label="Rax 3.6")
        ax.plot(thresholds, rax39, marker="^", label="Rax 3.9")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No candidate gates", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Candidate gates by Rax", loc="left")
    ax.set_xlabel("scan threshold")
    ax.set_ylabel("frames")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def scan_frames(args: argparse.Namespace) -> None:
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
        intensity = ds["intensity"]
        if "frame" not in intensity.dims:
            frame_values = [0]
            data = intensity.values[None, :]
        else:
            frame_values = [int(v) for v in intensity["frame"].values]
            data = intensity.transpose("frame", "wavelength").values
        for frame, spectrum in zip(frame_values, data):
            score = _score_scan_frame(
                wavelength,
                spectrum,
                fulcher_min_nm=args.fulcher_min_nm,
                fulcher_max_nm=args.fulcher_max_nm,
            )
            rows.append(
                {
                    "shot": shot,
                    "frame": frame,
                    "cube": str(cube_path),
                    **score,
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
        scored = pd.DataFrame(rows).sort_values("scan_score", ascending=False)
        scored["scan_rank"] = range(1, len(scored) + 1)
    else:
        scored = pd.DataFrame(
            columns=[
                "shot",
                "frame",
                "cube",
                "scan_score",
                "scan_score_reason",
                "fulcher_min_nm",
                "fulcher_max_nm",
                "fulcher_signal",
                "fulcher_evidence",
                "fulcher_p95_signal",
                "fulcher_median",
                "fulcher_p95",
                "fulcher_p99",
                "fulcher_max",
                "fulcher_n_finite",
                "fulcher_peak_signal",
                "fulcher_peak_count",
                "halpha_min_nm",
                "halpha_max_nm",
                "halpha_signal",
                "halpha_boost",
                "halpha_p95_signal",
                "halpha_median",
                "halpha_p95",
                "halpha_p99",
                "halpha_max",
                "halpha_n_finite",
                "halpha_peak_signal",
                "halpha_peak_count",
                "shot_groups",
                "rax",
                "config",
                "label",
                "scan_rank",
            ]
        )
    scored_path = output_dir / "frame_scan_scores.csv"
    scored.to_csv(scored_path, index=False)

    threshold_rows = _scan_threshold_rows(scored)
    threshold_path = output_dir / "threshold_summary.csv"
    _write_csv(threshold_path, threshold_rows)
    suggested = _suggest_threshold(threshold_rows)

    selected = scored.copy()
    if args.scan_threshold is None and args.select_top is None:
        selected = scored.head(0)
    if args.scan_threshold is not None:
        selected = selected[selected["scan_score"] >= args.scan_threshold]
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
    plot_path = output_dir / "scan_gate_summary.png"
    _write_scan_plot(plot_path, scored=scored, threshold_rows=threshold_rows)

    _print_artifact_summary(
        title="H2 Fulcher scan",
        output_dir=output_dir,
        artifacts=[
            ("scan scores", scored_path),
            ("thresholds", threshold_path),
            ("summary", report_path),
            ("plot", plot_path),
            ("selected", selected_path),
        ],
    )
    _print_metric("scored frames", len(scored), icon="🧮", color=_cyan)
    selected_color = _green if len(selected) else _yellow
    _print_metric("selected frames", len(selected), icon="✅", color=selected_color)
    if args.scan_threshold is None and args.select_top is None:
        print(_yellow("🧪 selection gate: none; choose a gate below, then rerun scan with --scan-threshold or --select-top"))
    if args.shot_groups:
        _print_metric("shot groups", args.shot_groups, icon="🎯", color=_cyan)
    if threshold_rows:
        print()
        print(_bold("📊 Candidate gates"))
        for row in threshold_rows:
            threshold_text = _cyan(f"{float(row['scan_threshold']):.3f}")
            print(
                "  "
                f"{threshold_text}  "
                f"{int(row['selected_frames']):>4} frames  "
                f"{_dim('Rax 3.6:')} {int(row.get('rax_3_6_frames', 0)):>3}  "
                f"{_dim('Rax 3.9:')} {int(row.get('rax_3_9_frames', 0)):>3}  "
                f"{_dim('Fulcher peaks:')} {int(row.get('fulcher_peak_frames', 0)):>4}"
            )
    if args.scan_threshold is None and args.select_top is None and suggested:
        threshold = float(suggested["scan_threshold"])
        print()
        print(_green("▶ suggested first pass"))
        print(f"  fulcher-h2-dataset --plan h2_dataset_plan.toml scan --scan-threshold {threshold:.3f}")
    if not selected.empty:
        score_range = f"{selected['scan_score'].min():.6g} .. {selected['scan_score'].max():.6g}"
        _print_metric("score range", score_range, icon="📈", color=_cyan)


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


def _extract_frame_task(task: dict) -> dict:
    cube_path = Path(task["cube_path"])
    shot = task["shot"]
    frame = int(task["frame"])
    stem = task["stem"]
    intensity_dir = Path(task["intensity_dir"])
    fit_report_dir = Path(task["fit_report_dir"])
    qc_region_dir = Path(task["qc_region_dir"])
    qc_line_dir = Path(task["qc_line_dir"])
    row = {"shot": shot, "frame": frame, "cube": str(cube_path), "status": "ok"}
    policy_rows: list[dict] = []
    progress_label = stem
    try:
        lines = load_lines()
        config = _fit_config()
        spectrum = load_spectrum(cube_path, frame=frame, engine=task["engine"])
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

        if task["write_region_qc"]:
            policy_set = load_line_policy_set()
            label_lines = overview_qc_lines(lines, policy_set=policy_set)
            fig = plot_region(
                spectrum,
                lines=lines,
                label_lines=label_lines,
                guide_lines=label_lines,
                output_path=qc_region_dir / f"{stem}_600_630.png",
            )
            plt.close(fig)
        if task["write_line_fit_qc"]:
            fig = plot_line_fit_page(spectrum, results, columns=5)
            if task["parallel_line_fit_qc"]:
                with PdfPages(qc_line_dir / f"{stem}_line_fits.pdf") as pdf:
                    pdf.savefig(fig)
            else:
                line_fit_pdf = task["line_fit_pdf"]
                if line_fit_pdf is not None:
                    line_fit_pdf.savefig(fig)
            plt.close(fig)
        row["n_lines"] = len(results)
        progress_label = f"{stem} lines={len(results)}"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc()
        progress_label = f"{stem} failed"
    return {
        "ordinal": int(task["ordinal"]),
        "row": row,
        "policy_rows": policy_rows,
        "progress_label": progress_label,
    }


def _worker_count(value: int | None) -> int:
    if value is None:
        return 1
    if value < 1:
        cpu_count = os.cpu_count() or 1
        return max(cpu_count - 1, 1)
    return value


def _stop_executor(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is not None:
        terminate_workers()
        return
    executor.shutdown(wait=False, cancel_futures=True)


def extract_dataset(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    intensity_dir = output_dir / "intensities"
    fit_report_dir = output_dir / "fit_reports"
    qc_region_dir = output_dir / "plots" / "extraction_region"
    qc_line_dir = output_dir / "plots" / "extraction_lines"
    for path in (intensity_dir, fit_report_dir, qc_region_dir, qc_line_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_frames = _load_manifest(args.manifest) if args.manifest else None
    requested_frames = _parse_frames(args.frames)
    cube_paths = _cube_paths(args.cube_glob, args.max_cubes)
    if not cube_paths:
        raise SystemExit(f"No cubes matched {args.cube_glob!r}")

    extraction_rows: list[dict] = []
    calibration_rows: list[dict] = []
    policy_rows: list[dict] = []
    start = time.monotonic()

    tasks: list[dict] = []
    manifest_by_cube = manifest_frames or {}
    for cube_index, cube_path in enumerate(cube_paths, start=1):
        shot, frames, cube_attrs = _cube_frames(cube_path, requested_frames, engine=args.engine)
        if manifest_frames is not None:
            frames = sorted(manifest_by_cube.get(str(cube_path), set()))
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
            if args.max_frames is not None and len(tasks) >= args.max_frames:
                break
            ordinal = len(tasks) + 1
            stem = f"{shot}_fr_{frame}"
            tasks.append(
                {
                    "ordinal": ordinal,
                    "cube_index": cube_index,
                    "cube_count": len(cube_paths),
                    "cube_path": str(cube_path),
                    "shot": shot,
                    "frame": frame,
                    "stem": stem,
                    "engine": args.engine,
                    "intensity_dir": str(intensity_dir),
                    "fit_report_dir": str(fit_report_dir),
                    "qc_region_dir": str(qc_region_dir),
                    "qc_line_dir": str(qc_line_dir),
                    "write_region_qc": bool(args.qc_every and ordinal % args.qc_every == 0),
                    "write_line_fit_qc": False,
                    "parallel_line_fit_qc": False,
                    "line_fit_pdf": None,
                }
            )
        if args.max_frames is not None and len(tasks) >= args.max_frames:
            break

    total_frames = len(tasks)
    update_progress, close_progress = _progress_updater(
        total=total_frames,
        desc="extract frames",
        every=args.progress_every,
        enabled=not args.no_progress,
    )
    workers = _worker_count(args.workers)
    if args.line_fit_qc and workers > 1:
        for task in tasks:
            task["write_line_fit_qc"] = True
            task["parallel_line_fit_qc"] = True

    results: list[dict] = []
    interrupted = False
    try:
        if workers == 1:
            open_pdfs: dict[str, PdfPages] = {}
            try:
                for task in tasks:
                    if args.line_fit_qc:
                        pdf_path = qc_line_dir / f"{Path(task['cube_path']).stem}_line_fits.pdf"
                        pdf_key = str(pdf_path)
                        open_pdfs.setdefault(pdf_key, PdfPages(pdf_path))
                        task["write_line_fit_qc"] = True
                        task["line_fit_pdf"] = open_pdfs[pdf_key]
                    result = _extract_frame_task(task)
                    results.append(result)
                    if args.no_progress and (
                        result["ordinal"] == 1 or result["ordinal"] % args.progress_every == 0
                    ):
                        elapsed = time.monotonic() - start
                        print(
                            f"extract frame {result['ordinal']}: "
                            f"{result['progress_label']}, elapsed {elapsed:.1f}s",
                            flush=True,
                        )
                    update_progress(result["ordinal"], result["progress_label"])
            finally:
                for pdf in open_pdfs.values():
                    pdf.close()
        else:
            print(f"extract workers: {workers}", file=sys.stderr, flush=True)
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            try:
                future_to_task = {executor.submit(_extract_frame_task, task): task for task in tasks}
                completed = 0
                for future in concurrent.futures.as_completed(future_to_task):
                    result = future.result()
                    completed += 1
                    results.append(result)
                    if result["row"].get("status") == "failed":
                        print(
                            f"failed {result['row']['shot']}_fr_{result['row']['frame']}: "
                            f"{result['row'].get('error', '')}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if args.no_progress and (
                        completed == 1 or completed % args.progress_every == 0
                    ):
                        elapsed = time.monotonic() - start
                        print(
                            f"extract frame {completed}/{total_frames}: "
                            f"{result['progress_label']}, elapsed {elapsed:.1f}s",
                            flush=True,
                        )
                    update_progress(completed, result["progress_label"])
            except KeyboardInterrupt:
                interrupted = True
                for future in future_to_task:
                    future.cancel()
                _stop_executor(executor)
            else:
                executor.shutdown()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        close_progress()
    if interrupted:
        print(
            "Extraction interrupted by Ctrl+C; writing summaries for completed frames only.",
            file=sys.stderr,
            flush=True,
        )
    for result in sorted(results, key=lambda item: item["ordinal"]):
        extraction_rows.append(result["row"])
        policy_rows.extend(result["policy_rows"])

    extraction_summary_path = output_dir / "extraction_summary.csv"
    policy_summary_path = output_dir / "policy_summary.csv"
    calibration_manifest_path = output_dir / "calibration_manifest.csv"
    readme_path = output_dir / "README.md"
    _write_csv(extraction_summary_path, extraction_rows)
    _write_csv(policy_summary_path, policy_rows)
    _write_csv(calibration_manifest_path, calibration_rows)
    readme_path.write_text(
        "\n".join(
            [
                "# H2 Fulcher Dataset Extraction",
                "",
                f"Cube glob: `{args.cube_glob}`",
                f"Plan: `{args.plan or ''}`",
                f"Manifest: `{args.manifest or ''}`",
                f"Frames: `{args.frames}`",
                f"Workers: `{workers}`",
                f"Extracted frames: `{len(extraction_rows)}`",
                "",
                "Primary outputs:",
                "",
                "- `intensities/`: fulcheranalyzer-compatible intensity/error matrices",
                "- `fit_reports/`: long-form extraction audit tables",
                "- `plots/extraction_lines/`: line-fit PDFs, per cube in serial mode or per frame in parallel mode",
                "- `calibration_manifest.csv`: cube calibration metadata and digests",
                "- `policy_summary.csv`: per-frame extraction policy counts",
                "",
                "Downstream analysis:",
                "",
                "```shell",
                "fulcher-analyze-batch --plan h2_dataset_plan.toml",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _print_artifact_summary(
        title="H2 Fulcher extraction",
        output_dir=output_dir,
        artifacts=[
            ("intensities", intensity_dir),
            ("fit reports", fit_report_dir),
            ("region plots", qc_region_dir),
            ("line PDFs", qc_line_dir),
            ("summary", extraction_summary_path),
            ("policy", policy_summary_path),
            ("calibration", calibration_manifest_path),
            ("readme", readme_path),
        ],
    )
    print(f"extracted frames: {len(extraction_rows)}")
    if interrupted:
        raise SystemExit(130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=None, help="TOML run plan with [common], [scan], and [extract] sections.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["scan", "extract"],
        default="extract",
        help="scan ranks candidate frames; extract writes intensity tables.",
    )
    parser.add_argument("--cube-glob", default=DEFAULT_CUBE_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", default="all", help="all, a comma list, or ranges like 0-10,20")
    parser.add_argument("--manifest", type=Path, default=None, help="CSV from scan, usually selected_frames.csv.")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--max-cubes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--shots-toml", type=Path, default=DEFAULT_SHOTS_TOML)
    parser.add_argument("--shot-groups", default="", help="scan: comma-separated groups from shots.toml, e.g. rax36,rax39.")
    parser.add_argument("--select-top", type=int, default=None, help="scan: optional top N frames by scan score.")
    parser.add_argument("--scan-threshold", type=float, default=None, help="scan: optional minimum scan score.")
    parser.add_argument("--fulcher-min-nm", type=float, default=DEFAULT_FULCHER_MIN_NM)
    parser.add_argument("--fulcher-max-nm", type=float, default=DEFAULT_FULCHER_MAX_NM)
    parser.add_argument("--progress-every", type=int, default=5, help="Print progress every N cubes/frames.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and use periodic log lines.")
    parser.add_argument("--qc-every", type=int, default=0, help="Write extraction plots every N frames; 0 disables.")
    parser.add_argument("--line-fit-qc", action="store_true", help="Write line-fit PDFs for extracted frames.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="extract: parallel frame workers. Use 0 for CPU count minus one.",
    )
    args = parser.parse_args()
    provided = _provided_destinations(parser, sys.argv[1:])
    if args.plan:
        _apply_plan(args, args.plan, provided)

    if args.command == "scan":
        scan_frames(args)
        return
    if args.command == "extract":
        extract_dataset(args)
        return


if __name__ == "__main__":
    main()
