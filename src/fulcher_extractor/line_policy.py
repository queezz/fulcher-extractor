"""Recovered per-line extraction policy hints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
    import tomli as tomllib  # type: ignore[no-redef]

from .fit import LineFitResult
from .line_database import FulcherLine

RESOURCE_PACKAGE = "fulcher_extractor.resources"
DEFAULT_POLICY_RESOURCE = "line_policies.toml"


@dataclass(frozen=True)
class LinePolicy:
    """One recovered human decision for a fitted line."""

    line_id: str
    decision: str
    matrix_action: str
    evidence: str
    fit_hint: str = ""
    source: str = ""
    line_scale_role: str = ""


@dataclass(frozen=True)
class LinePolicySet:
    """Loaded policy hints plus display-only QC selections."""

    policies: dict[str, LinePolicy]
    overview_qc_line_ids: frozenset[str]


def load_line_policy_set(path: str | Path | None = None) -> LinePolicySet:
    """Load recovered policy hints and display-only QC line selections."""
    payload = _load_policy_payload(path)
    policies, used_line_ids = _policies_from_payload(payload)
    overview_qc_line_ids = set(used_line_ids)
    overview_qc_line_ids.update(
        str(line_id)
        for line_id in payload.get("overview_qc", {}).get(
            "trusted_decontaminated_line_ids", []
        )
    )
    return LinePolicySet(
        policies=policies,
        overview_qc_line_ids=frozenset(overview_qc_line_ids),
    )


def load_line_policies(path: str | Path | None = None) -> dict[str, LinePolicy]:
    """Load recovered H2 line decisions keyed by line id."""
    return load_line_policy_set(path).policies


def _load_policy_payload(path: str | Path | None = None) -> dict:
    if path is None:
        return tomllib.loads(
            files(RESOURCE_PACKAGE).joinpath(DEFAULT_POLICY_RESOURCE).read_text()
        )
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def _policies_from_payload(payload: dict) -> tuple[dict[str, LinePolicy], set[str]]:
    used_line_ids: set[str] = set()
    line_scale_source = ""
    for section in payload.get("line_scale", {}).values():
        used_line_ids.update(str(line_id) for line_id in section.get("used_line_ids", []))
        line_scale_source = str(section.get("source", line_scale_source))

    policies: dict[str, LinePolicy] = {}
    for row in payload.get("policies", []):
        line_id = str(row["line_id"])
        policies[line_id] = LinePolicy(
            line_id=line_id,
            decision=str(row["decision"]),
            matrix_action=str(row["matrix_action"]),
            evidence=str(row.get("evidence", "")),
            fit_hint=str(row.get("fit_hint", "")),
            source=str(row.get("source", "")),
            line_scale_role="used" if line_id in used_line_ids else "",
        )

    for line_id in used_line_ids:
        if line_id in policies:
            continue
        policies[line_id] = LinePolicy(
            line_id=line_id,
            decision="line_scale_used",
            matrix_action="keep",
            evidence="Labelled as used in the old H2 line-scale overview plot.",
            source=line_scale_source,
            line_scale_role="used",
        )
    return policies, used_line_ids


def overview_qc_lines(
    lines: Iterable[FulcherLine],
    *,
    policy_set: LinePolicySet | None = None,
) -> list[FulcherLine]:
    """Return lines to label and guide as useful in the overview QC plot."""
    selected = policy_set or load_line_policy_set()
    return [line for line in lines if line.line_id in selected.overview_qc_line_ids]


def apply_line_policies(
    results: Iterable[LineFitResult],
    *,
    policies: dict[str, LinePolicy] | None = None,
) -> list[LineFitResult]:
    """Attach recovered policy metadata and matrix-export values to fit results."""
    policy_by_id = policies if policies is not None else load_line_policies()
    return [_apply_policy(result, policy_by_id.get(result.line_id)) for result in results]


def line_scale_role(line: FulcherLine, policies: dict[str, LinePolicy]) -> str:
    """Return whether a line was labelled in the old H2 line-scale plot."""
    policy = policies.get(line.line_id)
    if policy and policy.line_scale_role:
        return policy.line_scale_role
    if line.isotopologue == "H2":
        return "not_labelled"
    return ""


def _apply_policy(result: LineFitResult, policy: LinePolicy | None) -> LineFitResult:
    if policy is None:
        return replace(
            result,
            matrix_amplitude=result.amplitude,
            matrix_amplitude_stderr=result.amplitude_stderr,
        )

    matrix_amplitude = result.amplitude
    matrix_stderr = result.amplitude_stderr
    status = result.status
    if policy.matrix_action == "zero":
        matrix_amplitude = 0.0
        matrix_stderr = 0.0
        status = f"{status};legacy_zeroed"
    status = f"{status};legacy_{policy.decision}"
    return replace(
        result,
        status=status,
        legacy_policy=policy.decision,
        legacy_matrix_action=policy.matrix_action,
        legacy_evidence=policy.evidence,
        legacy_fit_hint=policy.fit_hint,
        legacy_line_scale_role=policy.line_scale_role,
        matrix_amplitude=matrix_amplitude,
        matrix_amplitude_stderr=matrix_stderr,
    )
