"""Reject/accept repair loop for ArchStudio-S2 layout-to-3D runs.

This module is the Stage-2 counterpart of the Stage-1 layout architect loop:

    validate -> reject/accept -> choose repair operations -> execute -> validate

The implementation is intentionally additive and trace-first.  It can run as a
pure local repair loop (mesh snap/resize and contract-surface splitting), or it
can invoke the already-existing T2I and Hunyuan-Omni entry points when command
templates are supplied by the CLI.  The geometry generation logic itself is not
changed here; the agent only decides when to retry and how to preserve/rebuild
run artifacts after a retry.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .assembly import (
    MeshData,
    bbox_center_size,
    create_box_mesh,
    mesh_bounds,
    normalize_mesh_file_to_bbox,
    read_obj_mesh,
    write_obj_mesh,
    write_placeholder_mesh,
)
from .hunyuan_adapter import HunyuanAdapter, HunyuanAdapterPolicy, HunyuanPartResult
from .prompting import (
    NEGATIVE_PROMPT,
    PROMPT_POLICY_VERSION,
    build_part_prompt,
    component_context_tail,
    component_core_description,
    component_source_description,
    component_visual_subject,
    negative_prompt_for_part,
    prompt_hash,
)
from .quality_agent import (
    DEFAULT_MAX_SAMPLE_VERTICES,
    build_part_geometry_contract,
    build_evidence_pack,
    build_report_asset_evidence_pack,
    build_visual_evaluator_prompt,
    collect_quality_metrics,
    evaluate_quality,
    load_json,
    make_quality_summary,
    quality_findings_from_focused_visual_critics,
    percentile_point_bounds,
    resolve_part_mesh_path,
    run_focused_visual_critics,
    write_json,
    write_s2_visual_critic_sidecars,
    write_visual_evaluator_request_artifacts,
    write_visual_evaluator_response_artifacts,
    write_quality_trace,
)
from .report_bundle import refresh_s2_9999_report
from .s2_agent_trace import write_agent_trace


AGENT_LOOP_SCHEMA_VERSION = "archstudio_s2_repair_agent_loop.v1"
VALIDATION_SCHEMA_VERSION = "archstudio_s2_validation_report.v1"
OPERATION_SCHEMA_VERSION = "archstudio_s2_repair_operation.v1"

DEFAULT_ACCEPTANCE_SCORE = 0.90
DEFAULT_REJECT_SEVERITY = 3
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_SNAP_PADDING = 1.0
DEFAULT_SPLIT_THICKNESS_FRACTION = 0.06
DEFAULT_MIN_SPLIT_THICKNESS = 0.012
DEFAULT_MAX_NO_IMPROVEMENT_ITERATIONS = 2
DEFAULT_MIN_SCORE_IMPROVEMENT = 0.01
PROMPT_REPAIR_SCHEMA_VERSION = "archstudio_s2_agent_prompt_revision.v1"


def _file_sha256(path: str | Path | None) -> Optional[str]:
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_or_replace_cli_option(command_template: str, option: str, value: Any) -> str:
    """Replace a CLI option or append it inside a bash -lc payload when possible."""
    pattern = rf"{re.escape(option)}(?:\s+|=)([^\s\"\']+)"
    replacement = f"{option} {value}"
    if re.search(pattern, command_template):
        return re.sub(pattern, replacement, command_template, count=1)

    stripped = command_template.rstrip()
    trailing_ws = command_template[len(stripped):]
    if stripped.endswith('"') and 'bash -lc "' in stripped:
        return f'{stripped[:-1]} {replacement}"{trailing_ws}'
    if stripped.endswith("'") and "bash -lc '" in stripped:
        return f"{stripped[:-1]} {replacement}'{trailing_ws}"
    return f"{command_template} {replacement}"

Vector3 = Tuple[float, float, float]
Bounds = Tuple[Vector3, Vector3]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _round(values: Sequence[float], digits: int = 6) -> List[float]:
    return [round(float(value), digits) for value in values]


def _sanitize_id(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "part"


def _as_float3(values: Sequence[Any]) -> Vector3:
    if len(values) != 3:
        raise ValueError("expected 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _bbox_bounds(part: Mapping[str, Any]) -> Bounds:
    center, size = bbox_center_size(part)
    return (
        (center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, center[2] - size[2] / 2.0),
        (center[0] + size[0] / 2.0, center[1] + size[1] / 2.0, center[2] + size[2] / 2.0),
    )


def _extent(bounds: Bounds) -> Vector3:
    mn, mx = bounds
    return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


def _center(bounds: Bounds) -> Vector3:
    mn, mx = bounds
    return ((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0)


def _manifest_by_part(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(part.get("part_id")): dict(part)
        for part in manifest.get("parts", []) or []
        if part.get("part_id")
    }


def _contract_by_part(contract: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(part.get("part_id")): dict(part)
        for part in contract.get("parts", []) or []
        if part.get("part_id")
    }


def _load_run_artifacts(run_dir: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest.json: {manifest_path}")
    manifest = load_json(manifest_path)
    contract_candidates = [
        run_dir / "contract" / "layout_mesh_contract.json",
        run_dir / "layout_mesh_contract.json",
    ]
    for part in manifest.get("parts", []) or []:
        if part.get("contract_path"):
            contract_candidates.append(Path(str(part["contract_path"])))
    contract = None
    for path in contract_candidates:
        if path.exists():
            payload = load_json(path)
            if isinstance(payload, dict) and payload.get("parts"):
                contract = payload
                break
    if contract is None:
        raise FileNotFoundError(f"cannot locate layout mesh contract for {run_dir}")
    failures_path = run_dir / "failures.json"
    failures = load_json(failures_path) if failures_path.exists() else {"failures": []}
    return contract, manifest, failures


@dataclass
class S2Finding:
    issue_id: str
    issue_type: str
    part_ids: List[str]
    severity: int
    confidence: float
    diagnosis: str
    recommended_operation: str
    evidence: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    repair_instruction: str = ""
    source: str = "deterministic_s2_agent"
    failure_stage: str = "assembly_repair"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class S2ValidationReport:
    iteration: int
    accepted: bool
    scene_score: float
    findings: List[S2Finding]
    metrics: Dict[str, Any]
    acceptance_policy: Dict[str, Any]
    evaluator_findings: Optional[Dict[str, Any]] = None
    evaluator_error: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = VALIDATION_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "iteration": self.iteration,
            "accepted": self.accepted,
            "scene_score": self.scene_score,
            "acceptance_policy": self.acceptance_policy,
            "evaluator_findings": self.evaluator_findings,
            "evaluator_error": self.evaluator_error,
            "summary": {
                "num_findings": len(self.findings),
                "max_severity": max((finding.severity for finding in self.findings), default=0),
                "rejecting_findings": sum(
                    1
                    for finding in self.findings
                    if finding.severity >= int(self.acceptance_policy.get("reject_severity", DEFAULT_REJECT_SEVERITY))
                ),
            },
            "findings": [finding.to_dict() for finding in self.findings],
            "metrics": self.metrics,
        }


@dataclass
class S2RepairOperation:
    operation_id: str
    operation_type: str
    part_ids: List[str]
    issue_ids: List[str] = field(default_factory=list)
    status: str = "planned"
    details: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    schema_version: str = OPERATION_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class S2AgentConfig:
    run_dir: Path
    texture_run_dir: Optional[Path] = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_no_improvement_iterations: int = DEFAULT_MAX_NO_IMPROVEMENT_ITERATIONS
    min_score_improvement: float = DEFAULT_MIN_SCORE_IMPROVEMENT
    acceptance_score: float = DEFAULT_ACCEPTANCE_SCORE
    reject_severity: int = DEFAULT_REJECT_SEVERITY
    max_sample_vertices: int = DEFAULT_MAX_SAMPLE_VERTICES
    dry_run: bool = False
    force_report: bool = True
    fast_report_refresh: bool = False
    evaluator_provider: str = "metrics"
    evaluator_model: Optional[str] = None
    evaluator_base_url: Optional[str] = None
    evaluator_auth_path: Optional[Path] = None
    evaluator_max_images: int = 12
    evaluator_build_evidence: Optional[bool] = None
    evaluator_fast_evidence: bool = True
    evaluator_mock_response: Optional[str | Path | Mapping[str, Any]] = None
    evaluator_required: bool = False
    evaluator_mode: str = "focused"
    focused_critic_max_parts: int = 0
    focused_critic_scopes: Optional[List[str]] = None
    agent_scope: str = "full_s2_repair"
    sequential_part_gate: bool = True
    focused_critic_score_threshold: float = 0.72
    provider: str = "qwen_image_local"
    t2i_python: Optional[Path] = None
    t2i_env: Dict[str, str] = field(default_factory=dict)
    t2i_gpus: List[str] = field(default_factory=list)
    t2i_seed: int = 1234
    t2i_rerun_seed_stride: int = 1009
    t2i_steps: Optional[int] = None
    t2i_guidance_scale: Optional[float] = None
    t2i_retries: int = 1
    t2i_retry_sleep: float = 5.0
    hunyuan_command: Optional[str] = None
    hunyuan_gpus: List[str] = field(default_factory=list)
    hunyuan_timeout_seconds: int = 20 * 60
    hunyuan_retry_count: int = 0
    hunyuan_seed: int = 1234
    hunyuan_rerun_seed_stride: int = 1009
    enable_t2i: bool = True
    enable_geometry: bool = True
    enable_snap: bool = True
    enable_split: bool = True
    split_then_generate: bool = True
    forbid_procedural_t2i: bool = True


def s2_findings_from_quality_findings(
    quality_findings: Mapping[str, Any] | None,
    *,
    iteration: int,
) -> List[S2Finding]:
    """Convert VLM quality-agent schema into S2 executable findings."""
    if not quality_findings:
        return []
    converted: List[S2Finding] = []
    for index, issue in enumerate(quality_findings.get("issues", []) or [], start=1):
        issue_type = str(issue.get("issue_type") or "low_geometry_quality")
        action = quality_action_to_s2_operation(issue.get("recommended_action"), issue_type)
        converted.append(
            S2Finding(
                issue_id=str(issue.get("issue_id") or f"vlm_{iteration:02d}_{index:03d}"),
                issue_type=issue_type,
                part_ids=[str(part_id) for part_id in issue.get("part_ids", []) or []],
                labels=[int(label) for label in issue.get("labels", []) or [] if str(label).lstrip("-").isdigit()],
                severity=int(issue.get("severity", 3)),
                confidence=float(issue.get("confidence", 0.75)),
                diagnosis=str(issue.get("diagnosis") or ""),
                recommended_operation=action,
                evidence=[str(item) for item in issue.get("evidence", []) or []],
                repair_instruction=str(issue.get("repair_instruction") or ""),
                source="vlm_gpt55_quality_agent",
                failure_stage=failure_stage_for_quality_issue(issue, issue_type, action),
            )
        )
    return converted



def failure_stage_for_quality_issue(issue: Mapping[str, Any], issue_type: Any, operation_type: Any = None) -> str:
    evidence = [str(item) for item in issue.get("evidence", []) or []] if isinstance(issue, Mapping) else []
    if any(item == "focused_scope:part_image" for item in evidence):
        return "t2i"
    if any(item == "focused_scope:part_mesh" for item in evidence):
        return "geometry"
    if any(item == "focused_scope:scene_assembly" for item in evidence):
        return "assembly_repair"
    return failure_stage_for_issue(issue_type, operation_type)

def failure_stage_for_issue(issue_type: Any, operation_type: Any = None) -> str:
    """Return a stable failure taxonomy for reporting and batch analytics."""
    issue = str(issue_type or "")
    operation = str(operation_type or "")
    if issue.startswith("t2i_") or operation in {"rerun_t2i", "rerun_t2i_geometry"}:
        return "t2i"
    if issue in {"artifact_base_or_panel", "wrong_semantics"}:
        return "t2i"
    if issue in {"geometry_generation_failed", "missing_part", "low_geometry_quality", "wrong_orientation"}:
        return "geometry"
    if issue in {"scale_mismatch", "gap_or_seam", "overlap", "floating", "penetration"} or operation == "deterministic_resize_snap":
        return "assembly_repair"
    if operation == "split_layout_part_faces" or issue == "box_like_surface_candidate":
        return "bbox_layout_mutation"
    if issue in {"image_edit_needed", "background_contamination"}:
        return "image_edit"
    if issue.startswith("report_"):
        return "report"
    return "assembly_repair"


def quality_action_to_s2_operation(action: Any, issue_type: str) -> str:
    action = str(action or "")
    if action == "accept":
        return "manual_review"
    if action == "deterministic_resize_snap":
        return "deterministic_resize_snap"
    if action == "rerun_t2i":
        return "rerun_t2i"
    if action == "rerun_geometry":
        return "rerun_geometry"
    if action == "rerun_texture":
        return "rerun_texture"
    if action == "rerun_t2i_geometry_texture":
        return "rerun_t2i_geometry"
    if issue_type in {"wrong_semantics", "artifact_base_or_panel", "low_geometry_quality", "missing_part"}:
        return "rerun_t2i_geometry"
    if issue_type in {"scale_mismatch", "wrong_orientation", "gap_or_seam", "overlap", "floating", "penetration"}:
        return "deterministic_resize_snap"
    if issue_type in {"texture_bad", "texture_direction_bad", "low_texture_quality", "style_inconsistent"}:
        return "rerun_texture"
    return "manual_review"


@dataclass
class S2AgentResult:
    run_dir: str
    accepted: bool
    stop_condition: str
    iterations: List[Dict[str, Any]]
    trace_path: str
    report_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)






def _safe_file_component(value: Any, *, limit: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "artifact")).strip("._-")
    return (text or "artifact")[:limit]

def _part_artifact_snapshot(
    run_dir: str | Path,
    part_id: str,
    *,
    freeze_dir: Optional[str | Path] = None,
    snapshot_label: str = "snapshot",
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    contract, manifest, _failures = _load_run_artifacts(run_dir)
    manifest_part = _manifest_by_part(manifest).get(str(part_id), {})
    paths = {
        "reference_image": run_dir / "parts" / str(part_id) / "reference.png",
        "prompt_txt": run_dir / "parts" / str(part_id) / "prompt.txt",
        "t2i_metadata": run_dir / "parts" / str(part_id) / "t2i_metadata.json",
        "normalized_mesh": manifest_part.get("normalized_output_path") or (run_dir / "assemblies" / "parts" / f"{part_id}.obj"),
        "raw_mesh": manifest_part.get("raw_output_path") or (run_dir / "parts" / str(part_id) / f"{part_id}.obj"),
    }
    freeze_root = Path(freeze_dir).resolve() if freeze_dir else None
    if freeze_root is not None:
        freeze_root.mkdir(parents=True, exist_ok=True)
    snapshot: Dict[str, Any] = {
        "part_id": str(part_id),
        "created_at": utc_now_iso(),
        "snapshot_label": str(snapshot_label),
        "paths": {},
        "metadata": {},
    }
    for key, value in paths.items():
        path = Path(str(value)) if value else Path("")
        record = {"path": str(path) if str(path) else "", "exists": bool(str(path) and path.exists())}
        if record["exists"]:
            try:
                stat = path.stat()
                record.update({"size_bytes": stat.st_size, "mtime": stat.st_mtime})
            except Exception:
                pass
            if freeze_root is not None and path.is_file():
                safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{snapshot_label}_{key}_{path.name}").strip("_")
                frozen_path = freeze_root / (safe_stem or f"{snapshot_label}_{key}")
                try:
                    shutil.copy2(path, frozen_path)
                    record["live_path"] = record["path"]
                    record["path"] = str(frozen_path)
                    record["frozen"] = True
                except Exception as exc:
                    record["freeze_error"] = str(exc)[:300]
            if key == "prompt_txt":
                prompt_path = Path(str(record.get("path") or path))
                try:
                    snapshot["metadata"]["prompt_text"] = prompt_path.read_text(encoding="utf-8", errors="ignore")[:5000]
                except Exception:
                    pass
            if key == "t2i_metadata":
                metadata_path = Path(str(record.get("path") or path))
                try:
                    snapshot["metadata"]["t2i_metadata"] = load_json(metadata_path)
                except Exception:
                    pass
        snapshot["paths"][key] = record
    return snapshot



def _snapshot_sha256(path: str | Path | None) -> Optional[str]:
    return _file_sha256(path)


def record_agent_artifact_version(
    run_dir: str | Path,
    *,
    iteration: int,
    operation_id: str,
    part_id: str,
    stage: str,
    moment: str,
    snapshot: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist immutable per-action artifacts for comparison and future rollback.

    Live files such as parts/<part>/reference.png are overwritten on every
    rerun.  This version record points at frozen copies captured by
    _part_artifact_snapshot, plus hashes and live origins, so the 9999 page and
    future rollback code never compare two paths that later mutate to the same
    file.
    """
    run_dir = Path(run_dir).resolve()
    version_dir = (
        run_dir
        / "agent_loop"
        / "artifact_versions"
        / f"iteration_{int(iteration):03d}"
        / _safe_file_component(operation_id)
        / _safe_file_component(part_id)
        / _safe_file_component(stage)
        / _safe_file_component(moment)
    )
    version_dir.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Any] = {}
    paths = snapshot.get("paths") if isinstance(snapshot.get("paths"), Mapping) else {}
    for key, value in paths.items():
        if not isinstance(value, Mapping):
            continue
        frozen_path = value.get("path") if value.get("exists") else None
        records[str(key)] = {
            "path": str(frozen_path or ""),
            "live_path": str(value.get("live_path") or frozen_path or ""),
            "exists": bool(value.get("exists")),
            "frozen": bool(value.get("frozen")),
            "sha256": _snapshot_sha256(frozen_path),
            "size_bytes": value.get("size_bytes"),
        }
    record = {
        "schema_version": "archstudio_s2.agent_artifact_version.v1",
        "created_at": utc_now_iso(),
        "iteration": int(iteration),
        "operation_id": str(operation_id),
        "part_id": str(part_id),
        "stage": str(stage),
        "moment": str(moment),
        "version_dir": str(version_dir),
        "artifacts": records,
        "metadata": snapshot.get("metadata") if isinstance(snapshot.get("metadata"), Mapping) else {},
        "extra": dict(extra or {}),
    }
    path = version_dir / "artifact_version.json"
    write_json(path, record)
    return {"version_record_path": str(path), **record}


def _artifact_path(snapshot: Mapping[str, Any], key: str) -> str:
    paths = snapshot.get("paths") if isinstance(snapshot.get("paths"), Mapping) else {}
    record = paths.get(key) if isinstance(paths.get(key), Mapping) else {}
    if record.get("exists") and record.get("path"):
        return str(record.get("path"))
    return ""


def _diff_prompt_text(before: str, after: str) -> Dict[str, Any]:
    before = normalize_prompt_text(before or "")
    after = normalize_prompt_text(after or "")
    before_words = before.split()
    after_words = after.split()
    before_set = set(before_words)
    after_set = set(after_words)
    added = [word for word in after_words if word not in before_set][:80]
    removed = [word for word in before_words if word not in after_set][:80]
    return {
        "changed": before != after,
        "before_length": len(before),
        "after_length": len(after),
        "added_words_preview": added,
        "removed_words_preview": removed,
    }

def _reference_status_for_part(part: Mapping[str, Any]) -> Dict[str, Any]:
    image_path = Path(str(part.get("reference_image_path") or part.get("reference_image") or ""))
    metadata_path = image_path.parent / "t2i_metadata.json" if str(image_path) else Path("")
    metadata = load_json(metadata_path) if metadata_path and metadata_path.exists() else {}
    return {
        "image_path": str(image_path) if str(image_path) else "",
        "image_exists": bool(str(image_path) and image_path.exists()),
        "metadata_path": str(metadata_path) if metadata_path else "",
        "metadata_exists": bool(metadata_path and metadata_path.exists()),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "status": (metadata or {}).get("status", "missing_metadata"),
        "provider": (metadata or {}).get("provider", part.get("t2i_provider", "")),
    }


def _prompt_revision_path(run_dir: str | Path, part_id: str) -> Path:
    return Path(run_dir).resolve() / "parts" / str(part_id) / "agent_prompt_revision.json"


def _load_prompt_revision(run_dir: str | Path, part_id: str) -> Dict[str, Any]:
    path = _prompt_revision_path(run_dir, part_id)
    if path.exists():
        payload = load_json(path)
        if isinstance(payload, dict):
            return payload
    return {}


def _part_text_subject(part: Mapping[str, Any]) -> str:
    return component_visual_subject(part) or component_core_description(part) or component_source_description(part)


def _geometry_contract_prompt_policy(part: Mapping[str, Any], geometry_contract: Mapping[str, Any], attempt: int) -> Dict[str, Any]:
    """Return geometry-family prompt constraints for the next T2I retry.

    The policy is intentionally bbox-family driven instead of case-specific.
    After repeated VLM rejects, it shifts from semantic-rich wording to a
    stricter primitive-first image prior so I23D gets a clean silhouette.
    """
    shape_family = str(geometry_contract.get("shape_family") or "block_like_mass")
    size = geometry_contract.get("bbox_size_xyz") or []
    try:
        x, y, z = [float(v) for v in list(size)[:3]]
    except Exception:
        x, y, z = 0.0, 0.0, 0.0
    width_to_depth = x / max(y, 1e-6) if x and y else None
    height_to_width = z / max(x, 1e-6) if x and z else None
    strictness = "semantic_geometry" if attempt <= 1 else "geometry_primitive" if attempt <= 3 else "minimal_i23d_silhouette"
    ratio_sentence = (
        f"layout bbox size is x={x:.3f}, y={y:.3f}, z={z:.3f}; "
        f"use x as long horizontal width, y as shallow depth/thickness, z as vertical height; "
        f"approx width/depth {width_to_depth:.2f} and height/width {height_to_width:.2f}."
        if width_to_depth is not None and height_to_width is not None
        else "match the bbox width/depth/height proportions exactly."
    )
    if shape_family == "tall_upright_thin_wall_or_gallery_bar":
        subject = "one isolated long upright thin rectangular gallery wall bar / rear gallery wing slab"
        positive = (
            "single clean monolithic elongated vertical wall-like architectural component, long horizontal x direction, "
            "very shallow y thickness visible only as a narrow side face, z-up vertical faces, flat roof edge, "
            "orthographic/isometric product view of the object only, simple facade relief painted on the wall surface, "
            "no separate structural objects"
        )
        negative = (
            "portal, entry gate, portico, columns, colonnade, archway, stairs, steps, ramp, platform, podium, plinth, base, "
            "paving slab, floor plane, courtyard ground, sidewalk, street scene, attached pavilion, bulky end tower, signs, "
            "whole museum, complete courtyard, facade crop with environment, separate foreground/background objects"
        )
    elif shape_family == "low_horizontal_slab_or_ground_plane":
        subject = "one isolated low horizontal stone slab or paving plane component"
        positive = (
            "single low flat horizontal component, top footprint clear, very small z height, orthographic product view, "
            "object only on empty white background"
        )
        negative = "tall wall, tower, facade, building, columns, plinth under another object, scene ground continuing into background"
    elif shape_family == "vertical_column_fin_or_tower":
        subject = "one isolated vertical upright column fin or tower-like component"
        positive = (
            "single clean vertical object, height dominant over footprint, z-up, visible front and side silhouette, no base or scene"
        )
        negative = "ground slab, wall facade scene, multiple columns, full building, pedestal, plinth, platform"
    elif shape_family == "thin_linear_architectural_bar_or_wall":
        subject = "one isolated elongated thin architectural bar or wall component"
        positive = (
            "single long thin component, shallow depth, clear long-axis silhouette, front and side faces visible, object only"
        )
        negative = "bulky block, whole building, base, plinth, platform, street/courtyard scene, background panel"
    else:
        subject = "one isolated bbox-proportional architectural mass component"
        positive = "single clean architectural mass matching bbox proportions, front and side faces visible, object only, no scene"
        negative = "whole building, courtyard, scene, base, plinth, platform, extra neighboring parts, background panel"
    if strictness == "geometry_primitive":
        positive += "; prioritize primitive bbox silhouette over decorative semantic richness; keep details as flat low-relief surface texture only"
    elif strictness == "minimal_i23d_silhouette":
        positive += "; minimal geometry-first asset, plain readable silhouette, no ornaments that change outline, no contextual architecture"
    return {
        "shape_family": shape_family,
        "strictness": strictness,
        "subject": subject,
        "positive_constraints": positive,
        "negative_constraints": negative,
        "ratio_sentence": ratio_sentence,
    }


def build_prompt_revision_for_part(
    run_dir: str | Path,
    part: Mapping[str, Any],
    *,
    iteration: int,
    operation: S2RepairOperation,
    repair_instruction: str = "",
) -> Dict[str, Any]:
    """Create a generic, traceable prompt override for a rejected part.

    This is the key bridge from VLM critique to the next T2I attempt.  It keeps
    the open-vocabulary subject from S1/S2, then appends a bounded correction
    derived from the finding.  The T2I script reads this file before falling
    back to the default prompt builder.
    """
    part_id = str(part.get("part_id"))
    current = _load_prompt_revision(run_dir, part_id)
    current_attempt = int(current.get("revision_attempt", 0) or 0)
    next_attempt = current_attempt + 1
    subject = _part_text_subject(part)
    default_prompt = build_part_prompt(part)
    default_negative = negative_prompt_for_part(part)
    geometry_contract = build_part_geometry_contract(part)
    geometry_policy = _geometry_contract_prompt_policy(part, geometry_contract, next_attempt)
    instruction = normalize_repair_instruction(repair_instruction or operation.details.get("repair_instruction") or "")
    issue_type = str(operation.details.get("source_issue_type") or "agent_repair")
    if next_attempt >= 2:
        base_prompt = (
            "T2I geometry-prior image for I23D, not an architectural scene: {geo_subject}. "
            "Original semantic label only for material/style: {subject}. "
            "{positive}. {ratio_sentence} "
            "Plain seamless white background, centered, object fills most of frame, no cast-shadow floor."
        ).format(
            geo_subject=geometry_policy["subject"],
            subject=subject,
            positive=geometry_policy["positive_constraints"],
            ratio_sentence=geometry_policy["ratio_sentence"],
        )
    else:
        base_prompt = default_prompt
    correction = (
        "Agent repair instruction for this retry: {instruction}. "
        "Geometry contract shape_family={shape_family}; strictness={strictness}. "
        "Generate only this one open-vocabulary architectural component: {geo_subject}. "
        "Positive constraints: {positive}. "
        "Do not draw the full building, courtyard, surrounding scene, neighboring parts, or a facade crop. "
        "Keep architectural up vertical and keep the outward/front face visible. "
        "The object silhouette must match the layout bbox aspect ratio; fill the image with the component itself, not a background board. "
        "Hard negative constraints: {negative_constraints}."
    ).format(
        instruction=instruction or "make the reference image a clean single isolated part",
        shape_family=geometry_policy["shape_family"],
        strictness=geometry_policy["strictness"],
        geo_subject=geometry_policy["subject"],
        positive=geometry_policy["positive_constraints"],
        negative_constraints=geometry_policy["negative_constraints"],
    )
    prompt = normalize_prompt_text(f"{base_prompt}, {correction}")
    negative = normalize_prompt_text(
        f"{default_negative}, {geometry_policy['negative_constraints']}, previous failed artifact, repeated failed composition, wrong semantic object, "
        "whole building, entire site, contextual building mass, background panel, large slab used as filler, "
        "support base, pedestal, plinth, floor plane, ground plane, street, courtyard floor, rotated sideways component, upside down component"
    )
    return {
        "schema_version": PROMPT_REPAIR_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "part_id": part_id,
        "iteration": int(iteration),
        "operation_id": operation.operation_id,
        "operation_type": operation.operation_type,
        "issue_type": issue_type,
        "failure_stage": failure_stage_for_issue(issue_type, operation.operation_type),
        "revision_attempt": next_attempt,
        "source_prompt_hash": prompt_hash(default_prompt, default_negative),
        "prompt": prompt,
        "negative_prompt": negative,
        "repair_instruction": instruction,
        "subject": subject,
        "geometry_contract": geometry_contract,
        "geometry_prompt_policy": geometry_policy,
        "policy": "geometry_contract_aware_i23d_prior_repair_v2",
        "reversible": True,
    }


def normalize_prompt_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def normalize_repair_instruction(value: Any) -> str:
    text = normalize_prompt_text(value)
    if not text:
        return ""
    # Keep provider feedback useful but bounded.  This file is persisted and
    # later included in reports, so avoid huge model paragraphs in prompts.
    words = text.split()
    if len(words) > 80:
        text = " ".join(words[:80])
    return text


def write_prompt_revisions_for_operation(
    run_dir: str | Path,
    operation: S2RepairOperation,
    *,
    iteration: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    contract, _manifest, _failures = _load_run_artifacts(run_dir)
    contract_by_id = _contract_by_part(contract)
    revisions: List[Dict[str, Any]] = []
    for part_id in dict.fromkeys(str(item) for item in operation.part_ids):
        part = contract_by_id.get(part_id)
        if not part:
            revisions.append({"part_id": part_id, "status": "missing_contract_part"})
            continue
        path = _prompt_revision_path(run_dir, part_id)
        before = _load_prompt_revision(run_dir, part_id)
        revision = build_prompt_revision_for_part(
            run_dir,
            part,
            iteration=iteration,
            operation=operation,
            repair_instruction=(
                (operation.details.get("repair_instructions_by_part") or {}).get(part_id)
                if isinstance(operation.details.get("repair_instructions_by_part"), Mapping)
                else None
            ) or operation.details.get("repair_instruction", ""),
        )
        record = {
            "part_id": part_id,
            "status": "dry_run" if dry_run else "succeeded",
            "prompt_revision_path": str(path),
            "previous_revision_attempt": before.get("revision_attempt"),
            "revision_attempt": revision["revision_attempt"],
            "prompt_hash": prompt_hash(revision["prompt"], revision["negative_prompt"]),
            "prompt_preview": revision["prompt"],
            "negative_prompt_preview": revision["negative_prompt"],
            "repair_instruction": revision["repair_instruction"],
            "subject_preview": revision["subject"],
        }
        if not dry_run:
            write_json(path, revision)
            (path.parent / "prompt_agent_repaired.txt").write_text(revision["prompt"] + "\n", encoding="utf-8")
            (path.parent / "negative_prompt_agent_repaired.txt").write_text(revision["negative_prompt"] + "\n", encoding="utf-8")
        revisions.append(record)
    return {"status": "dry_run" if dry_run else "succeeded", "revisions": revisions}


def _plain_box_like(metric: Mapping[str, Any]) -> bool:
    """Detect generic box-like outputs without naming any part class."""
    face_count = int(metric.get("face_count") or 0)
    if face_count > 14 or face_count <= 0:
        return False
    fill = [float(value) for value in metric.get("axis_fill_ratio", []) or []]
    bbox_extent = [float(value) for value in (metric.get("bbox", {}) or {}).get("extent", []) or []]
    if len(fill) != 3 or len(bbox_extent) != 3:
        return False
    if not all(value >= 0.88 for value in fill):
        return False
    positive = [value for value in bbox_extent if value > 1e-9]
    if not positive:
        return False
    aspect = max(positive) / max(min(positive), 1e-9)
    return aspect >= 2.25


def _is_surface_like_metric(metric: Mapping[str, Any]) -> bool:
    bbox_extent = [float(value) for value in (metric.get("bbox", {}) or {}).get("extent", []) or []]
    if len(bbox_extent) != 3 or any(value <= 1e-9 for value in bbox_extent):
        return False
    min_extent = min(bbox_extent)
    max_extent = max(bbox_extent)
    mid_extent = sorted(bbox_extent)[1]
    return min_extent <= max_extent * 0.18 and mid_extent >= min_extent * 2.5


def _part_label_index(metrics: Mapping[str, Any]) -> Dict[str, int]:
    return {
        str(part.get("part_id")): int(part.get("index", 0)) + 1
        for part in metrics.get("parts", []) or []
        if part.get("part_id")
    }


def validate_s2_run(
    run_dir: str | Path,
    *,
    texture_run_dir: str | Path | None = None,
    iteration: int = 0,
    acceptance_score: float = DEFAULT_ACCEPTANCE_SCORE,
    reject_severity: int = DEFAULT_REJECT_SEVERITY,
    max_sample_vertices: int = DEFAULT_MAX_SAMPLE_VERTICES,
    forbid_procedural_t2i: bool = True,
    enable_split: bool = True,
    evaluator_findings: Mapping[str, Any] | None = None,
    evaluator_error: str | None = None,
) -> S2ValidationReport:
    """Validate one S2 run and produce reject/accept findings."""
    contract, manifest, _failures = _load_run_artifacts(run_dir)
    metrics = collect_quality_metrics(
        run_dir,
        texture_run_dir,
        max_sample_vertices=max_sample_vertices,
        deep_mesh_components=False,
    )
    manifest_by_id = _manifest_by_part(manifest)
    label_by_part = _part_label_index(metrics)
    findings: List[S2Finding] = []
    issue_index = 1

    def add_finding(
        issue_type: str,
        part_ids: Sequence[str],
        severity: int,
        confidence: float,
        diagnosis: str,
        recommended_operation: str,
        *,
        evidence: Sequence[str] = (),
        repair_instruction: str = "",
    ) -> None:
        nonlocal issue_index
        findings.append(
            S2Finding(
                issue_id=f"s2_{iteration:02d}_{issue_index:03d}",
                issue_type=issue_type,
                part_ids=[str(part_id) for part_id in part_ids],
                labels=[label_by_part.get(str(part_id), 0) for part_id in part_ids if label_by_part.get(str(part_id), 0)],
                severity=int(severity),
                confidence=float(confidence),
                diagnosis=diagnosis,
                recommended_operation=recommended_operation,
                evidence=list(evidence),
                repair_instruction=repair_instruction,
                failure_stage=failure_stage_for_issue(issue_type, recommended_operation),
            )
        )
        issue_index += 1

    for part in contract.get("parts", []) or []:
        part_id = str(part.get("part_id"))
        reference = _reference_status_for_part(part)
        provider = str(reference.get("provider") or "")
        if not reference["image_exists"]:
            add_finding(
                "t2i_missing_reference",
                [part_id],
                5,
                1.0,
                "Reference image is missing; Hunyuan-Omni cannot be expected to generate a part-level mesh.",
                "rerun_t2i_geometry",
                evidence=["reference_image"],
                repair_instruction="Generate a fresh part-only reference image, then rerun geometry for this part.",
            )
        elif reference["status"] != "succeeded":
            add_finding(
                "t2i_failed",
                [part_id],
                5,
                0.95,
                f"T2I metadata status is {reference['status']!r}, not succeeded.",
                "rerun_t2i_geometry",
                evidence=["t2i_metadata"],
                repair_instruction="Regenerate the reference image with part-only prompt constraints.",
            )
        elif forbid_procedural_t2i and provider in {"procedural_reference", "procedural", "offline_reference"}:
            add_finding(
                "t2i_procedural_fallback",
                [part_id],
                4,
                0.92,
                "Reference image came from procedural fallback instead of a generative T2I model.",
                "rerun_t2i_geometry",
                evidence=["t2i_metadata"],
                repair_instruction="Use local Qwen-Image/SiliconFlow Qwen-Image for this part and rerun Hunyuan geometry.",
            )

        manifest_part = manifest_by_id.get(part_id, {})
        attempts = manifest_part.get("attempts") or []
        final_failure = None
        if attempts:
            final_failure = attempts[-1].get("failure_type")
        if final_failure:
            add_finding(
                "geometry_generation_failed",
                [part_id],
                5,
                0.95,
                f"Hunyuan geometry final failure: {final_failure}.",
                "rerun_geometry" if reference["image_exists"] else "rerun_t2i_geometry",
                evidence=["manifest_attempts"],
                repair_instruction="Rerun Hunyuan-Omni for this exact part after ensuring the reference image is valid.",
            )

    for metric in metrics.get("parts", []) or []:
        part_id = str(metric.get("part_id"))
        flags = set(metric.get("issue_flags", []) or [])
        if "missing_mesh" in flags or "empty_mesh" in flags:
            add_finding(
                "missing_part",
                [part_id],
                5,
                1.0,
                "No usable normalized mesh exists for this contract part.",
                "rerun_t2i_geometry",
                evidence=["metrics_summary"],
                repair_instruction="Regenerate reference and Hunyuan geometry; keep the same layout bbox unless later assembly validation still fails.",
            )
        elif flags.intersection({"mesh_exceeds_bbox", "mesh_center_shift"}):
            add_finding(
                "scale_mismatch",
                [part_id],
                4,
                0.86,
                f"Mesh bounds are shifted or exceed bbox: {sorted(flags)}.",
                "deterministic_resize_snap",
                evidence=["metrics_summary", "assembly_geometry_views"],
                repair_instruction="Affine-snap the current mesh to the layout bbox first; rerun geometry only if visual semantics remain wrong.",
            )
        elif flags.intersection({"low_percentile_bbox_fill", "fragmented_mesh"}):
            add_finding(
                "low_geometry_quality",
                [part_id],
                3,
                0.72,
                f"Mesh only weakly fills its bbox or appears fragmented: {sorted(flags)}.",
                "rerun_t2i_geometry",
                evidence=["metrics_summary", "partboard"],
                repair_instruction="Regenerate a more bbox-suitable single-part reference image and Hunyuan geometry.",
            )

        if enable_split and _plain_box_like(metric) and _is_surface_like_metric(metric):
            add_finding(
                "box_like_surface_candidate",
                [part_id],
                3,
                0.70,
                "The generated mesh is essentially a low-face-count bbox-like solid for an elongated component.",
                "split_layout_part_faces",
                evidence=["metrics_summary", "partboard"],
                repair_instruction=(
                    "Decompose the bbox into generic surface-face child parts so T2I/Hunyuan can focus on "
                    "single exterior faces rather than one huge solid box."
                ),
            )

    for gap in metrics.get("mesh_contact_gaps", []) or []:
        if not gap.get("above_tolerance"):
            continue
        part_ids = [str(gap.get("part_a")), str(gap.get("part_b"))]
        add_finding(
            "gap_or_seam",
            part_ids,
            4,
            0.82,
            "Layout bboxes are connected, but generated mesh percentile bounds leave a visible gap.",
            "deterministic_resize_snap",
            evidence=["mesh_contact_gaps", "assembly_geometry_views"],
            repair_instruction="Snap/resize affected meshes to their layout bboxes before spending a full generation retry.",
        )

    vlm_findings = s2_findings_from_quality_findings(evaluator_findings, iteration=iteration)
    deterministic_keys = {
        (
            finding.issue_type,
            tuple(sorted(finding.part_ids)),
            finding.source,
        )
        for finding in findings
    }
    for finding in vlm_findings:
        key = (finding.issue_type, tuple(sorted(finding.part_ids)), finding.source)
        if key not in deterministic_keys:
            findings.append(finding)
            deterministic_keys.add(key)

    severity_mass = sum(max(0, int(finding.severity)) for finding in findings)
    scene_score = max(0.0, 1.0 - min(1.0, severity_mass / max(5.0 * float(metrics.get("num_parts", 1)), 1.0)))
    max_severity = max((finding.severity for finding in findings), default=0)
    accepted = scene_score >= acceptance_score and max_severity < reject_severity
    return S2ValidationReport(
        iteration=iteration,
        accepted=accepted,
        scene_score=round(scene_score, 4),
        findings=findings,
        metrics=metrics,
        acceptance_policy={
            "acceptance_score": acceptance_score,
            "reject_severity": reject_severity,
            "forbid_procedural_t2i": forbid_procedural_t2i,
            "enable_split": enable_split,
            "evaluator_provider": (evaluator_findings or {}).get("provider_metadata", {}).get("provider") if evaluator_findings else "none",
            "evaluator_error": evaluator_error,
            "accepted_if": "scene_score >= acceptance_score and max_finding_severity < reject_severity",
        },
        evaluator_findings=dict(evaluator_findings) if evaluator_findings else None,
        evaluator_error=evaluator_error,
    )


def repair_operations_from_findings(
    findings: Sequence[S2Finding],
    *,
    enable_t2i: bool = True,
    enable_geometry: bool = True,
    enable_snap: bool = True,
    enable_split: bool = True,
) -> List[S2RepairOperation]:
    """Convert validation findings into S1-style executable repair operations."""
    priority = {
        "t2i_missing_reference": 100,
        "t2i_failed": 100,
        "t2i_procedural_fallback": 95,
        "geometry_generation_failed": 92,
        "missing_part": 90,
        "scale_mismatch": 80,
        "gap_or_seam": 75,
        "floating": 74,
        "penetration": 74,
        "overlap": 74,
        "box_like_surface_candidate": 70,
        "low_geometry_quality": 65,
    }
    op_enable = {
        "rerun_t2i_geometry": enable_t2i and enable_geometry,
        "rerun_t2i": enable_t2i,
        "rerun_geometry": enable_geometry,
        "deterministic_resize_snap": enable_snap,
        "split_layout_part_faces": enable_split,
        "rerun_texture": True,
        "manual_review": True,
    }
    selected_by_part: Dict[str, S2Finding] = {}
    multi_part_ops: List[S2Finding] = []
    sorted_findings = sorted(
        findings,
        key=lambda finding: (
            -int(finding.severity),
            -priority.get(finding.issue_type, 0),
            -float(finding.confidence),
            finding.issue_id,
        ),
    )
    for finding in sorted_findings:
        operation_type = finding.recommended_operation
        if not op_enable.get(operation_type, True):
            continue
        if finding.issue_type == "gap_or_seam" and len(finding.part_ids) > 1:
            multi_part_ops.append(finding)
            continue
        for part_id in finding.part_ids:
            if part_id not in selected_by_part:
                selected_by_part[part_id] = finding

    operations: List[S2RepairOperation] = []
    counter = 1
    grouped_by_operation: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for finding in selected_by_part.values():
        key = (finding.recommended_operation, finding.issue_type, finding.failure_stage)
        group = grouped_by_operation.setdefault(key, {"finding": finding, "part_ids": [], "issue_ids": []})
        for part_id in finding.part_ids:
            if part_id not in group["part_ids"]:
                group["part_ids"].append(part_id)
        group["issue_ids"].append(finding.issue_id)
        group.setdefault("repair_instructions_by_part", {})
        for part_id in finding.part_ids:
            if finding.repair_instruction:
                group["repair_instructions_by_part"][str(part_id)] = finding.repair_instruction
        group.setdefault("diagnoses_by_part", {})
        for part_id in finding.part_ids:
            if finding.diagnosis:
                group["diagnoses_by_part"][str(part_id)] = finding.diagnosis

    for group in grouped_by_operation.values():
        finding = group["finding"]
        operations.append(
            S2RepairOperation(
                operation_id=f"op_{counter:03d}",
                operation_type=finding.recommended_operation,
                part_ids=list(group["part_ids"]),
                issue_ids=list(group["issue_ids"]),
                details={
                    "source_issue_type": finding.issue_type,
                    "failure_stage": finding.failure_stage,
                    "severity": finding.severity,
                    "confidence": finding.confidence,
                    "diagnosis": finding.diagnosis,
                    "repair_instruction": finding.repair_instruction,
                    "repair_instructions_by_part": dict(group.get("repair_instructions_by_part") or {}),
                    "diagnoses_by_part": dict(group.get("diagnoses_by_part") or {}),
                },
            )
        )
        counter += 1

    # Add seam operations for pairs not already covered by a stronger per-part op.
    covered = {part_id for op in operations for part_id in op.part_ids}
    for finding in multi_part_ops:
        if all(part_id in covered for part_id in finding.part_ids):
            continue
        if not enable_snap:
            continue
        operations.append(
            S2RepairOperation(
                operation_id=f"op_{counter:03d}",
                operation_type="deterministic_resize_snap",
                part_ids=list(finding.part_ids),
                issue_ids=[finding.issue_id],
                details={
                    "source_issue_type": finding.issue_type,
                    "failure_stage": finding.failure_stage,
                    "severity": finding.severity,
                    "confidence": finding.confidence,
                    "diagnosis": finding.diagnosis,
                    "repair_instruction": finding.repair_instruction,
                },
            )
        )
        counter += 1

    return operations


def transform_mesh_bounds_to_bbox(
    mesh: MeshData,
    part: Mapping[str, Any],
    *,
    padding: float = DEFAULT_SNAP_PADDING,
    source_bounds: Optional[Bounds] = None,
    clamp_to_bbox: bool = False,
) -> MeshData:
    """Affine-transform an existing mesh so selected source bounds match bbox.

    For seam repair we often care about percentile bounds rather than exact
    min/max: Hunyuan meshes can contain tiny outlier vertices at the bbox face
    while the visible body still floats away from the neighboring part.  Passing
    percentile bounds enlarges the visible body instead of being fooled by the
    outlier.
    """
    source_bounds = source_bounds or mesh_bounds(mesh)
    source_min, source_max = source_bounds
    source_size = [source_max[i] - source_min[i] for i in range(3)]
    target_min, target_max = _bbox_bounds(part)
    target_center = _center((target_min, target_max))
    target_size = list(_extent((target_min, target_max)))
    padded_size = [max(value * float(padding), 1e-8) for value in target_size]
    padded_min = [target_center[i] - padded_size[i] / 2.0 for i in range(3)]

    vertices: List[Vector3] = []
    for vertex in mesh.vertices:
        output = []
        for axis in range(3):
            if source_size[axis] <= 1e-9:
                output.append(target_center[axis])
            else:
                unit = (vertex[axis] - source_min[axis]) / source_size[axis]
                value = padded_min[axis] + unit * padded_size[axis]
                if clamp_to_bbox:
                    value = max(target_min[axis], min(target_max[axis], value))
                output.append(value)
        vertices.append((output[0], output[1], output[2]))
    return MeshData(vertices=vertices, faces=list(mesh.faces))


def _snap_source_bounds(mesh: MeshData, part: Mapping[str, Any]) -> Tuple[Bounds, str]:
    full_bounds = mesh_bounds(mesh)
    try:
        percentile_bounds = percentile_point_bounds(mesh.vertices, low=0.01, high=0.99)
    except Exception:
        return full_bounds, "full_bounds"
    bbox_extent = _extent(_bbox_bounds(part))
    full_extent = _extent(full_bounds)
    percentile_extent = _extent(percentile_bounds)
    full_fill = [full_extent[i] / max(bbox_extent[i], 1e-9) for i in range(3)]
    percentile_fill = [percentile_extent[i] / max(bbox_extent[i], 1e-9) for i in range(3)]
    if any(p < 0.92 and f >= 0.95 for p, f in zip(percentile_fill, full_fill)):
        return percentile_bounds, "percentile_01_99_bounds"
    return full_bounds, "full_bounds"


def update_manifest_paths_for_part(
    run_dir: Path,
    manifest: Dict[str, Any],
    part_id: str,
    *,
    normalized_output_path: str | Path | None = None,
    placeholder_output_path: str | Path | None = None,
    raw_output_path: str | Path | None = None,
    attempts: Optional[List[Mapping[str, Any]]] = None,
    lifecycle_state: Optional[str] = None,
    repair_record: Optional[Mapping[str, Any]] = None,
) -> None:
    parts = manifest.setdefault("parts", [])
    entry = None
    for part in parts:
        if str(part.get("part_id")) == str(part_id):
            entry = part
            break
    if entry is None:
        entry = {
            "layout_id": manifest.get("layout_id"),
            "schema_version": manifest.get("schema_version"),
            "part_id": part_id,
            "attempts": [],
        }
        parts.append(entry)
    if normalized_output_path is not None:
        entry["normalized_output_path"] = str(normalized_output_path)
    if placeholder_output_path is not None:
        entry["placeholder_output_path"] = str(placeholder_output_path)
    if raw_output_path is not None:
        entry["raw_output_path"] = str(raw_output_path)
    if attempts is not None:
        entry["attempts"] = [dict(item) for item in attempts]
    states = list(entry.get("lifecycle_states") or [])
    if lifecycle_state and lifecycle_state not in states:
        states.append(lifecycle_state)
    if states:
        entry["lifecycle_states"] = states
    if repair_record:
        entry.setdefault("agent_repair_history", []).append(dict(repair_record))
    entry["raw_assembly_path"] = str(run_dir / "assemblies" / "raw_hunyuan_assembly.obj")
    entry["placeholder_assembly_path"] = str(run_dir / "assemblies" / "placeholder_filled_assembly.obj")


def rebuild_run_assemblies(run_dir: str | Path) -> Dict[str, Any]:
    """Rebuild raw/complete assembly OBJs after any per-part repair."""
    run_dir = Path(run_dir).resolve()
    contract, manifest, _failures = _load_run_artifacts(run_dir)
    contract_by_id = _contract_by_part(contract)
    manifest_by_id = _manifest_by_part(manifest)
    assembly_dir = run_dir / "assemblies"
    raw_path = assembly_dir / "raw_hunyuan_assembly.obj"
    complete_path = assembly_dir / "placeholder_filled_assembly.obj"
    raw_mesh = MeshData(vertices=[], faces=[])
    complete_mesh = MeshData(vertices=[], faces=[])
    rebuilt_parts: List[Dict[str, Any]] = []

    for part in contract.get("parts", []) or []:
        part_id = str(part.get("part_id"))
        manifest_part = manifest_by_id.get(part_id, {})
        normalized_path = None
        for key in ("normalized_output_path",):
            candidate = manifest_part.get(key)
            if candidate and Path(str(candidate)).exists():
                normalized_path = Path(str(candidate))
                break
        if normalized_path is None:
            candidate = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
            if candidate.exists():
                normalized_path = candidate

        placeholder_path = None
        if normalized_path is not None:
            placeholder_path = normalized_path
            raw_mesh.extend(read_obj_mesh(normalized_path))
            complete_mesh.extend(read_obj_mesh(normalized_path))
        else:
            for candidate in (
                manifest_part.get("placeholder_output_path"),
                run_dir / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj",
            ):
                if candidate and Path(str(candidate)).exists():
                    placeholder_path = Path(str(candidate))
                    break
            if placeholder_path is None:
                placeholder_path = Path(write_placeholder_mesh(part, run_dir / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj"))
            complete_mesh.extend(read_obj_mesh(placeholder_path))

        update_manifest_paths_for_part(
            run_dir,
            manifest,
            part_id,
            normalized_output_path=normalized_path if normalized_path else None,
            placeholder_output_path=placeholder_path if placeholder_path else None,
            lifecycle_state="agent_rebuilt_assembly",
        )
        if part_id in contract_by_id:
            rebuilt_parts.append(
                {
                    "part_id": part_id,
                    "normalized_output_path": str(normalized_path) if normalized_path else None,
                    "placeholder_output_path": str(placeholder_path) if placeholder_path else None,
                }
            )

    raw_assembly_path = None
    if raw_mesh.vertices:
        raw_assembly_path = write_obj_mesh(raw_mesh, raw_path, header="archstudio_s2_agent_rebuilt_raw")
    complete_assembly_path = write_obj_mesh(complete_mesh, complete_path, header="archstudio_s2_agent_rebuilt_complete")
    manifest["raw_assembly_path"] = raw_assembly_path
    manifest["placeholder_assembly_path"] = complete_assembly_path
    manifest.setdefault("agent_assembly_rebuild_history", []).append(
        {
            "created_at": utc_now_iso(),
            "num_parts": len(rebuilt_parts),
            "raw_assembly_path": raw_assembly_path,
            "placeholder_assembly_path": complete_assembly_path,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    return {
        "raw_assembly_path": raw_assembly_path,
        "placeholder_assembly_path": complete_assembly_path,
        "num_parts": len(rebuilt_parts),
        "raw_vertex_count": len(raw_mesh.vertices),
        "complete_vertex_count": len(complete_mesh.vertices),
    }


def _backup_file(path: Path, backup_root: Path, *, suffix: str) -> Optional[str]:
    if not path.exists():
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / f"{path.name}.{suffix}.bak"
    shutil.copy2(path, backup_path)
    return str(backup_path)


def apply_snap_resize_operation(
    run_dir: str | Path,
    part_ids: Sequence[str],
    *,
    iteration: int,
    dry_run: bool = False,
    padding: float = DEFAULT_SNAP_PADDING,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    contract, manifest, _failures = _load_run_artifacts(run_dir)
    contract_by_id = _contract_by_part(contract)
    manifest_by_id = _manifest_by_part(manifest)
    actions: List[Dict[str, Any]] = []
    backup_root = run_dir / "agent_loop" / "backups" / f"iteration_{iteration:03d}"
    for part_id in dict.fromkeys(str(item) for item in part_ids):
        part = contract_by_id.get(part_id)
        if not part:
            actions.append({"part_id": part_id, "status": "missing_contract_part"})
            continue
        mesh_path = resolve_part_mesh_path(run_dir, part_id, manifest_by_id.get(part_id))
        if not mesh_path or not Path(mesh_path).exists():
            actions.append({"part_id": part_id, "status": "missing_mesh"})
            continue
        mesh = read_obj_mesh(mesh_path)
        before_bounds = mesh_bounds(mesh)
        source_bounds, source_bounds_mode = _snap_source_bounds(mesh, part)
        snapped = transform_mesh_bounds_to_bbox(
            mesh,
            part,
            padding=padding,
            source_bounds=source_bounds,
            clamp_to_bbox=source_bounds_mode.startswith("percentile"),
        )
        after_bounds = mesh_bounds(snapped)
        output_path = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        action = {
            "part_id": part_id,
            "status": "dry_run" if dry_run else "succeeded",
            "source_path": str(mesh_path),
            "output_path": str(output_path),
            "padding": padding,
            "source_bounds_mode": source_bounds_mode,
            "source_bounds": {"min": _round(source_bounds[0]), "max": _round(source_bounds[1])},
            "before_bounds": {"min": _round(before_bounds[0]), "max": _round(before_bounds[1])},
            "after_bounds": {"min": _round(after_bounds[0]), "max": _round(after_bounds[1])},
        }
        if not dry_run:
            action["backup_path"] = _backup_file(Path(mesh_path), backup_root, suffix=f"{part_id}_snap")
            write_obj_mesh(snapped, output_path, header=f"archstudio_s2_agent_snap_resize iteration={iteration} part_id={part_id}")
            update_manifest_paths_for_part(
                run_dir,
                manifest,
                part_id,
                normalized_output_path=output_path,
                placeholder_output_path=output_path,
                lifecycle_state="agent_snap_resize_applied",
                repair_record={
                    "type": "deterministic_resize_snap",
                    "iteration": iteration,
                    "created_at": utc_now_iso(),
                    "before_bounds": action["before_bounds"],
                    "after_bounds": action["after_bounds"],
                },
            )
        actions.append(action)
    if not dry_run:
        write_json(run_dir / "manifest.json", manifest)
        rebuild_info = rebuild_run_assemblies(run_dir)
    else:
        rebuild_info = None
    return {"actions": actions, "rebuild": rebuild_info}


def _t2i_command(
    config: S2AgentConfig,
    part_ids: Sequence[str],
    *,
    force: bool = True,
    iteration: int = 0,
) -> List[str]:
    python = str(config.t2i_python or sys.executable)
    rerun_seed = int(config.t2i_seed) + max(0, int(iteration)) * int(config.t2i_rerun_seed_stride)
    command = [
        python,
        "scripts/generate_missing_reference_images.py",
        "--contract",
        str(config.run_dir / "contract" / "layout_mesh_contract.json"),
        "--provider",
        config.provider,
        "--output-run-dir",
        str(config.run_dir),
        "--seed",
        str(rerun_seed),
        "--ratio-aware",
        "--retries",
        str(config.t2i_retries),
        "--retry-sleep",
        str(config.t2i_retry_sleep),
    ]
    for part_id in part_ids:
        command.extend(["--part-id", str(part_id)])
    if force:
        command.append("--force")
    if config.t2i_steps is not None:
        command.extend(["--steps", str(config.t2i_steps)])
    if config.t2i_guidance_scale is not None:
        command.extend(["--guidance-scale", str(config.t2i_guidance_scale)])
    return command


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _t2i_min_free_gpu_mem_mib() -> int:
    return max(0, _safe_int(os.environ.get("S2_T2I_MIN_FREE_GPU_MEM_MIB"), 30000))


def _query_gpu_memory_snapshot() -> List[Dict[str, Any]]:
    """Return lightweight nvidia-smi memory data for scheduler decisions."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics should be surfaced, not fatal.
        return [{"status": "query_failed", "error": repr(exc)}]
    if completed.returncode != 0:
        return [{"status": "query_failed", "returncode": completed.returncode, "stderr_tail": completed.stderr[-1000:]}]
    snapshot: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 3:
            continue
        snapshot.append(
            {
                "index": parts[0],
                "memory_used_mib": _safe_int(parts[1]),
                "memory_free_mib": _safe_int(parts[2]),
            }
        )
    return snapshot


def _eligible_t2i_gpus(requested_gpus: Sequence[str], min_free_mib: int) -> Dict[str, Any]:
    requested = [str(gpu).strip() for gpu in requested_gpus if str(gpu).strip()]
    snapshot = _query_gpu_memory_snapshot() if requested and min_free_mib > 0 else []
    by_index = {str(item.get("index")): item for item in snapshot if isinstance(item, Mapping) and item.get("index") is not None}
    # If nvidia-smi is unavailable, keep the old explicit-GPU behavior but expose
    # the missing scheduler evidence in the returned trace.
    query_failed = bool(requested and snapshot and not by_index)
    if query_failed:
        return {
            "requested_gpus": requested,
            "eligible_gpus": requested,
            "skipped_gpus": [],
            "min_free_mib": min_free_mib,
            "gpu_memory_snapshot": snapshot,
            "gpu_filter_status": "unavailable_kept_requested",
        }
    eligible: List[str] = []
    skipped: List[Dict[str, Any]] = []
    for gpu in requested:
        item = by_index.get(str(gpu))
        if item is None:
            skipped.append({"gpu": gpu, "reason": "missing_from_nvidia_smi"})
            continue
        free_mib = _safe_int(item.get("memory_free_mib"))
        if free_mib >= min_free_mib:
            eligible.append(gpu)
        else:
            skipped.append({"gpu": gpu, "reason": "insufficient_free_memory", "memory_free_mib": free_mib})
    return {
        "requested_gpus": requested,
        "eligible_gpus": eligible,
        "skipped_gpus": skipped,
        "min_free_mib": min_free_mib,
        "gpu_memory_snapshot": snapshot,
        "gpu_filter_status": "filtered" if skipped else "all_requested_eligible",
    }


def _is_cuda_oom_result(result: Mapping[str, Any]) -> bool:
    haystack = "\n".join(
        str(result.get(key, ""))
        for key in ("stderr_tail", "stdout_tail", "error", "status")
    ).lower()
    return "cuda out of memory" in haystack or "outofmemoryerror" in haystack


def _t2i_attempt_snapshot(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe subprocess attempt without recursive retry history."""
    return {str(key): value for key, value in result.items() if key != "attempts"}


def _update_t2i_metadata_with_wrapper_result(
    run_dir: Path,
    result: Mapping[str, Any],
    gpu_selection: Mapping[str, Any],
) -> None:
    """Persist GPU scheduling/subprocess evidence next to T2I semantic metadata."""
    attempts = result.get("attempts")
    if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes)):
        safe_attempts = [_t2i_attempt_snapshot(item) for item in attempts if isinstance(item, Mapping)]
    else:
        safe_attempts = [_t2i_attempt_snapshot(result)]
    for part_id in result.get("part_ids", []) or []:
        metadata_path = run_dir / "parts" / str(part_id) / "t2i_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("part_id", str(part_id))
        metadata.setdefault("provider", metadata.get("primary_provider"))
        metadata.setdefault("status", result.get("status", "unknown"))
        metadata["selected_gpu"] = result.get("gpu")
        metadata["wrapper_status"] = result.get("status")
        metadata["wrapper_returncode"] = result.get("returncode")
        metadata["wrapper_command"] = result.get("command")
        metadata["wrapper_stdout_tail"] = result.get("stdout_tail")
        metadata["wrapper_stderr_tail"] = result.get("stderr_tail")
        metadata["wrapper_attempts"] = safe_attempts
        metadata["gpu_selection"] = dict(gpu_selection)
        if result.get("status") != "succeeded":
            metadata["status"] = result.get("status", metadata.get("status", "failed"))
            metadata["error"] = metadata.get("error") or result.get("stderr_tail") or result.get("stdout_tail")
        metadata["reference_image_path"] = str(run_dir / "parts" / str(part_id) / "reference.png")
        write_json(metadata_path, metadata)


def run_t2i_for_parts(config: S2AgentConfig, part_ids: Sequence[str], *, iteration: int = 0) -> Dict[str, Any]:
    unique_part_ids = [str(part_id) for part_id in dict.fromkeys(str(item) for item in part_ids)]
    rerun_seed = int(config.t2i_seed) + max(0, int(iteration)) * int(config.t2i_rerun_seed_stride)
    command = _t2i_command(config, unique_part_ids, iteration=iteration)
    requested_gpus = [str(gpu).strip() for gpu in (config.t2i_gpus or []) if str(gpu).strip()]
    min_free_mib = _t2i_min_free_gpu_mem_mib()
    gpu_selection = _eligible_t2i_gpus(requested_gpus, min_free_mib) if requested_gpus else {
        "requested_gpus": [],
        "eligible_gpus": [],
        "skipped_gpus": [],
        "min_free_mib": min_free_mib,
        "gpu_memory_snapshot": [],
        "gpu_filter_status": "no_explicit_gpus",
    }
    if config.dry_run:
        return {"status": "dry_run", "command": command, "t2i_seed": rerun_seed, **gpu_selection}
    repo_root = Path(__file__).resolve().parents[2]

    eligible_gpus = [str(gpu) for gpu in gpu_selection.get("eligible_gpus", []) if str(gpu).strip()]
    if requested_gpus and not eligible_gpus:
        stderr_tail = (
            "No requested T2I GPU has enough free memory; skipped launching Qwen subprocesses "
            "to avoid before/after reusing stale reference.png. "
            f"min_free_mib={min_free_mib}; requested_gpus={requested_gpus}"
        )
        for part_id in unique_part_ids:
            metadata_path = config.run_dir / "parts" / str(part_id) / "t2i_metadata.json"
            previous_metadata = load_json(metadata_path) if metadata_path.exists() else {}
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                metadata_path,
                {
                    "part_id": str(part_id),
                    "provider": config.provider,
                    "seed": rerun_seed,
                    "status": "failed_no_eligible_gpu",
                    "error": stderr_tail,
                    "reason": "no_requested_gpu_meets_min_free_memory",
                    "gpu_selection": gpu_selection,
                    "previous_status": previous_metadata.get("status") if isinstance(previous_metadata, Mapping) else None,
                    "previous_prompt_hash": previous_metadata.get("prompt_hash") if isinstance(previous_metadata, Mapping) else None,
                    "reference_image_path": str(config.run_dir / "parts" / str(part_id) / "reference.png"),
                },
            )
        return {
            "status": "failed_no_eligible_gpu",
            "reason": "no_requested_gpu_meets_min_free_memory",
            "command": command,
            "part_ids": unique_part_ids,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": stderr_tail,
            "subprocesses": [],
            **gpu_selection,
        }

    def env_for_gpu(gpu: str | None) -> Dict[str, str]:
        env = dict(os.environ)
        env.update(config.t2i_env)
        if gpu is not None and str(gpu) != "":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["QWEN_IMAGE_DEVICE"] = "cuda"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(repo_root) if not pythonpath else f"{repo_root}:{pythonpath}"
        return env

    def run_one(subset: Sequence[str], gpu: str | None = None, *, attempt: int = 1) -> Dict[str, Any]:
        sub_command = _t2i_command(config, subset, iteration=iteration)
        completed = subprocess.run(
            sub_command,
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
            env=env_for_gpu(gpu),
        )
        return {
            "part_ids": [str(item) for item in subset],
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "gpu": str(gpu) if gpu is not None else config.t2i_env.get("CUDA_VISIBLE_DEVICES", ""),
            "attempt": int(attempt),
            "command": sub_command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }

    def run_one_with_oom_retry(subset: Sequence[str], gpu: str | None, retry_gpus: Sequence[str]) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        tried: set[str] = set()

        def current_retry_gpus() -> List[str]:
            refreshed = _eligible_t2i_gpus(requested_gpus, min_free_mib) if requested_gpus else gpu_selection
            return [str(item) for item in refreshed.get("eligible_gpus", []) if str(item).strip()]

        def record_child_metadata(result: Mapping[str, Any]) -> None:
            # The child generator writes provider-level t2i_metadata.json before
            # the wrapper can see its return code.  Snapshot that file per
            # attempt so a later retry does not hide the actual child failure.
            result = dict(result)
            child_metadata: Dict[str, Any] = {}
            for part_id in result.get("part_ids", []) or []:
                metadata_path = config.run_dir / "parts" / str(part_id) / "t2i_metadata.json"
                if metadata_path.exists():
                    try:
                        child_metadata[str(part_id)] = load_json(metadata_path)
                    except Exception as exc:  # noqa: BLE001 - diagnostics only.
                        child_metadata[str(part_id)] = {"status": "metadata_load_failed", "error": repr(exc)}
            result["child_t2i_metadata"] = child_metadata
            attempts.append(result)

        def mark_attempt_superseded(result: Mapping[str, Any], reason: str) -> None:
            for part_id in result.get("part_ids", []) or []:
                metadata_path = config.run_dir / "parts" / str(part_id) / "t2i_metadata.json"
                metadata = load_json(metadata_path) if metadata_path.exists() else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["wrapper_status"] = result.get("status")
                metadata["wrapper_returncode"] = result.get("returncode")
                metadata["selected_gpu"] = result.get("gpu")
                metadata["superseded_by_wrapper_retry"] = True
                metadata["superseded_reason"] = reason
                metadata["wrapper_attempts_so_far"] = [_t2i_attempt_snapshot(item) for item in attempts]
                write_json(metadata_path, metadata)

        first = run_one(subset, gpu, attempt=1)
        record_child_metadata(first)
        if first.get("status") == "succeeded" or not _is_cuda_oom_result(first):
            first["attempts"] = [_t2i_attempt_snapshot(item) for item in attempts]
            return first
        if gpu is not None:
            tried.add(str(gpu))
        mark_attempt_superseded(first, "cuda_oom_retry_on_refreshed_eligible_gpu")

        for candidate in current_retry_gpus() or retry_gpus:
            candidate = str(candidate)
            if candidate in tried:
                continue
            retry = run_one(subset, candidate, attempt=len(attempts) + 1)
            record_child_metadata(retry)
            if retry.get("status") == "succeeded" or not _is_cuda_oom_result(retry):
                retry["attempts"] = [_t2i_attempt_snapshot(item) for item in attempts]
                retry["retried_after_cuda_oom"] = True
                return retry
            tried.add(candidate)
            mark_attempt_superseded(retry, "cuda_oom_retry_on_refreshed_eligible_gpu")
        final = dict(attempts[-1])
        final["attempts"] = [_t2i_attempt_snapshot(item) for item in attempts]
        final["retried_after_cuda_oom"] = len(attempts) > 1
        return final

    gpus = eligible_gpus or requested_gpus
    fallback_gpu = config.t2i_env.get("CUDA_VISIBLE_DEVICES")
    if len(gpus) <= 1 or len(unique_part_ids) <= 1:
        selected_gpu = gpus[0] if gpus else fallback_gpu
        single = run_one_with_oom_retry(unique_part_ids, selected_gpu, gpus)
        _update_t2i_metadata_with_wrapper_result(config.run_dir, single, gpu_selection)
        return {
            "status": single["status"],
            "command": single["command"],
            "returncode": single["returncode"],
            "stdout_tail": single["stdout_tail"],
            "stderr_tail": single["stderr_tail"],
            "part_ids": unique_part_ids,
            "t2i_seed": rerun_seed,
            "gpu": single["gpu"],
            "subprocesses": [single],
            **gpu_selection,
        }

    subprocesses: List[Dict[str, Any]] = []
    max_workers_env = os.environ.get("S2_T2I_MAX_PARALLEL")
    max_workers = min(len(gpus), len(unique_part_ids), max(1, _safe_int(max_workers_env, len(gpus))))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_one_with_oom_retry, [part_id], gpus[index % len(gpus)], gpus): part_id
            for index, part_id in enumerate(unique_part_ids)
        }
        for future in as_completed(future_map):
            result = future.result()
            _update_t2i_metadata_with_wrapper_result(config.run_dir, result, gpu_selection)
            subprocesses.append(result)
    subprocesses.sort(key=lambda item: unique_part_ids.index(str(item.get("part_ids", [""])[0])))
    status = "succeeded" if all(item.get("status") == "succeeded" for item in subprocesses) else "partial_or_failed"
    return {
        "status": status,
        "command": command,
        "part_ids": unique_part_ids,
        "t2i_seed": rerun_seed,
        "parallel": True,
        "gpus": gpus,
        "subprocesses": subprocesses,
        "stdout_tail": "\n".join(str(item.get("stdout_tail", "")) for item in subprocesses)[-4000:],
        "stderr_tail": "\n".join(str(item.get("stderr_tail", "")) for item in subprocesses)[-4000:],
        **gpu_selection,
    }


def _attempts_dict(result: HunyuanPartResult) -> List[Dict[str, Any]]:
    return [attempt.to_dict() for attempt in result.attempts]


def run_geometry_for_parts(config: S2AgentConfig, part_ids: Sequence[str], *, iteration: int = 0) -> Dict[str, Any]:
    if not config.hunyuan_command:
        if config.dry_run:
            return {
                "status": "dry_run",
                "reason": "hunyuan_command_missing_but_dry_run_records_workflow",
                "part_ids": list(part_ids),
                "results": [{"part_id": str(part_id), "status": "dry_run", "command_template": None} for part_id in part_ids],
                "rebuild": None,
            }
        return {
            "status": "blocked",
            "reason": "hunyuan_command_missing",
            "part_ids": list(part_ids),
        }
    run_dir = config.run_dir.resolve()
    contract, manifest, failures = _load_run_artifacts(run_dir)
    contract_by_id = _contract_by_part(contract)
    manifest_by_id = _manifest_by_part(manifest)
    unique_part_ids = [str(part_id) for part_id in dict.fromkeys(str(item) for item in part_ids)]

    def command_for_gpu(gpu: str | None, *, rerun_seed: int | None = None) -> str:
        command_template = str(config.hunyuan_command)
        if rerun_seed is not None:
            command_template = _append_or_replace_cli_option(command_template, "--seed", rerun_seed)
        if gpu is not None and str(gpu).strip():
            command_template = re.sub(r"CUDA_VISIBLE_DEVICES=([^\s\"\']+)", f"CUDA_VISIBLE_DEVICES={gpu}", command_template)
            if "CUDA_VISIBLE_DEVICES=" not in command_template:
                command_template = f"CUDA_VISIBLE_DEVICES={gpu} {command_template}"
        return command_template

    def localized_geometry_part(part: Mapping[str, Any]) -> Dict[str, Any]:
        item = dict(part)
        local_reference = run_dir / "parts" / str(item.get("part_id")) / "reference.png"
        if local_reference.exists():
            item["reference_image_path"] = str(local_reference)
            item["reference_image"] = str(local_reference)
        return item

    def run_one_geometry(part_id: str, gpu: str | None = None, part_index: int = 0) -> Dict[str, Any]:
        part = contract_by_id.get(part_id)
        if not part:
            return {"part_id": part_id, "status": "missing_contract_part", "gpu": str(gpu or "")}
        part = localized_geometry_part(part)
        rerun_seed = int(config.hunyuan_seed) + max(0, int(iteration) + 1) * int(config.hunyuan_rerun_seed_stride) + int(part_index)
        previous_raw = manifest_by_id.get(part_id, {}).get("raw_output_path")
        previous_raw_hash = _file_sha256(previous_raw)
        part_output_dir = run_dir / "parts" / part_id
        for stale_path in part_output_dir.glob(f"{part_id}.*"):
            if stale_path.suffix.lower() in {".obj", ".glb", ".ply"}:
                try:
                    stale_path.unlink()
                except FileNotFoundError:
                    pass
        command_template = command_for_gpu(gpu, rerun_seed=rerun_seed)
        if config.dry_run:
            return {"part_id": part_id, "status": "dry_run", "command_template": command_template, "gpu": str(gpu or ""), "reference_image_path": part.get("reference_image_path"), "hunyuan_seed": rerun_seed}
        adapter = HunyuanAdapter(
            command_template=command_template,
            output_root=run_dir / "parts",
            policy=HunyuanAdapterPolicy(
                timeout_seconds=config.hunyuan_timeout_seconds,
                retry_count=config.hunyuan_retry_count,
                concurrency=1,
            ),
        )
        result = adapter.run_part(part)
        result_record = result.to_dict()
        result_record["gpu"] = str(gpu or "")
        result_record["command_template"] = command_template
        result_record["reference_image_path"] = str(part.get("reference_image_path") or "")
        result_record["hunyuan_seed"] = rerun_seed
        result_record["previous_raw_output_path"] = str(previous_raw or "")
        result_record["previous_raw_sha256"] = previous_raw_hash
        if result.raw_output_path:
            new_raw_hash = _file_sha256(result.raw_output_path)
            result_record["new_raw_sha256"] = new_raw_hash
            if previous_raw_hash and new_raw_hash == previous_raw_hash:
                result_record["status"] = "unchanged_output"
                result_record["error"] = "Hunyuan rerun produced byte-identical raw mesh; not treating as a successful visible mesh repair."
                update_manifest_paths_for_part(
                    run_dir,
                    manifest,
                    part_id,
                    raw_output_path=result.raw_output_path,
                    attempts=_attempts_dict(result),
                    lifecycle_state="agent_geometry_rerun_unchanged",
                    repair_record={
                        "type": "rerun_geometry",
                        "created_at": utc_now_iso(),
                        "status": "unchanged_output",
                        "hunyuan_seed": rerun_seed,
                        "previous_raw_sha256": previous_raw_hash,
                        "new_raw_sha256": new_raw_hash,
                    },
                )
                return result_record
            normalized_path = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
            try:
                normalize_mesh_file_to_bbox(result.raw_output_path, normalized_path, part, cleanup=True)
                update_manifest_paths_for_part(
                    run_dir,
                    manifest,
                    part_id,
                    raw_output_path=result.raw_output_path,
                    normalized_output_path=normalized_path,
                    placeholder_output_path=normalized_path,
                    attempts=_attempts_dict(result),
                    lifecycle_state="agent_geometry_rerun_succeeded",
                    repair_record={
                        "type": "rerun_geometry",
                        "created_at": utc_now_iso(),
                        "raw_output_path": result.raw_output_path,
                        "normalized_output_path": str(normalized_path),
                    },
                )
                result_record["normalized_output_path"] = str(normalized_path)
                result_record["status"] = "succeeded"
            except Exception as exc:  # noqa: BLE001 - trace every failure.
                update_manifest_paths_for_part(
                    run_dir,
                    manifest,
                    part_id,
                    raw_output_path=result.raw_output_path,
                    attempts=_attempts_dict(result),
                    lifecycle_state="agent_geometry_normalization_failed",
                    repair_record={
                        "type": "rerun_geometry",
                        "created_at": utc_now_iso(),
                        "error": repr(exc),
                    },
                )
                result_record["status"] = "normalization_failed"
                result_record["error"] = repr(exc)
        else:
            update_manifest_paths_for_part(
                run_dir,
                manifest,
                part_id,
                attempts=_attempts_dict(result),
                lifecycle_state="agent_geometry_rerun_failed",
                repair_record={
                    "type": "rerun_geometry",
                    "created_at": utc_now_iso(),
                    "final_failure_type": result.final_failure_type,
                },
            )
            failures.setdefault("failures", []).append(
                {
                    "layout_id": result.layout_id,
                    "schema_version": result.schema_version,
                    "part_id": part_id,
                    "attempt_number": result.attempts[-1].attempt_number if result.attempts else None,
                    "failure_type": result.final_failure_type,
                    "exit_code": result.attempts[-1].exit_code if result.attempts else None,
                    "timeout": result.attempts[-1].timeout if result.attempts else None,
                    "stderr_path": result.attempts[-1].stderr_path if result.attempts else None,
                }
            )
            result_record["status"] = "failed"
        return result_record

    gpus = [str(gpu).strip() for gpu in (config.hunyuan_gpus or []) if str(gpu).strip()]
    if len(gpus) > 1 and len(unique_part_ids) > 1 and not config.dry_run:
        results = []
        max_workers = min(len(gpus), len(unique_part_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(run_one_geometry, part_id, gpus[index % len(gpus)], index): part_id
                for index, part_id in enumerate(unique_part_ids)
            }
            for future in as_completed(future_map):
                results.append(future.result())
        results.sort(key=lambda item: unique_part_ids.index(str(item.get("part_id"))))
    else:
        results = [
            run_one_geometry(part_id, gpus[index % len(gpus)] if gpus else None, index)
            for index, part_id in enumerate(unique_part_ids)
        ]
    if not config.dry_run:
        write_json(run_dir / "manifest.json", manifest)
        write_json(run_dir / "failures.json", failures)
        rebuild_info = rebuild_run_assemblies(run_dir)
    else:
        rebuild_info = None
    status = "succeeded" if all(item.get("status") in {"succeeded", "dry_run"} for item in results) else "partial_or_failed"
    return {"status": status, "results": results, "rebuild": rebuild_info}


def _derived_reference_path(run_dir: Path, part_id: str) -> str:
    return str((run_dir / "parts" / part_id / "reference.png").resolve())


def _refresh_prompt_fields(part: Dict[str, Any]) -> None:
    part["part_prompt"] = build_part_prompt(part)
    part["negative_prompt"] = negative_prompt_for_part(part)
    part["prompt_hash"] = prompt_hash(part["part_prompt"], part["negative_prompt"])
    part["prompt_policy_version"] = PROMPT_POLICY_VERSION
    part["part_description_source"] = component_source_description(part)
    part["part_description_core"] = component_core_description(part)
    part["part_visual_subject"] = component_visual_subject(part)
    part["part_context_tail"] = component_context_tail(part)


def decompose_part_into_surface_children(
    run_dir: str | Path,
    part: Mapping[str, Any],
    *,
    thickness_fraction: float = DEFAULT_SPLIT_THICKNESS_FRACTION,
    min_thickness: float = DEFAULT_MIN_SPLIT_THICKNESS,
) -> List[Dict[str, Any]]:
    """Create generic surface-face child parts for a box-like component.

    This is deliberately geometry-driven rather than class-driven.  The child
    prompts inherit the open-vocabulary text but constrain each child to one
    exterior face segment.
    """
    run_dir = Path(run_dir).resolve()
    center, size = bbox_center_size(part)
    min_extent = max(min(size), 1e-6)
    thickness = min(min_extent, max(min_thickness, min_extent * thickness_fraction))
    axis_names = ("x", "y", "z")

    # For very flat panels, split along the thin normal only.  For solid boxes,
    # use vertical exterior faces.  Top/bottom faces are intentionally not the
    # default because most architectural exterior problems are seen in elevation
    # and this remains a geometry rule (z-up vertical faces), not a class rule.
    min_axis = min(range(3), key=lambda axis: size[axis])
    aspect = max(size) / max(min(size), 1e-9)
    if aspect >= 3.0:
        face_specs = [(min_axis, -1), (min_axis, 1)]
    else:
        face_specs = [(0, -1), (0, 1), (1, -1), (1, 1)]

    children: List[Dict[str, Any]] = []
    for axis, sign in face_specs:
        child = copy.deepcopy(dict(part))
        face_name = f"{axis_names[axis]}{'pos' if sign > 0 else 'neg'}"
        child_id = _sanitize_id(f"{part.get('part_id')}__face_{face_name}")
        child_center = list(center)
        child_size = list(size)
        child_center[axis] = center[axis] + sign * (size[axis] / 2.0 - thickness / 2.0)
        child_size[axis] = thickness
        inherited_description = (
            part.get("part_description")
            or part.get("open_vocab_label")
            or part.get("source_actor_label")
            or part.get("target_prompt")
            or "architectural component"
        )
        face_direction = "positive" if sign > 0 else "negative"
        child_description = (
            f"single exterior surface face segment of {inherited_description}; "
            f"{face_direction} {axis_names[axis]} side; one standalone architectural face part"
        )
        child.update(
            {
                "part_id": child_id,
                "source_part_id": part.get("source_part_id") or part.get("part_id"),
                "agent_split_parent_part_id": part.get("part_id"),
                "agent_split_face_axis": axis_names[axis],
                "agent_split_face_sign": sign,
                "agent_split_policy": "generic_bbox_surface_decomposition_v1",
                "source_actor_label": child_description,
                "part_description": child_description,
                "open_vocab_label": child_description,
                "generation_prompt": child_description,
                "target_prompt": (
                    f"3D mesh of one isolated exterior surface face segment: {inherited_description}. "
                    "It must be a single part, not a whole building."
                ),
                "bbox": {"center": _round(child_center), "size": _round(child_size)},
                "actor_location": _round(child_center),
                "actor_size": _round(child_size),
                "reference_image_path": _derived_reference_path(run_dir, child_id),
                "reference_image": _derived_reference_path(run_dir, child_id),
                "lifecycle_state": "agent_split_child_contract_emitted",
            }
        )
        _refresh_prompt_fields(child)
        child["contract_path"] = str((run_dir / "contract" / "layout_mesh_contract.json").resolve())
        children.append(child)
    return children


def apply_split_operation(
    run_dir: str | Path,
    part_ids: Sequence[str],
    *,
    iteration: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    contract, manifest, _failures = _load_run_artifacts(run_dir)
    parts = list(contract.get("parts", []) or [])
    manifest_parts = list(manifest.get("parts", []) or [])
    split_part_ids = set(str(item) for item in part_ids)
    actions: List[Dict[str, Any]] = []
    new_parts: List[Dict[str, Any]] = []
    replaced: Dict[str, List[str]] = {}

    for part in parts:
        part_id = str(part.get("part_id"))
        if part_id not in split_part_ids:
            new_parts.append(part)
            continue
        children = decompose_part_into_surface_children(run_dir, part)
        replaced[part_id] = [child["part_id"] for child in children]
        new_parts.extend(children)
        actions.append(
            {
                "part_id": part_id,
                "status": "dry_run" if dry_run else "succeeded",
                "child_part_ids": [child["part_id"] for child in children],
                "num_children": len(children),
            }
        )

    if dry_run:
        return {"actions": actions, "replaced": replaced}

    contract_backup = _backup_file(run_dir / "contract" / "layout_mesh_contract.json", run_dir / "agent_loop" / "backups" / f"iteration_{iteration:03d}", suffix="contract_pre_split")
    manifest_backup = _backup_file(run_dir / "manifest.json", run_dir / "agent_loop" / "backups" / f"iteration_{iteration:03d}", suffix="manifest_pre_split")
    contract["parts"] = new_parts
    contract.setdefault("agent_contract_mutation_history", []).append(
        {
            "type": "split_layout_part_faces",
            "iteration": iteration,
            "created_at": utc_now_iso(),
            "replaced": replaced,
            "contract_backup": contract_backup,
            "manifest_backup": manifest_backup,
        }
    )
    write_json(run_dir / "contract" / "layout_mesh_contract.json", contract)

    removed = set(replaced.keys())
    manifest["parts"] = [part for part in manifest_parts if str(part.get("part_id")) not in removed]
    for child in new_parts:
        part_id = str(child.get("part_id"))
        if any(part_id == str(item.get("part_id")) for item in manifest["parts"]):
            continue
        if not child.get("agent_split_parent_part_id"):
            continue
        placeholder_path = run_dir / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj"
        write_placeholder_mesh(child, placeholder_path)
        manifest["parts"].append(
            {
                "layout_id": contract.get("layout_id"),
                "schema_version": contract.get("schema_version"),
                "contract_path": str((run_dir / "contract" / "layout_mesh_contract.json").resolve()),
                "part_id": part_id,
                "source_actor_label": child.get("source_actor_label"),
                "target_class": child.get("target_class"),
                "attempts": [],
                "raw_output_path": None,
                "normalized_output_path": None,
                "placeholder_output_path": str(placeholder_path),
                "raw_assembly_path": str(run_dir / "assemblies" / "raw_hunyuan_assembly.obj"),
                "placeholder_assembly_path": str(run_dir / "assemblies" / "placeholder_filled_assembly.obj"),
                "lifecycle_states": ["agent_split_child_manifest_emitted", "placeholder_used"],
            }
        )
        part_dir = run_dir / "parts" / part_id
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / "request.json").write_text(json.dumps(child, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest.setdefault("agent_contract_mutation_history", []).append(contract["agent_contract_mutation_history"][-1])
    write_json(run_dir / "manifest.json", manifest)
    rebuild_info = rebuild_run_assemblies(run_dir)
    return {"actions": actions, "replaced": replaced, "rebuild": rebuild_info}


class S2Repairer:
    """Single-owner S2 repair loop with S1-style reject/accept semantics."""

    def __init__(self, config: S2AgentConfig):
        self.config = config
        self.run_dir = config.run_dir.resolve()
        self.agent_dir = self.run_dir / "agent_loop"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.trace: Dict[str, Any] = {
            "schema_version": AGENT_LOOP_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "run_dir": str(self.run_dir),
            "status": "running",
            "iterations": [],
            "policy": {
                "max_iterations": config.max_iterations,
                "max_no_improvement_iterations": config.max_no_improvement_iterations,
                "min_score_improvement": config.min_score_improvement,
                "acceptance_score": config.acceptance_score,
                "reject_severity": config.reject_severity,
                "provider": config.provider,
                "t2i_gpus": list(config.t2i_gpus or []),
                "hunyuan_gpus": list(config.hunyuan_gpus or []),
                "enable_t2i": config.enable_t2i,
                "enable_geometry": config.enable_geometry,
                "enable_snap": config.enable_snap,
                "enable_split": config.enable_split,
                "split_then_generate": config.split_then_generate,
                "forbid_procedural_t2i": config.forbid_procedural_t2i,
                "evaluator_provider": config.evaluator_provider,
                "evaluator_model": config.evaluator_model,
                "evaluator_max_images": config.evaluator_max_images,
                "evaluator_build_evidence": config.evaluator_build_evidence,
                "evaluator_fast_evidence": config.evaluator_fast_evidence,
                "evaluator_required": config.evaluator_required,
                "agent_scope": config.agent_scope,
                "sequential_part_gate": bool(config.sequential_part_gate),
                "post_experiment_9999_hook": bool(config.force_report and not config.dry_run),
                "post_experiment_9999_hook_target": str(config.run_dir / "report" / "index.html"),
            },
            "score_trajectory": [],
            "failure_taxonomy": {},
            "workflow_transitions": [],
        }
        self._assemble_geometry_phase = "per_part_input_and_mesh_precheck"
        self._sequential_part_ids = self._load_ordered_part_ids()
        self._sequential_part_index = 0
        self._sequential_part_stage = "part_image"

    def _load_ordered_part_ids(self) -> List[str]:
        try:
            contract, _manifest, _failures = _load_run_artifacts(self.run_dir)
            return [str(part.get("part_id")) for part in contract.get("parts", []) or [] if part.get("part_id")]
        except Exception:
            return []

    def _current_sequential_part_id(self) -> Optional[str]:
        if not self._sequential_part_ids:
            return None
        if self._sequential_part_index < 0 or self._sequential_part_index >= len(self._sequential_part_ids):
            return None
        return self._sequential_part_ids[self._sequential_part_index]

    def _sequential_gate_enabled(self) -> bool:
        return bool(self.config.sequential_part_gate and self._is_assemble_geometry_scope() and self._assemble_geometry_phase == "per_part_input_and_mesh_precheck")

    def _evaluator_enabled(self) -> bool:
        return self.config.evaluator_provider.lower().replace("-", "_") not in {
            "metrics",
            "metrics_only",
            "none",
            "off",
            "disabled",
        }

    def _should_build_evaluator_evidence(self) -> bool:
        if self.config.evaluator_build_evidence is not None:
            return bool(self.config.evaluator_build_evidence)
        provider = self.config.evaluator_provider.lower().replace("-", "_")
        # Mock/schema tests should stay fast.  Real multimodal providers need
        # visual evidence; metrics-only is handled by _evaluator_enabled.
        return provider not in {"mock"}

    def _agent_scope(self) -> str:
        return str(self.config.agent_scope or "full_s2_repair").lower().replace("-", "_")

    def _is_assemble_geometry_scope(self) -> bool:
        return self._agent_scope() in {"assemble_geometry", "assembly_geometry", "assemble_only"}

    def _focused_scopes_for_iteration(self, iteration: int) -> Optional[List[str]]:
        if self.config.focused_critic_scopes:
            return self.config.focused_critic_scopes
        if self._is_assemble_geometry_scope():
            if self._assemble_geometry_phase == "global_assembly_geometry_review":
                return ["scene_assembly"]
            if self.config.sequential_part_gate:
                return [self._sequential_part_stage]
            return ["part_image", "part_mesh"]
        return None

    def _focused_iteration_phase(self, iteration: int) -> str:
        if self._is_assemble_geometry_scope():
            return self._assemble_geometry_phase
        return "full_s2_focused_review"

    @staticmethod
    def _is_per_part_repair_finding(finding: S2Finding) -> bool:
        return str(finding.failure_stage or "") in {"t2i", "geometry"}

    @staticmethod
    def _is_global_assembly_finding(finding: S2Finding) -> bool:
        return str(finding.failure_stage or "") in {"assembly_repair", "bbox_layout_mutation"}

    def _findings_for_active_assemble_phase(self, findings: Sequence[S2Finding]) -> List[S2Finding]:
        if not self._is_assemble_geometry_scope():
            return list(findings)
        if self._assemble_geometry_phase == "per_part_input_and_mesh_precheck":
            if self.config.sequential_part_gate:
                return self._sequential_active_part_findings(findings)
            return [finding for finding in findings if self._is_per_part_repair_finding(finding)]
        return list(findings)

    def _sequential_active_part_findings(self, findings: Sequence[S2Finding]) -> List[S2Finding]:
        part_id = self._current_sequential_part_id()
        if not part_id:
            return []
        target_stage = "t2i" if self._sequential_part_stage == "part_image" else "geometry"
        return [
            finding
            for finding in findings
            if str(finding.failure_stage or "") == target_stage and str(part_id) in {str(item) for item in finding.part_ids}
        ]

    def _maybe_advance_sequential_part_gate(self, *, iteration: int, findings: Sequence[S2Finding]) -> bool:
        if not self._sequential_gate_enabled():
            return False
        current_part = self._current_sequential_part_id()
        if not current_part:
            self._assemble_geometry_phase = "global_assembly_geometry_review"
            self.trace.setdefault("workflow_transitions", []).append({
                "iteration": iteration,
                "from_phase": "per_part_input_and_mesh_precheck",
                "to_phase": "global_assembly_geometry_review",
                "reason": "no_parts_available_for_sequential_gate",
                "blocking_finding_count": 0,
                "created_at": utc_now_iso(),
            })
            return True
        blocking = self._sequential_active_part_findings(findings)
        if blocking:
            self.trace.setdefault("workflow_transitions", []).append({
                "iteration": iteration,
                "from_phase": "per_part_input_and_mesh_precheck",
                "to_phase": "per_part_input_and_mesh_precheck",
                "reason": "current_part_stage_findings_remain",
                "current_part_id": current_part,
                "current_part_stage": self._sequential_part_stage,
                "blocking_finding_count": len(blocking),
                "created_at": utc_now_iso(),
            })
            return False
        previous_stage = self._sequential_part_stage
        previous_part = current_part
        if self._sequential_part_stage == "part_image":
            self._sequential_part_stage = "part_mesh"
            reason = "current_part_image_accepted_advance_to_mesh"
        else:
            self._sequential_part_index += 1
            self._sequential_part_stage = "part_image"
            reason = "current_part_mesh_accepted_advance_to_next_part"
        next_part = self._current_sequential_part_id()
        if next_part is None:
            self._assemble_geometry_phase = "global_assembly_geometry_review"
            to_phase = "global_assembly_geometry_review"
            reason = "all_parts_passed_sequential_image_and_mesh_checks"
        else:
            to_phase = "per_part_input_and_mesh_precheck"
        self.trace.setdefault("workflow_transitions", []).append({
            "iteration": iteration,
            "from_phase": "per_part_input_and_mesh_precheck",
            "to_phase": to_phase,
            "reason": reason,
            "previous_part_id": previous_part,
            "previous_part_stage": previous_stage,
            "next_part_id": next_part,
            "next_part_stage": None if next_part is None else self._sequential_part_stage,
            "blocking_finding_count": 0,
            "created_at": utc_now_iso(),
        })
        return next_part is None

    def _maybe_advance_assemble_geometry_phase(self, *, iteration: int, findings: Sequence[S2Finding]) -> bool:
        """Advance from per-part checks to global assembly only after all parts pass.

        The Assemble Geometry agent is a state machine, not iteration-number
        driven: iteration 0, 1, ... stay in per-part T2I/mesh review as long as
        any part-level T2I or I3D/mesh issue remains.  Global scene assembly is
        reached only after those blocking findings disappear.
        """
        if not self._is_assemble_geometry_scope():
            return False
        if self._assemble_geometry_phase != "per_part_input_and_mesh_precheck":
            return False
        blocking = [finding for finding in findings if self._is_per_part_repair_finding(finding)]
        if blocking:
            self.trace.setdefault("workflow_transitions", []).append({
                "iteration": iteration,
                "from_phase": self._assemble_geometry_phase,
                "to_phase": self._assemble_geometry_phase,
                "reason": "part_level_t2i_or_mesh_findings_remain",
                "blocking_finding_count": len(blocking),
                "created_at": utc_now_iso(),
            })
            return False
        self.trace.setdefault("workflow_transitions", []).append({
            "iteration": iteration,
            "from_phase": self._assemble_geometry_phase,
            "to_phase": "global_assembly_geometry_review",
            "reason": "all_part_image_and_part_mesh_checks_passed",
            "blocking_finding_count": 0,
            "created_at": utc_now_iso(),
        })
        self._assemble_geometry_phase = "global_assembly_geometry_review"
        return True

    def _write_trace(self) -> str:
        path = self.agent_dir / "agent_trace.json"
        write_json(path, self.trace)
        self._write_html_report()
        # 9999 serves report/index.html, not agent_loop/index.html.  Treat each
        # trace write as a safe post-experiment hook so the main page always
        # reflects the latest agent step/input/output without waiting for a
        # full report rebuild.
        if self.config.force_report and not self.config.dry_run:
            try:
                refresh_result = refresh_s2_9999_report(
                    self.run_dir,
                    full=False,
                    trigger="agent_trace_write",
                )
                self.trace["last_9999_refresh"] = refresh_result
                write_json(path, self.trace)
            except Exception as exc:  # noqa: BLE001 - trace writing must not fail because report refresh failed.
                self.trace.setdefault("report_errors", []).append(repr(exc))
                write_json(path, self.trace)
        return str(path)

    def _write_iteration_report(self, report: S2ValidationReport) -> str:
        path = self.agent_dir / f"iteration_{report.iteration:03d}_validation.json"
        write_json(path, report.to_dict())
        return str(path)

    def _build_iteration_evidence(self, iteration: int) -> Dict[str, Any]:
        quality_dir = self.run_dir / "quality" / "v9_agent"
        quality_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = quality_dir / f"iteration_{iteration:03d}"
        if self.config.evaluator_fast_evidence:
            return build_report_asset_evidence_pack(self.run_dir, evidence_dir, self.config.texture_run_dir)
        return build_evidence_pack(self.run_dir, evidence_dir, self.config.texture_run_dir)

    def _stage_record(
        self,
        *,
        stage: str,
        status: str,
        iteration: int,
        part_ids: Sequence[str] = (),
        operation: Optional[S2RepairOperation] = None,
        failure_stage: Optional[str] = None,
        primary_artifact: Optional[str | Path] = None,
        inputs: Optional[Mapping[str, Any]] = None,
        outputs: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "stage": stage,
            "status": status,
            "iteration": int(iteration),
            "part_ids": [str(part_id) for part_id in part_ids],
            "operation_id": operation.operation_id if operation else None,
            "operation_type": operation.operation_type if operation else None,
            "failure_stage": failure_stage or (operation.details.get("failure_stage") if operation else None),
            "primary_artifact": str(primary_artifact) if primary_artifact else None,
            "inputs": dict(inputs or {}),
            "outputs": dict(outputs or {}),
            "error": error,
            "created_at": utc_now_iso(),
        }

    def _run_visual_evaluator(
        self,
        iteration: int,
        metrics: Mapping[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
        if not self._evaluator_enabled():
            return None, None, None, None
        evaluator_mode = str(self.config.evaluator_mode or "focused").lower().replace("-", "_")
        provider_name = str(self.config.evaluator_provider or "").lower().replace("-", "_")
        if evaluator_mode == "focused" and (provider_name not in {"mock"} or self.config.evaluator_mock_response is None):
            return self._run_focused_visual_evaluator(iteration, metrics)
        evidence = None
        interaction_dir = self.agent_dir / "vlm_interactions" / f"iteration_{iteration:03d}"
        request_manifest = None
        response_manifest = None
        try:
            if self._should_build_evaluator_evidence():
                evidence = self._build_iteration_evidence(iteration)
            else:
                # Mock evaluators still need a human-readable question and, if
                # available, reused report images so the agent process is
                # inspectable without paying a heavy render cost.
                evidence = build_report_asset_evidence_pack(
                    self.run_dir,
                    self.run_dir / "quality" / "v9_agent" / f"iteration_{iteration:03d}",
                    self.config.texture_run_dir,
                )
            request_manifest = write_visual_evaluator_request_artifacts(
                metrics,
                evidence,
                interaction_dir,
                max_images=self.config.evaluator_max_images,
            )
            findings = evaluate_quality(
                metrics,
                evidence,
                provider=self.config.evaluator_provider,
                mock_response=self.config.evaluator_mock_response,
                model=self.config.evaluator_model,
                base_url=self.config.evaluator_base_url,
                auth_path=self.config.evaluator_auth_path,
                max_images=self.config.evaluator_max_images,
            )
            response_manifest = write_visual_evaluator_response_artifacts(
                interaction_dir,
                findings=findings,
            )
            visual_critic = write_s2_visual_critic_sidecars(
                run_dir=self.run_dir,
                iteration=iteration,
                provider=self.config.evaluator_provider,
                model=self.config.evaluator_model,
                base_url=self.config.evaluator_base_url,
                request_manifest=request_manifest,
                response_manifest=response_manifest,
                metrics=metrics,
                evidence=evidence,
                output_dir=interaction_dir,
            )
            return findings, evidence, None, {
                "interaction_dir": str(interaction_dir),
                "request": request_manifest,
                "response": response_manifest,
                "question_prompt_path": str(interaction_dir / "question_prompt.txt"),
                "answer_path": str(interaction_dir / "answer.json"),
                "visual_critic": visual_critic,
            }
        except Exception as exc:  # noqa: BLE001 - VLM failures should be traceable.
            error = repr(exc)
            if self.config.evaluator_required:
                raise
            response_manifest = write_visual_evaluator_response_artifacts(
                interaction_dir,
                findings=None,
                error=error,
            )
            visual_critic = write_s2_visual_critic_sidecars(
                run_dir=self.run_dir,
                iteration=iteration,
                provider=self.config.evaluator_provider,
                model=self.config.evaluator_model,
                base_url=self.config.evaluator_base_url,
                request_manifest=request_manifest,
                response_manifest=response_manifest,
                metrics=metrics,
                evidence=evidence,
                output_dir=interaction_dir,
            )
            self.trace.setdefault("evaluator_errors", []).append(
                {
                    "iteration": iteration,
                    "provider": self.config.evaluator_provider,
                    "error": error,
                    "created_at": utc_now_iso(),
                }
            )
            return None, evidence, error, {
                "interaction_dir": str(interaction_dir),
                "request": request_manifest,
                "response": response_manifest,
                "question_prompt_path": str(interaction_dir / "question_prompt.txt"),
                "answer_path": str(interaction_dir / "answer.json"),
                "visual_critic": visual_critic,
            }

    def _run_focused_visual_evaluator(
        self,
        iteration: int,
        metrics: Mapping[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
        """Run stage-specific VLM critics instead of an omnibus image prompt.

        Each call asks exactly one question/scope:
        - scene_assembly: several views of the assembled mesh only.
        - part_image: one T2I reference image for one part.
        - part_mesh: one part mesh render, optionally with that part's reference
          image and bbox render as local context.
        """
        interaction_dir = self.agent_dir / "focused_vlm_interactions" / f"iteration_{iteration:03d}"
        interaction_dir.mkdir(parents=True, exist_ok=True)
        try:
            current_part_id = self._current_sequential_part_id() if self._sequential_gate_enabled() else None
            focused = run_focused_visual_critics(
                self.run_dir,
                provider=self.config.evaluator_provider,
                model=self.config.evaluator_model or "gpt-5.5",
                base_url=self.config.evaluator_base_url,
                auth_path=self.config.evaluator_auth_path,
                output_dir=interaction_dir,
                max_parts=self.config.focused_critic_max_parts,
                part_ids=[current_part_id] if current_part_id else None,
                scopes=self._focused_scopes_for_iteration(iteration),
            )
            failed_cases = [
                result for result in focused.get("results", []) or []
                if isinstance(result, Mapping) and result.get("status") != "completed"
            ]
            if failed_cases and self.config.evaluator_required:
                details = "; ".join(
                    f"{case.get('case_id')}: {case.get('error') or case.get('status')}"
                    for case in failed_cases[:5]
                )
                raise RuntimeError(f"focused VLM evaluator has failed cases after retries: {details}")
            valid_part_ids = [str(part.get("part_id")) for part in metrics.get("parts", []) or []]
            findings = quality_findings_from_focused_visual_critics(
                focused,
                valid_part_ids=valid_part_ids,
                score_threshold=self.config.focused_critic_score_threshold,
            )
            response_manifest = write_visual_evaluator_response_artifacts(
                interaction_dir,
                findings=findings,
            )
            summary_path = interaction_dir / "focused_visual_critics_summary.json"
            return findings, None, None, {
                "mode": "focused",
                "interaction_dir": str(interaction_dir),
                "focused_summary": focused,
                "focused_summary_path": str(summary_path),
                "current_part_id": current_part_id,
                "current_part_stage": self._sequential_part_stage if self._sequential_gate_enabled() else None,
                "response": response_manifest,
                "answer_path": str(interaction_dir / "answer.json"),
                "question_prompt_path": None,
                "request": {
                    "schema_version": "archstudio_s2.focused_visual_evaluator_request_set.v1",
                    "questions": [
                        str((load_json(result["request_path"]).get("prompt") or {}).get("question"))
                        for result in focused.get("results", []) or []
                        if result.get("request_path") and Path(str(result.get("request_path"))).exists()
                    ],
                    "input_images": [
                        image
                        for result in focused.get("results", []) or []
                        for image in (result.get("input_images") or [])
                    ],
                },
                "visual_critic": {
                    "qa_path": str(summary_path),
                    "repair_plan_path": None,
                    "critic_board_path": None,
                },
            }
        except Exception as exc:  # noqa: BLE001 - VLM failures should be traceable.
            error = repr(exc)
            if self.config.evaluator_required:
                raise
            response_manifest = write_visual_evaluator_response_artifacts(
                interaction_dir,
                findings=None,
                error=error,
            )
            self.trace.setdefault("evaluator_errors", []).append(
                {
                    "iteration": iteration,
                    "provider": self.config.evaluator_provider,
                    "mode": "focused",
                    "error": error,
                    "created_at": utc_now_iso(),
                }
            )
            return None, None, error, {
                "mode": "focused",
                "interaction_dir": str(interaction_dir),
                "answer_path": str(interaction_dir / "answer.json"),
                "response": response_manifest,
                "request": {"questions": [], "input_images": []},
                "visual_critic": {},
            }

    def _write_html_report(self) -> str:
        def rel_link(path: Any, label: str | None = None) -> str:
            if not path:
                return ""
            try:
                rel = os.path.relpath(str(path), str(self.agent_dir))
            except Exception:
                rel = str(path)
            return f'<a href="{html.escape(rel)}">{html.escape(label or os.path.basename(str(path)) or str(path))}</a>'

        def load_json_optional(path: Any) -> Dict[str, Any]:
            if not path:
                return {}
            try:
                candidate = Path(str(path))
                if candidate.exists():
                    payload = load_json(candidate)
                    return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}
            return {}

        def text_optional(path: Any, limit: int = 5000) -> str:
            if not path:
                return ""
            try:
                candidate = Path(str(path))
                if candidate.exists():
                    return candidate.read_text(encoding="utf-8", errors="ignore")[:limit]
            except Exception:
                return ""
            return ""

        def image_tag(path: Any, caption: str, *, board: bool = False) -> str:
            if not path:
                return ""
            try:
                rel = os.path.relpath(str(path), str(self.agent_dir))
            except Exception:
                rel = str(path)
            css = "critic-board-img" if board else "critic-thumb-img"
            return (
                "<figure class='critic-figure'>"
                f"<a href='{html.escape(rel)}' target='_blank' rel='noopener' title='Open full-size evidence image'>"
                f"<img class='{css}' src='{html.escape(rel)}'>"
                "</a>"
                f"<figcaption>{html.escape(str(caption))} · click to open full size</figcaption>"
                "</figure>"
            )

        def render_questions(question_text: str, questions: Sequence[Any]) -> str:
            rows = [f"<li>{html.escape(str(question))}</li>" for question in list(questions or [])[:8]]
            if rows:
                return "<ol class='critic-questions'>" + "".join(rows) + "</ol>"
            if question_text:
                return f"<pre class='critic-prompt'>{html.escape(question_text[:4000])}</pre>"
            return "<p>No question prompt recorded.</p>"

        def render_visual_critic_panel(iteration: Mapping[str, Any], next_iteration: Optional[Mapping[str, Any]]) -> str:
            evaluator = iteration.get("evaluator") or {}
            if evaluator.get("mode") == "focused":
                return render_focused_critic_panel(iteration, next_iteration)
            qa = load_json_optional(evaluator.get("visual_critic_qa_path"))
            plan = load_json_optional(evaluator.get("visual_critic_repair_plan_path"))
            answer_payload = load_json_optional(evaluator.get("answer_path"))
            question_text = text_optional(evaluator.get("question_prompt_path"))
            board = evaluator.get("critic_board_path")
            if not board and qa.get("image"):
                candidate = self.run_dir / str(qa.get("image"))
                board = str(candidate) if candidate.exists() else qa.get("image")
            figures = []
            if board:
                figures.append(image_tag(board, "S2 visual critic board", board=True))
            if not figures:
                for image in (evaluator.get("input_images") or [])[:8]:
                    path = image.get("copied_path") or image.get("source_path")
                    if path:
                        figures.append(image_tag(path, image.get("label") or image.get("index")))
            findings = answer_payload.get("findings") if isinstance(answer_payload.get("findings"), dict) else {}
            qa_answer = qa.get("answer") if isinstance(qa.get("answer"), dict) else {}
            issues = findings.get("issues") if isinstance(findings.get("issues"), list) else qa_answer.get("findings", [])
            issue_rows = []
            for issue in (issues or [])[:16]:
                if not isinstance(issue, Mapping):
                    continue
                issue_rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(issue.get('issue_id') or issue.get('issue_type') or 'issue'))}</td>"
                    f"<td>{html.escape(str(issue.get('severity', '')))}</td>"
                    f"<td>{html.escape(', '.join(str(item) for item in (issue.get('labels') or issue.get('part_numbers') or [])))}</td>"
                    f"<td>{html.escape(str(issue.get('issue_type', '')))}</td>"
                    f"<td>{html.escape(str(issue.get('diagnosis') or issue.get('answer') or '')[:520])}</td>"
                    f"<td>{html.escape(str(issue.get('recommended_action') or ''))}</td>"
                    "</tr>"
                )
            repair_rows = []
            repairs = plan.get("recommended_repairs") if isinstance(plan.get("recommended_repairs"), list) else iteration.get("operations", [])
            for repair in (repairs or [])[:16]:
                if not isinstance(repair, Mapping):
                    continue
                repair_rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(repair.get('action') or repair.get('operation_type') or ''))}</td>"
                    f"<td>{html.escape(str(repair.get('status', 'proposed')))}</td>"
                    f"<td>{html.escape(', '.join(str(item) for item in (repair.get('part_numbers') or repair.get('labels') or [])))}</td>"
                    f"<td>{html.escape(', '.join(str(item) for item in (repair.get('part_ids') or []))[:260])}</td>"
                    f"<td>{html.escape(str(repair.get('instruction') or repair.get('repair_instruction') or repair.get('rationale') or '')[:520])}</td>"
                    "</tr>"
                )
            questions = evaluator.get("visual_critic_questions") or qa.get("questions") or []
            summary_text = findings.get("global_summary") or qa_answer.get("summary") or evaluator.get("status") or ""
            return (
                "<section class='critic-card'>"
                "<div class='critic-head'>"
                f"<div><div class='eyebrow'>Iteration {html.escape(str(iteration.get('iteration')))} · {html.escape(str(evaluator.get('provider', '')))} / {html.escape(str(evaluator.get('model', '')))}</div>"
                "<h3>Visual Critic QA</h3></div>"
                f"<div class='critic-score'><span>score {html.escape(str(iteration.get('scene_score')))}</span><span>next {html.escape(str((next_iteration or {}).get('scene_score', '—')))}</span><span>{html.escape(str(evaluator.get('status', '')))}</span></div>"
                "</div>"
                "<div class='critic-grid'>"
                f"<div>{''.join(figures) if figures else '<p>No critic images recorded.</p>'}</div>"
                "<div class='critic-copy'>"
                "<h4>Question prompt</h4>"
                f"{render_questions(question_text, questions)}"
                "<h4>Answer summary</h4>"
                f"<p>{html.escape(str(summary_text or 'No answer summary recorded.'))}</p>"
                "<div class='links'>"
                f"{rel_link(evaluator.get('question_prompt_path'), 'question_prompt.txt')} "
                f"{rel_link(evaluator.get('answer_path'), 'answer.json')} "
                f"{rel_link(evaluator.get('visual_critic_qa_path'), 'visual_critic_qa.json')} "
                f"{rel_link(evaluator.get('visual_critic_repair_plan_path'), 'visual_critic_repair_plan.json')}"
                "</div>"
                "</div></div>"
                "<details open><summary>Findings / reject reasons</summary>"
                "<table><thead><tr><th>id</th><th>sev</th><th>#</th><th>type</th><th>diagnosis</th><th>action</th></tr></thead>"
                f"<tbody>{''.join(issue_rows) if issue_rows else '<tr><td colspan=\"6\">No visual findings recorded.</td></tr>'}</tbody></table></details>"
                "<details open><summary>Repair plan / operations</summary>"
                "<table><thead><tr><th>action</th><th>status</th><th>#</th><th>part ids</th><th>instruction</th></tr></thead>"
                f"<tbody>{''.join(repair_rows) if repair_rows else '<tr><td colspan=\"5\">No repair plan recorded.</td></tr>'}</tbody></table></details>"
                "</section>"
            )

        def render_focused_critic_panel(iteration: Mapping[str, Any], next_iteration: Optional[Mapping[str, Any]]) -> str:
            evaluator = iteration.get("evaluator") or {}
            focused = load_json_optional(evaluator.get("focused_summary_path"))
            answer_payload = load_json_optional(evaluator.get("answer_path"))
            findings = answer_payload.get("findings") if isinstance(answer_payload.get("findings"), dict) else {}
            case_cards = []
            for result in (focused.get("results") or []):
                if not isinstance(result, Mapping):
                    continue
                request = load_json_optional(result.get("request_path"))
                response = load_json_optional(result.get("answer_path"))
                prompt = request.get("prompt") if isinstance(request.get("prompt"), dict) else {}
                answer = response.get("answer") if isinstance(response.get("answer"), dict) else {}
                figures = []
                for image in result.get("input_images", []) or []:
                    path = image.get("copied_path") or image.get("source_path")
                    if path:
                        figures.append(image_tag(path, image.get("label") or image.get("index")))
                local_issues = []
                for issue in answer.get("issues", []) if isinstance(answer.get("issues"), list) else []:
                    if not isinstance(issue, Mapping):
                        continue
                    local_issues.append(
                        "<li>"
                        f"<b>{html.escape(str(issue.get('issue_type', 'issue')))}</b> "
                        f"sev={html.escape(str(issue.get('severity', '')))} · "
                        f"{html.escape(str(issue.get('evidence', ''))[:260])} · "
                        f"<i>{html.escape(str(issue.get('repair_hint', issue.get('repair_instruction', '')))[:320])}</i>"
                        "</li>"
                    )
                case_cards.append(
                    "<article class='focused-case'>"
                    "<div class='critic-head'>"
                    f"<div><div class='eyebrow'>{html.escape(str(result.get('scope', '')))} · {html.escape(str(result.get('case_id', '')))}</div>"
                    f"<h3>{html.escape(str(result.get('part_id') or 'scene assembly'))}</h3></div>"
                    f"<div class='critic-score'><span>score {html.escape(str(answer.get('score', result.get('score', ''))))}</span>"
                    f"<span>{html.escape(str(answer.get('verdict', result.get('verdict', ''))))}</span>"
                    f"<span>{html.escape(str(result.get('status', '')))}</span></div>"
                    "</div>"
                    f"<p><b>Question:</b> {html.escape(str(prompt.get('question', '')))}</p>"
                    f"<p><b>Do not judge:</b> {html.escape(', '.join(str(item) for item in (prompt.get('do_not_judge') or [])))}</p>"
                    f"<p><b>Answer:</b> {html.escape(str(answer.get('summary') or result.get('summary') or ''))}</p>"
                    f"<div class='focused-images'>{''.join(figures) if figures else '<p>No input images recorded for this focused case.</p>'}</div>"
                    f"<ul>{''.join(local_issues) if local_issues else '<li>No local focused issues.</li>'}</ul>"
                    "<div class='links'>"
                    f"{rel_link(result.get('request_path'), 'request.json')} "
                    f"{rel_link(result.get('answer_path'), 'answer.json')}"
                    "</div>"
                    "</article>"
                )
            issue_rows = []
            for issue in findings.get("issues", []) if isinstance(findings.get("issues"), list) else []:
                if not isinstance(issue, Mapping):
                    continue
                issue_rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(issue.get('issue_id', '')))}</td>"
                    f"<td>{html.escape(str(issue.get('severity', '')))}</td>"
                    f"<td>{html.escape(', '.join(str(item) for item in (issue.get('labels') or [])))}</td>"
                    f"<td>{html.escape(str(issue.get('issue_type', '')))}</td>"
                    f"<td>{html.escape(str(issue.get('diagnosis', ''))[:520])}</td>"
                    f"<td>{html.escape(str(issue.get('recommended_action') or ''))}</td>"
                    "</tr>"
                )
            return (
                "<section class='critic-card focused-critic-card'>"
                "<div class='critic-head'>"
                f"<div><div class='eyebrow'>Iteration {html.escape(str(iteration.get('iteration')))} · {html.escape(str(evaluator.get('agent_phase', 'focused')))} · focused visual critics · {html.escape(str(evaluator.get('provider', '')))} / {html.escape(str(evaluator.get('model', '')))}</div>"
                "<h3>Focused Visual Critic QA</h3></div>"
                f"<div class='critic-score'><span>score {html.escape(str(iteration.get('scene_score')))}</span><span>next {html.escape(str((next_iteration or {}).get('scene_score', '—')))}</span><span>{html.escape(str(evaluator.get('status', '')))}</span></div>"
                "</div>"
                "<p>每个卡片只审查一个问题：assemble_geometry 的 iteration 0 先逐个 part 检查 T2I 图和单体 mesh；后续 iteration 才看整体 scene_assembly。part_image 只看单个 T2I 图；part_mesh 只看单个 mesh（可附本 part 的 reference 和 bbox 作为局部约束）。不再把所有 layout / mesh / reference 全部 cat 到一次 VLM。</p>"
                f"{''.join(case_cards) if case_cards else '<p>No focused critic cases recorded.</p>'}"
                "<details open><summary>Converted reject/repair findings</summary>"
                "<table><thead><tr><th>id</th><th>sev</th><th>#</th><th>type</th><th>diagnosis</th><th>action</th></tr></thead>"
                f"<tbody>{''.join(issue_rows) if issue_rows else '<tr><td colspan=\"6\">No converted findings.</td></tr>'}</tbody></table></details>"
                "<div class='links'>"
                f"{rel_link(evaluator.get('focused_summary_path'), 'focused_visual_critics_summary.json')} "
                f"{rel_link(evaluator.get('answer_path'), 'converted_answer.json')}"
                "</div>"
                "</section>"
            )

        score_rows = []
        for point in self.trace.get("score_trajectory", []) or []:
            score_rows.append(
                "<tr>"
                f"<td>{html.escape(str(point.get('iteration', '')))}</td>"
                f"<td>{html.escape(str(point.get('scene_score', '')))}</td>"
                f"<td>{html.escape(str(point.get('delta', '')))}</td>"
                f"<td>{html.escape(str(point.get('no_improvement_count', '')))}</td>"
                "</tr>"
            )

        rows = []
        stage_blocks = []
        critic_blocks = []
        iterations = self.trace.get("iterations", [])
        for iteration_index, iteration in enumerate(iterations):
            ops = iteration.get("operations", []) or []
            op_text = "<br>".join(
                f"{op.get('operation_id')} · {op.get('operation_type')} · {op.get('status')} · {', '.join(op.get('part_ids', []))}"
                for op in ops
            ) or "none"
            stage_records = iteration.get("stage_records", []) or []
            stage_text = "<br>".join(
                "{} · {} · {} · {}".format(
                    html.escape(str(stage.get("stage", ""))),
                    html.escape(str(stage.get("status", ""))),
                    html.escape(str(stage.get("failure_stage") or "")),
                    rel_link(stage.get("primary_artifact"), "artifact") if stage.get("primary_artifact") else "",
                )
                for stage in stage_records
            ) or "none"
            rows.append(
                "<tr>"
                f"<td>{iteration.get('iteration')}</td>"
                f"<td>{iteration.get('accepted')}</td>"
                f"<td>{iteration.get('scene_score')}</td>"
                f"<td>{iteration.get('num_findings')}</td>"
                f"<td>{op_text}</td>"
                f"<td>{stage_text}</td>"
                "</tr>"
            )
            finding_items = []
            for finding in iteration.get("findings", []) or []:
                finding_items.append(
                    "<li>"
                    f"<b>{html.escape(str(finding.get('issue_type', '')))}</b> "
                    f"[{html.escape(str(finding.get('failure_stage', '')))}] "
                    f"parts={html.escape(', '.join(str(item) for item in finding.get('part_ids', []) or []))}: "
                    f"{html.escape(str(finding.get('diagnosis', ''))[:500])}"
                    "</li>"
                )
            stage_blocks.append(
                "<details>"
                f"<summary>Iteration {html.escape(str(iteration.get('iteration')))} details · "
                f"{html.escape(str(iteration.get('num_findings')))} findings · "
                f"{html.escape(str(len(stage_records)))} stages</summary>"
                f"<p>Validation: {rel_link(iteration.get('validation_report_path'), 'validation JSON')} · "
                f"Evidence: {html.escape(str((iteration.get('evaluator') or {}).get('evidence_dir') or 'none'))} · "
                f"Question: {rel_link((iteration.get('evaluator') or {}).get('question_prompt_path'), 'question_prompt.txt')} · "
                f"Answer: {rel_link((iteration.get('evaluator') or {}).get('answer_path'), 'answer.json')}</p>"
                f"{self._render_interaction_images(iteration)}"
                f"<ul>{''.join(finding_items) if finding_items else '<li>No findings</li>'}</ul>"
                "</details>"
            )
            next_iteration = iterations[iteration_index + 1] if iteration_index + 1 < len(iterations) else None
            critic_blocks.append(render_visual_critic_panel(iteration, next_iteration))
        taxonomy_rows = []
        for stage, count in sorted((self.trace.get("failure_taxonomy") or {}).items()):
            taxonomy_rows.append(
                f"<tr><td>{html.escape(str(stage))}</td><td>{html.escape(str(count))}</td></tr>"
            )
        html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ArchStudio-S2 Reject/Accept Agent</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background: #f7f8fb; color: #1f2630; }}
    .panel {{ background: white; border: 1px solid #d8e0ef; border-radius: 12px; padding: 16px; margin: 14px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d8e0ef; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: #44608a; text-transform: uppercase; letter-spacing: .04em; font-size: 12px; }}
    code {{ font-size: 12px; }}
    a {{ color: #174ea6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
	    .kpi {{ background: #fff; border: 1px solid #d8e0ef; border-radius: 10px; padding: 12px; }}
	    .kpi b {{ display: block; color: #44608a; font-size: 12px; text-transform: uppercase; }}
	    .kpi span {{ font-size: 22px; font-weight: 800; }}
	    .critic-card {{ background: #fffdf7; border: 1px solid #d6c4a2; border-radius: 16px; padding: 16px; margin: 16px 0; }}
	    .critic-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
	    .critic-head h3 {{ margin: 4px 0 0; font-size: 24px; }}
	    .eyebrow {{ color: #755b27; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; }}
	    .critic-score {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
	    .critic-score span {{ background: #223145; color: white; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}
	    .critic-grid {{ display: grid; grid-template-columns: minmax(460px, 1.35fr) minmax(320px, .85fr); gap: 16px; align-items: start; }}
	    .critic-figure {{ margin: 10px 0; }}
	    .critic-board-img {{ width: 100%; max-height: 760px; object-fit: contain; background: #151922; border-radius: 12px; border: 1px solid #222; }}
	    .critic-thumb-img {{ width: 220px; height: 150px; object-fit: contain; background: #151922; border-radius: 8px; border: 1px solid #222; }}
	    .critic-figure figcaption {{ font-size: 12px; color: #665c4d; margin-top: 6px; }}
	    .critic-copy {{ background: #f7f0df; border: 1px solid #ddcda9; border-radius: 12px; padding: 14px; }}
	    .critic-copy h4 {{ margin: 0 0 8px; color: #4f3f24; }}
	    .critic-questions {{ margin: 0 0 14px 20px; padding: 0; font-size: 13px; }}
	    .critic-questions li {{ margin-bottom: 7px; }}
	    .critic-prompt {{ max-height: 360px; overflow: auto; white-space: pre-wrap; background: #1e222b; color: #f4efe5; border-radius: 10px; padding: 12px; font-size: 11px; }}
	    .focused-case {{ border: 1px solid #ead8b3; background: #fffaf0; border-radius: 12px; padding: 12px; margin: 12px 0; }}
	    .focused-images {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }}
	    .focused-critic-card .critic-thumb-img {{ width: 300px; height: 220px; }}
	    .links {{ display: flex; flex-wrap: wrap; gap: 10px; }}
	    @media (max-width: 900px) {{ .critic-grid {{ grid-template-columns: 1fr; }} .critic-head {{ flex-direction: column; }} }}
	  </style>
</head>
<body>
  <h1>ArchStudio-S2 Geometry-Only VLM Repair Agent</h1>
  <div class="panel">
    <p><b>Status:</b> {self.trace.get('status')} · <b>Accepted:</b> {self.trace.get('accepted')}</p>
    <p><b>Stop condition:</b> {self.trace.get('stop_condition', 'running')}</p>
    <p><b>Run:</b> <code>{self.run_dir}</code></p>
    <p><a href="agent_trace.json">Download agent_trace.json</a> · <a href="../report/index.html">Open S2 report</a></p>
  </div>
  <div class="grid">
    <div class="kpi"><b>Iterations</b><span>{len(self.trace.get('iterations', []) or [])}</span></div>
    <div class="kpi"><b>Final score</b><span>{self.trace.get('final_scene_score', '')}</span></div>
    <div class="kpi"><b>No improvement</b><span>{self.trace.get('no_improvement_count', 0)}</span></div>
    <div class="kpi"><b>Failure stages</b><span>{len(self.trace.get('failure_taxonomy', {}) or {})}</span></div>
  </div>
  <div class="panel">
    <h2>Score trajectory / auto-stop</h2>
    <table><thead><tr><th>iter</th><th>score</th><th>delta</th><th>no improvement count</th></tr></thead>
    <tbody>{''.join(score_rows) if score_rows else '<tr><td colspan="4">No score data yet.</td></tr>'}</tbody></table>
  </div>
  <div class="panel">
    <h2>Failure taxonomy</h2>
    <table><thead><tr><th>stage</th><th>count</th></tr></thead>
    <tbody>{''.join(taxonomy_rows) if taxonomy_rows else '<tr><td colspan="2">No unresolved failures.</td></tr>'}</tbody></table>
  </div>
	  <div class="panel">
	    <h2>S1-style Visual Critic QA</h2>
	    {''.join(critic_blocks) if critic_blocks else '<p>No visual critic QA yet.</p>'}
	  </div>
	  <div class="panel">
	    <h2>Iterations</h2>
    <table><thead><tr><th>iter</th><th>accepted</th><th>score</th><th>findings</th><th>operations</th><th>stages</th></tr></thead>
    <tbody>{''.join(rows) if rows else '<tr><td colspan="6">No iterations yet.</td></tr>'}</tbody></table>
  </div>
  <div class="panel">
    <h2>Explainable VLM / repair details</h2>
    {''.join(stage_blocks) if stage_blocks else '<p>No iteration details yet.</p>'}
  </div>
</body>
</html>
"""
        path = self.agent_dir / "index.html"
        path.write_text(html_doc, encoding="utf-8")
        return str(path)

    def _render_interaction_images(self, iteration: Mapping[str, Any]) -> str:
        evaluator = iteration.get("evaluator") or {}
        images = evaluator.get("input_images") or []
        if not images:
            return "<p><i>No VLM input images recorded.</i></p>"
        cells = []
        for image in images[:12]:
            path = image.get("copied_path") or image.get("source_path")
            label = image.get("label") or image.get("index")
            if not path:
                continue
            try:
                rel = os.path.relpath(str(path), str(self.agent_dir))
            except Exception:
                rel = str(path)
            cells.append(
                "<figure style='margin:0'>"
                f"<img src='{html.escape(rel)}' style='width:180px;height:130px;object-fit:contain;background:#fff;border:1px solid #ddd;border-radius:8px'>"
                f"<figcaption style='font-size:11px;color:#555'>{html.escape(str(label))}</figcaption>"
                "</figure>"
            )
        if not cells:
            return "<p><i>No readable VLM input images recorded.</i></p>"
        return "<div style='display:flex;flex-wrap:wrap;gap:10px;margin:10px 0'>" + "".join(cells) + "</div>"

    def _execute_operation(self, operation: S2RepairOperation, iteration: int) -> Tuple[S2RepairOperation, List[Dict[str, Any]]]:
        operation.started_at = utc_now_iso()
        stage_records: List[Dict[str, Any]] = []
        try:
            if operation.operation_type == "deterministic_resize_snap":
                operation.details["execution"] = apply_snap_resize_operation(
                    self.run_dir,
                    operation.part_ids,
                    iteration=iteration,
                    dry_run=self.config.dry_run,
                )
                operation.status = "dry_run" if self.config.dry_run else "succeeded"
                stage_records.append(self._stage_record(
                    stage="assembly_repair",
                    status=operation.status,
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    primary_artifact=self.run_dir / "assemblies" / "placeholder_filled_assembly.obj",
                    outputs={"execution": operation.details["execution"]},
                ))
            elif operation.operation_type == "rerun_t2i":
                snapshot_dir = self.agent_dir / "artifact_snapshots" / f"iteration_{iteration:03d}" / _safe_file_component(operation.operation_id)
                before_snapshots = {str(part_id): _part_artifact_snapshot(self.run_dir, str(part_id), freeze_dir=snapshot_dir / "before" / _safe_file_component(str(part_id)), snapshot_label="before") for part_id in operation.part_ids}
                prompt_revision = write_prompt_revisions_for_operation(
                    self.run_dir,
                    operation,
                    iteration=iteration,
                    dry_run=self.config.dry_run,
                )
                operation.details["prompt_revision"] = prompt_revision
                prompt_revision_outputs = dict(prompt_revision)
                prompt_revision_outputs["before_artifacts"] = before_snapshots
                stage_records.append(self._stage_record(
                    stage="prompt_repair",
                    status=prompt_revision.get("status", "unknown"),
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="t2i",
                    primary_artifact=(
                        prompt_revision.get("revisions", [{}])[0].get("prompt_revision_path")
                        if prompt_revision.get("revisions") else None
                    ),
                    outputs=prompt_revision_outputs,
                ))
                operation.details["execution"] = run_t2i_for_parts(self.config, operation.part_ids, iteration=iteration)
                operation.status = operation.details["execution"].get("status", "unknown")
                after_snapshots = {str(part_id): _part_artifact_snapshot(self.run_dir, str(part_id), freeze_dir=snapshot_dir / "after_t2i" / _safe_file_component(str(part_id)), snapshot_label="after_t2i") for part_id in operation.part_ids}
                artifact_versions = {
                    str(part_id): {
                        "before": record_agent_artifact_version(self.run_dir, iteration=iteration, operation_id=operation.operation_id, part_id=str(part_id), stage="t2i_generation", moment="before", snapshot=before_snapshots[str(part_id)]),
                        "after": record_agent_artifact_version(self.run_dir, iteration=iteration, operation_id=operation.operation_id, part_id=str(part_id), stage="t2i_generation", moment="after", snapshot=after_snapshots[str(part_id)], extra={"t2i_seed": operation.details.get("execution", {}).get("t2i_seed")}),
                    }
                    for part_id in operation.part_ids
                }
                operation.details["execution"]["before_artifacts"] = before_snapshots
                operation.details["execution"]["after_artifacts"] = after_snapshots
                operation.details["execution"]["artifact_versions"] = artifact_versions
                operation.details["execution"]["prompt_diffs"] = {
                    str(part_id): _diff_prompt_text(
                        before_snapshots.get(str(part_id), {}).get("metadata", {}).get("prompt_text", ""),
                        after_snapshots.get(str(part_id), {}).get("metadata", {}).get("prompt_text", ""),
                    )
                    for part_id in operation.part_ids
                }
                stage_records.append(self._stage_record(
                    stage="t2i_generation",
                    status=operation.status,
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="t2i",
                    outputs=operation.details["execution"],
                ))
            elif operation.operation_type == "rerun_geometry":
                snapshot_dir = self.agent_dir / "artifact_snapshots" / f"iteration_{iteration:03d}" / _safe_file_component(operation.operation_id)
                before_snapshots = {str(part_id): _part_artifact_snapshot(self.run_dir, str(part_id), freeze_dir=snapshot_dir / "before" / _safe_file_component(str(part_id)), snapshot_label="before") for part_id in operation.part_ids}
                geometry = run_geometry_for_parts(self.config, operation.part_ids, iteration=iteration)
                operation.details["execution"] = geometry
                operation.status = geometry.get("status", "unknown")
                after_snapshots = {str(part_id): _part_artifact_snapshot(self.run_dir, str(part_id), freeze_dir=snapshot_dir / "after_geometry" / _safe_file_component(str(part_id)), snapshot_label="after_geometry") for part_id in operation.part_ids}
                geometry_artifact_versions = {
                    str(part_id): {
                        "before": record_agent_artifact_version(self.run_dir, iteration=iteration, operation_id=operation.operation_id, part_id=str(part_id), stage="geometry_generation", moment="before", snapshot=before_snapshots[str(part_id)]),
                        "after": record_agent_artifact_version(self.run_dir, iteration=iteration, operation_id=operation.operation_id, part_id=str(part_id), stage="geometry_generation", moment="after", snapshot=after_snapshots[str(part_id)]),
                    }
                    for part_id in operation.part_ids
                }
                geometry["before_artifacts"] = before_snapshots
                geometry["after_artifacts"] = after_snapshots
                geometry["artifact_versions"] = geometry_artifact_versions
                stage_records.append(self._stage_record(
                    stage="geometry_generation",
                    status=operation.status,
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="geometry",
                    primary_artifact=self.run_dir / "assemblies" / "parts",
                    outputs=geometry,
                ))
            elif operation.operation_type == "rerun_t2i_geometry":
                snapshot_dir = self.agent_dir / "artifact_snapshots" / f"iteration_{iteration:03d}" / _safe_file_component(operation.operation_id)
                before_snapshots = {str(part_id): _part_artifact_snapshot(self.run_dir, str(part_id), freeze_dir=snapshot_dir / "before" / _safe_file_component(str(part_id)), snapshot_label="before") for part_id in operation.part_ids}
                prompt_revision = write_prompt_revisions_for_operation(
                    self.run_dir,
                    operation,
                    iteration=iteration,
                    dry_run=self.config.dry_run,
                )
                operation.details["prompt_revision"] = prompt_revision
                prompt_revision_outputs = dict(prompt_revision)
                prompt_revision_outputs["before_artifacts"] = before_snapshots
                stage_records.append(self._stage_record(
                    stage="prompt_repair",
                    status=prompt_revision.get("status", "unknown"),
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="t2i",
                    primary_artifact=(
                        prompt_revision.get("revisions", [{}])[0].get("prompt_revision_path")
                        if prompt_revision.get("revisions") else None
                    ),
                    outputs=prompt_revision_outputs,
                ))
                t2i = run_t2i_for_parts(self.config, operation.part_ids, iteration=iteration)
                after_t2i_snapshots = {str(part_id): _part_artifact_snapshot(self.run_dir, str(part_id), freeze_dir=snapshot_dir / "after_t2i" / _safe_file_component(str(part_id)), snapshot_label="after_t2i") for part_id in operation.part_ids}
                t2i_artifact_versions = {
                    str(part_id): {
                        "before": record_agent_artifact_version(self.run_dir, iteration=iteration, operation_id=operation.operation_id, part_id=str(part_id), stage="t2i_generation", moment="before", snapshot=before_snapshots[str(part_id)]),
                        "after": record_agent_artifact_version(self.run_dir, iteration=iteration, operation_id=operation.operation_id, part_id=str(part_id), stage="t2i_generation", moment="after", snapshot=after_t2i_snapshots[str(part_id)], extra={"t2i_seed": t2i.get("t2i_seed")}),
                    }
                    for part_id in operation.part_ids
                }
                t2i["before_artifacts"] = before_snapshots
                t2i["after_artifacts"] = after_t2i_snapshots
                t2i["artifact_versions"] = t2i_artifact_versions
                t2i["prompt_diffs"] = {
                    str(part_id): _diff_prompt_text(
                        before_snapshots.get(str(part_id), {}).get("metadata", {}).get("prompt_text", ""),
                        after_t2i_snapshots.get(str(part_id), {}).get("metadata", {}).get("prompt_text", ""),
                    )
                    for part_id in operation.part_ids
                }
                stage_records.append(self._stage_record(
                    stage="t2i_generation",
                    status=t2i.get("status", "unknown"),
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="t2i",
                    outputs=t2i,
                ))
                geometry = None
                if t2i.get("status") in {"succeeded", "dry_run"}:
                    before_geometry_snapshots = after_t2i_snapshots
                    geometry = run_geometry_for_parts(self.config, operation.part_ids, iteration=iteration)
                    after_geometry_snapshots = {str(part_id): _part_artifact_snapshot(self.run_dir, str(part_id), freeze_dir=snapshot_dir / "after_geometry" / _safe_file_component(str(part_id)), snapshot_label="after_geometry") for part_id in operation.part_ids}
                    geometry_artifact_versions = {
                        str(part_id): {
                            "before": record_agent_artifact_version(self.run_dir, iteration=iteration, operation_id=operation.operation_id, part_id=str(part_id), stage="geometry_generation", moment="before", snapshot=before_geometry_snapshots[str(part_id)]),
                            "after": record_agent_artifact_version(self.run_dir, iteration=iteration, operation_id=operation.operation_id, part_id=str(part_id), stage="geometry_generation", moment="after", snapshot=after_geometry_snapshots[str(part_id)]),
                        }
                        for part_id in operation.part_ids
                    }
                    geometry["before_artifacts"] = before_geometry_snapshots
                    geometry["after_artifacts"] = after_geometry_snapshots
                    geometry["artifact_versions"] = geometry_artifact_versions
                    stage_records.append(self._stage_record(
                        stage="geometry_generation",
                        status=geometry.get("status", "unknown"),
                        iteration=iteration,
                        part_ids=operation.part_ids,
                        operation=operation,
                        failure_stage="geometry",
                        primary_artifact=self.run_dir / "assemblies" / "parts",
                        outputs=geometry,
                    ))
                operation.details["execution"] = {"t2i": t2i, "geometry": geometry}
                if geometry is None:
                    operation.status = t2i.get("status", "failed")
                else:
                    operation.status = geometry.get("status", "unknown")
            elif operation.operation_type == "split_layout_part_faces":
                split = apply_split_operation(
                    self.run_dir,
                    operation.part_ids,
                    iteration=iteration,
                    dry_run=self.config.dry_run,
                )
                operation.details["execution"] = split
                operation.status = "dry_run" if self.config.dry_run else "succeeded"
                stage_records.append(self._stage_record(
                    stage="layout_mutation",
                    status=operation.status,
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="bbox_layout_mutation",
                    primary_artifact=self.run_dir / "contract" / "layout_mesh_contract.json",
                    outputs=split,
                ))
                child_ids = [
                    child_id
                    for action in split.get("actions", [])
                    for child_id in action.get("child_part_ids", [])
                ]
                if child_ids and self.config.split_then_generate and not self.config.dry_run:
                    t2i = run_t2i_for_parts(self.config, child_ids, iteration=iteration)
                    stage_records.append(self._stage_record(
                        stage="t2i_generation",
                        status=t2i.get("status", "unknown"),
                        iteration=iteration,
                        part_ids=child_ids,
                        operation=operation,
                        failure_stage="t2i",
                        outputs=t2i,
                    ))
                    geometry = None
                    if t2i.get("status") in {"succeeded", "dry_run"}:
                        geometry = run_geometry_for_parts(self.config, child_ids, iteration=iteration)
                        stage_records.append(self._stage_record(
                            stage="geometry_generation",
                            status=geometry.get("status", "unknown"),
                            iteration=iteration,
                            part_ids=child_ids,
                            operation=operation,
                            failure_stage="geometry",
                            outputs=geometry,
                        ))
                    operation.details["child_generation"] = {"child_part_ids": child_ids, "t2i": t2i, "geometry": geometry}
                    if geometry is not None:
                        operation.status = geometry.get("status", operation.status)
            elif operation.operation_type == "rerun_geometry" and self._is_assemble_geometry_scope():
                operation.status = "planned_external"
                operation.details["execution"] = {
                    "note": "Assemble Geometry Agent v1 routed this selected part back to I3D/geometry. GPU rerun is recorded for the upstream stage rather than launched inside assemble-only mode.",
                    "part_ids": list(operation.part_ids),
                }
                stage_records.append(self._stage_record(
                    stage="geometry_rerun_routed",
                    status=operation.status,
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="geometry",
                    outputs=operation.details["execution"],
                ))
            elif operation.operation_type == "rerun_t2i_geometry" and self._is_assemble_geometry_scope():
                prompt_revision = write_prompt_revisions_for_operation(
                    self.run_dir,
                    operation,
                    iteration=iteration,
                    dry_run=self.config.dry_run,
                )
                operation.details["prompt_revision"] = prompt_revision
                operation.details["execution"] = {
                    "note": "Assemble Geometry Agent v1 routed this selected part back to T2I+I3D. Prompt revision is persisted, but upstream generation is not launched inside assemble-only mode.",
                    "part_ids": list(operation.part_ids),
                    "prompt_revision": prompt_revision,
                }
                operation.status = "planned_external"
                stage_records.append(self._stage_record(
                    stage="t2i_i3d_rerun_routed",
                    status=operation.status,
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="t2i",
                    primary_artifact=(
                        prompt_revision.get("revisions", [{}])[0].get("prompt_revision_path")
                        if prompt_revision.get("revisions") else None
                    ),
                    outputs=operation.details["execution"],
                ))
            elif operation.operation_type == "rerun_texture":
                operation.status = "planned_external"
                operation.details["execution"] = {
                    "note": "Texture rerun is recorded here; use texture_archstudio_s2_hunyuan_paint.py for the texture stage.",
                }
                stage_records.append(self._stage_record(
                    stage="texture_skipped_out_of_scope",
                    status=operation.status,
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage="texture",
                    outputs=operation.details["execution"],
                ))
            else:
                operation.status = "manual_review"
                stage_records.append(self._stage_record(
                    stage="manual_review",
                    status=operation.status,
                    iteration=iteration,
                    part_ids=operation.part_ids,
                    operation=operation,
                    failure_stage=operation.details.get("failure_stage", "assembly_repair"),
                    outputs={"reason": "no executable operation registered"},
                ))
        except Exception as exc:  # noqa: BLE001 - repair trace must preserve failures.
            operation.status = "failed"
            operation.details["error"] = repr(exc)
            stage_records.append(self._stage_record(
                stage="operation_exception",
                status="failed",
                iteration=iteration,
                part_ids=operation.part_ids,
                operation=operation,
                failure_stage=operation.details.get("failure_stage", "assembly_repair"),
                error=repr(exc),
            ))
        operation.ended_at = utc_now_iso()
        return operation, stage_records

    def _refresh_downstream_reports(self) -> Optional[str]:
        try:
            if self.config.dry_run:
                return None
            quality_dir = self.run_dir / "quality" / "v9_agent"
            quality_dir.mkdir(parents=True, exist_ok=True)
            metrics = collect_quality_metrics(
                self.run_dir,
                self.config.texture_run_dir,
                max_sample_vertices=self.config.max_sample_vertices,
                deep_mesh_components=False,
            )
            write_json(quality_dir / "metrics_summary.json", metrics)
            if self._is_assemble_geometry_scope() and self.config.fast_report_refresh:
                # Assemble-only runs already persist their VLM input board under
                # agent_loop/focused_vlm_interactions.  Do not spend minutes
                # regenerating a second full quality evidence pack before the
                # 9999 hook can show the fresh agent trace.
                evidence = {
                    "schema_version": "archstudio_s2_quality_evidence.fast_agent_scope.v1",
                    "run_dir": str(self.run_dir),
                    "texture_run_dir": str(self.config.texture_run_dir) if self.config.texture_run_dir else None,
                    "evidence_dir": str(quality_dir / "evidence"),
                    "assets": {},
                    "source": "assemble_geometry_fast_report_refresh_reuses_agent_loop_vlm_inputs",
                }
                write_json(quality_dir / "evidence_manifest.json", evidence)
                last_report_path = None
                for item in reversed(self.trace.get("iterations", []) or []):
                    candidate = item.get("validation_report_path")
                    if candidate and Path(candidate).exists():
                        last_report_path = Path(candidate)
                        break
                last_findings = None
                if last_report_path:
                    try:
                        last_findings = load_json(last_report_path).get("evaluator_findings")
                        if last_findings:
                            write_json(quality_dir / "quality_findings.json", last_findings)
                    except Exception:
                        last_findings = None
                summary = make_quality_summary(self.run_dir, quality_dir, metrics, evidence, last_findings, None)
                summary["agent_loop_trace_path"] = str(self.agent_dir / "agent_trace.json")
                summary["agent_loop_stop_condition"] = self.trace.get("stop_condition", "running")
                write_json(quality_dir / "quality_summary.json", summary)
                write_agent_trace(self.run_dir, texture_dir=self.config.texture_run_dir)
                if self.config.force_report:
                    refresh_s2_9999_report(self.run_dir, full=False, trigger="downstream_report_refresh")
                return str(self.run_dir / "report" / "index.html")
            # Keep the post-experiment 9999 hook responsive in fast mode.
            # Full evidence rendering/export can spend minutes resampling dense
            # Hunyuan meshes after the agent trace is already written; real VLM
            # inputs are preserved under agent_loop/focused_vlm_interactions.
            if self.config.fast_report_refresh or self.config.evaluator_fast_evidence:
                evidence = build_report_asset_evidence_pack(self.run_dir, quality_dir, self.config.texture_run_dir)
            else:
                evidence = build_evidence_pack(self.run_dir, quality_dir, self.config.texture_run_dir)
            write_json(quality_dir / "evidence_manifest.json", evidence)
            last_report_path = None
            for item in reversed(self.trace.get("iterations", []) or []):
                candidate = item.get("validation_report_path")
                if candidate and Path(candidate).exists():
                    last_report_path = Path(candidate)
                    break
            last_findings = None
            if last_report_path:
                try:
                    last_findings = load_json(last_report_path).get("evaluator_findings")
                    if last_findings:
                        write_json(quality_dir / "quality_findings.json", last_findings)
                except Exception:
                    last_findings = None
            summary = make_quality_summary(self.run_dir, quality_dir, metrics, evidence, last_findings, None)
            summary["agent_loop_trace_path"] = str(self.agent_dir / "agent_trace.json")
            summary["agent_loop_stop_condition"] = self.trace.get("stop_condition", "running")
            write_json(quality_dir / "quality_summary.json", summary)
            write_quality_trace(
                quality_dir,
                run_dir=self.run_dir,
                texture_run_dir=self.config.texture_run_dir,
                metrics=metrics,
                evidence=evidence,
                findings=last_findings,
                repair_plan=None,
                provider=f"s2-reject-accept-agent+{self.config.evaluator_provider}",
            )
            write_agent_trace(self.run_dir, texture_dir=self.config.texture_run_dir)
            if self.config.force_report:
                refresh_s2_9999_report(
                    self.run_dir,
                    full=not self.config.fast_report_refresh,
                    trigger="downstream_report_refresh",
                )
            return str(self.run_dir / "report" / "index.html")
        except Exception as exc:  # noqa: BLE001
            self.trace.setdefault("report_errors", []).append(repr(exc))
            return None

    def run(self) -> S2AgentResult:
        report_path: Optional[str] = None
        accepted = False
        stop_condition = "max_iterations_exhausted"
        previous_score: Optional[float] = None
        no_improvement_count = 0
        sequential_stage_attempt_count = 0
        max_stage_attempts = max(1, int(self.config.max_iterations))
        sequential_stage_budget = bool(self.config.sequential_part_gate and self._is_assemble_geometry_scope())
        global_iteration_limit = int(self.config.max_iterations)
        if sequential_stage_budget:
            # In assemble_geometry sequential mode, max_iterations is the retry
            # budget for the current (part, stage), not a global loop budget.
            # Keep a conservative global safety bound so a full real run can
            # advance image -> mesh -> next part without stopping just because
            # an earlier stage accepted.
            global_iteration_limit = max_stage_attempts * max(1, len(self._sequential_part_ids) * 2 + 2)
            self.trace["policy"]["max_iterations_semantics"] = "per_part_stage_retry_budget"
            self.trace["policy"]["effective_global_iteration_safety_limit"] = global_iteration_limit
        else:
            self.trace["policy"]["max_iterations_semantics"] = "global_loop_budget"
        for iteration in range(global_iteration_limit + 1):
            metrics = collect_quality_metrics(
                self.run_dir,
                self.config.texture_run_dir,
                max_sample_vertices=self.config.max_sample_vertices,
                deep_mesh_components=False,
            )
            evaluator_findings, evaluator_evidence, evaluator_error, evaluator_interaction = self._run_visual_evaluator(iteration, metrics)
            report = validate_s2_run(
                self.run_dir,
                texture_run_dir=self.config.texture_run_dir,
                iteration=iteration,
                acceptance_score=self.config.acceptance_score,
                reject_severity=self.config.reject_severity,
                max_sample_vertices=self.config.max_sample_vertices,
                forbid_procedural_t2i=self.config.forbid_procedural_t2i,
                enable_split=self.config.enable_split,
                evaluator_findings=evaluator_findings,
                evaluator_error=evaluator_error,
            )
            validation_path = self._write_iteration_report(report)
            score_delta = None if previous_score is None else round(float(report.scene_score) - float(previous_score), 6)
            if previous_score is not None:
                if float(report.scene_score) < float(previous_score) + float(self.config.min_score_improvement):
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
            previous_score = float(report.scene_score)
            self.trace["score_trajectory"].append(
                {
                    "iteration": iteration,
                    "scene_score": report.scene_score,
                    "delta": score_delta,
                    "no_improvement_count": no_improvement_count,
                }
            )
            active_findings = self._findings_for_active_assemble_phase(report.findings)
            taxonomy: Dict[str, int] = {}
            for finding in report.findings:
                taxonomy[finding.failure_stage] = taxonomy.get(finding.failure_stage, 0) + 1
            active_taxonomy: Dict[str, int] = {}
            for finding in active_findings:
                active_taxonomy[finding.failure_stage] = active_taxonomy.get(finding.failure_stage, 0) + 1
            self.trace["failure_taxonomy"] = taxonomy
            self.trace["final_scene_score"] = report.scene_score
            self.trace["no_improvement_count"] = no_improvement_count
            iteration_record: Dict[str, Any] = {
                "iteration": iteration,
                "created_at": utc_now_iso(),
                "accepted": report.accepted,
                "scene_score": report.scene_score,
                "score_delta_from_previous": score_delta,
                "no_improvement_count": no_improvement_count,
                "current_stage_attempt_count": sequential_stage_attempt_count if sequential_stage_budget else None,
                "max_stage_attempts": max_stage_attempts if sequential_stage_budget else None,
                "failure_taxonomy": taxonomy,
                "num_findings": len(report.findings),
                "num_active_findings": len(active_findings),
                "max_severity": max((finding.severity for finding in report.findings), default=0),
                "active_max_severity": max((finding.severity for finding in active_findings), default=0),
                "validation_report_path": str(validation_path),
                "findings": [finding.to_dict() for finding in report.findings],
                "active_findings": [finding.to_dict() for finding in active_findings],
                "active_failure_taxonomy": active_taxonomy,
                "evaluator": {
                    "provider": self.config.evaluator_provider,
                    "enabled": self._evaluator_enabled(),
                    "model": self.config.evaluator_model,
                    "agent_scope": self._agent_scope(),
                    "agent_phase": self._focused_iteration_phase(iteration),
                    "status": "completed" if evaluator_findings else "error" if evaluator_error else "skipped",
                    "error": evaluator_error,
                    "issue_count": len(evaluator_findings.get("issues", []) or []) if evaluator_findings else 0,
                    "evidence_dir": evaluator_evidence.get("evidence_dir") if evaluator_evidence else None,
                    "interaction_dir": (evaluator_interaction or {}).get("interaction_dir"),
                    "question_prompt_path": (evaluator_interaction or {}).get("question_prompt_path"),
                    "answer_path": (evaluator_interaction or {}).get("answer_path"),
                    "focused_summary_path": (evaluator_interaction or {}).get("focused_summary_path"),
                    "current_part_id": (evaluator_interaction or {}).get("current_part_id"),
                    "current_part_stage": (evaluator_interaction or {}).get("current_part_stage"),
                    "sequential_part_index": self._sequential_part_index if self._sequential_gate_enabled() else None,
                    "mode": (evaluator_interaction or {}).get("mode") or self.config.evaluator_mode,
                    "input_images": ((evaluator_interaction or {}).get("request") or {}).get("input_images", []),
                    "critic_board_path": ((evaluator_interaction or {}).get("visual_critic") or {}).get("critic_board_path"),
                    "visual_critic_qa_path": ((evaluator_interaction or {}).get("visual_critic") or {}).get("qa_path"),
                    "visual_critic_repair_plan_path": ((evaluator_interaction or {}).get("visual_critic") or {}).get("repair_plan_path"),
                    "visual_critic_questions": ((evaluator_interaction or {}).get("request") or {}).get("questions", []),
                },
                "operations": [],
                "stage_records": [
                    self._stage_record(
                        stage="assembly_validation",
                        status="completed",
                        iteration=iteration,
                        failure_stage=None,
                        primary_artifact=validation_path,
                        inputs={
                            "agent_scope": self._agent_scope(),
                            "agent_phase": self._focused_iteration_phase(iteration),
                            "focused_scopes": self._focused_scopes_for_iteration(iteration),
                            "current_part_id": (evaluator_interaction or {}).get("current_part_id"),
                            "current_part_stage": (evaluator_interaction or {}).get("current_part_stage"),
                            "sequential_part_index": self._sequential_part_index if self._sequential_gate_enabled() else None,
                            "metrics_summary": "in-memory collect_quality_metrics",
                            "evaluator_provider": self.config.evaluator_provider,
                            "evidence_dir": evaluator_evidence.get("evidence_dir") if evaluator_evidence else None,
                            "vlm_question_prompt_path": (evaluator_interaction or {}).get("question_prompt_path"),
                            "vlm_input_image_count": len(((evaluator_interaction or {}).get("request") or {}).get("input_images", []) or []),
                        },
                        outputs={
                            "validation_report_path": str(validation_path),
                            "scene_score": report.scene_score,
                            "num_findings": len(report.findings),
                            "num_active_findings": len(active_findings),
                            "active_phase_gate": self._focused_iteration_phase(iteration),
                            "vlm_answer_path": (evaluator_interaction or {}).get("answer_path"),
                        },
                    )
                ],
            }
            if self._is_assemble_geometry_scope() and self._assemble_geometry_phase == "per_part_input_and_mesh_precheck" and not active_findings:
                if self.config.sequential_part_gate:
                    advanced = self._maybe_advance_sequential_part_gate(iteration=iteration, findings=report.findings)
                else:
                    advanced = self._maybe_advance_assemble_geometry_phase(iteration=iteration, findings=report.findings)
                iteration_record["workflow_transition"] = self.trace.get("workflow_transitions", [])[-1] if self.trace.get("workflow_transitions") else None
                self.trace["iterations"].append(iteration_record)
                sequential_stage_attempt_count = 0
                no_improvement_count = 0
                self.trace["no_improvement_count"] = no_improvement_count
                if iteration >= global_iteration_limit:
                    stop_condition = "global_iteration_safety_limit_exhausted"
                    break
                # Sequential mode intentionally spends one outer iteration per
                # blocking part-stage review.  If current image accepted, the
                # next iteration immediately reviews that same part's mesh; if
                # mesh accepted, it moves to the next part's image.  No repair
                # op is run for accepted stages.
                self._write_trace()
                report_path = self._refresh_downstream_reports()
                self._write_trace()
                continue

            if report.accepted and not (self._is_assemble_geometry_scope() and self._assemble_geometry_phase == "per_part_input_and_mesh_precheck"):
                accepted = True
                stop_condition = "accepted"
                self.trace["iterations"].append(iteration_record)
                break
            if iteration >= global_iteration_limit:
                stop_condition = "global_iteration_safety_limit_exhausted" if sequential_stage_budget else "max_iterations_exhausted"
                self.trace["iterations"].append(iteration_record)
                break
            if no_improvement_count >= int(self.config.max_no_improvement_iterations):
                stop_condition = "no_improvement"
                self.trace["iterations"].append(iteration_record)
                break

            if self._is_assemble_geometry_scope() and self._assemble_geometry_phase == "per_part_input_and_mesh_precheck":
                if self.config.sequential_part_gate:
                    self._maybe_advance_sequential_part_gate(iteration=iteration, findings=report.findings)
                else:
                    self._maybe_advance_assemble_geometry_phase(iteration=iteration, findings=report.findings)
                iteration_record["workflow_transition"] = self.trace.get("workflow_transitions", [])[-1] if self.trace.get("workflow_transitions") else None

            if (
                sequential_stage_budget
                and self._assemble_geometry_phase == "per_part_input_and_mesh_precheck"
                and active_findings
                and sequential_stage_attempt_count >= max_stage_attempts
            ):
                stop_condition = "current_part_stage_max_iterations_exhausted"
                iteration_record["current_stage_attempt_count"] = sequential_stage_attempt_count
                iteration_record["max_stage_attempts"] = max_stage_attempts
                self.trace["iterations"].append(iteration_record)
                break

            operations = repair_operations_from_findings(
                active_findings,
                enable_t2i=self.config.enable_t2i,
                enable_geometry=self.config.enable_geometry,
                enable_snap=self.config.enable_snap,
                enable_split=self.config.enable_split,
            )
            if not operations:
                stop_condition = "rejected_but_no_active_phase_operations"
                self.trace["iterations"].append(iteration_record)
                break
            executed: List[S2RepairOperation] = []
            operation_stage_records: List[Dict[str, Any]] = []
            for operation in operations:
                executed_operation, stage_records = self._execute_operation(operation, iteration)
                executed.append(executed_operation)
                operation_stage_records.extend(stage_records)
            iteration_record["operations"] = [operation.to_dict() for operation in executed]
            iteration_record["stage_records"].extend(operation_stage_records)
            if sequential_stage_budget and self._assemble_geometry_phase == "per_part_input_and_mesh_precheck" and active_findings:
                sequential_stage_attempt_count += 1
                iteration_record["current_stage_attempt_count_after_operations"] = sequential_stage_attempt_count
            self.trace["iterations"].append(iteration_record)
            self.trace["last_operation_statuses"] = [operation.status for operation in executed]
            if executed and not any(
                operation.status in {"succeeded", "dry_run", "partial_or_failed"}
                for operation in executed
            ):
                stop_condition = "all_operations_failed"
                self._write_trace()
                report_path = self._refresh_downstream_reports()
                self._write_trace()
                break
            self._write_trace()
            report_path = self._refresh_downstream_reports()
            self._write_trace()

        self.trace["status"] = "complete"
        self.trace["accepted"] = accepted
        self.trace["stop_condition"] = stop_condition
        self.trace["completed_at"] = utc_now_iso()
        report_path = self._refresh_downstream_reports() or report_path
        trace_path = self._write_trace()
        return S2AgentResult(
            run_dir=str(self.run_dir),
            accepted=accepted,
            stop_condition=stop_condition,
            iterations=list(self.trace.get("iterations", [])),
            trace_path=trace_path,
            report_path=report_path,
        )
