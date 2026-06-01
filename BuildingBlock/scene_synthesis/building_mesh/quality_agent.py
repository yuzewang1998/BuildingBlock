"""ArchStudio-S2 quality evaluation and repair-planning helpers.

This module is intentionally additive: it reads existing S2 run artifacts,
creates diagnostic metrics/evidence, validates evaluator findings, and plans
local repairs without invoking T2I, Hunyuan-Omni, or Hunyuan3D-Paint.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import base64
import mimetypes
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


QUALITY_SCHEMA_VERSION = "archstudio_s2_quality_findings.v1"
QUALITY_TRACE_VERSION = "archstudio_s2_quality_trace.v1"
REPAIR_PLAN_VERSION = "archstudio_s2_repair_plan.v1"

DEFAULT_QUALITY_SUBDIR = Path("quality") / "v9_agent"
DEFAULT_CONTACT_TOLERANCE = 0.035
DEFAULT_MIN_CONTACT_AREA_RATIO = 0.015
DEFAULT_MESH_GAP_TOLERANCE = 0.025
DEFAULT_MAX_SAMPLE_VERTICES = 50000
DEFAULT_EVIDENCE_MAX_FACES_PER_MESH = 8000
DEFAULT_AICODEMIRROR_BASE_URL = "https://api.aicodemirror.com/api/codex/backend-api/codex/v1"
DEFAULT_MULTIMODAL_MODEL = "gpt-5.5"
FOCUSED_VISUAL_CRITICS_SCHEMA_VERSION = "archstudio_s2.focused_visual_critics.v1"
FOCUSED_VISUAL_QUALITY_SCHEMA_VERSION = "archstudio_s2.focused_visual_quality_findings.v1"
S2_VISUAL_CRITIC_QUESTIONS = [
    "1. Do the numbered generated parts depict the requested architectural components, rather than a whole building or unrelated object?",
    "2. Do the part meshes fit their layout bounding boxes in scale, aspect ratio, orientation, and gravity direction?",
    "3. Are adjacent parts visually connected without obvious seams, floating, sinking, or unintended large overlaps?",
    "4. Is the assembled asset coherent as one architectural scene while preserving each part's text semantics?",
    "5. Which numbered parts should be accepted, moved/resized, regenerated from T2I, regenerated in 3D, split, merged, or deleted?",
]

ISSUE_TYPES = {
    "gap_or_seam",
    "scale_mismatch",
    "wrong_orientation",
    "wrong_semantics",
    "style_inconsistent",
    "texture_bad",
    "texture_direction_bad",
    "artifact_base_or_panel",
    "missing_part",
    "overlap",
    "low_geometry_quality",
    "low_texture_quality",
    "global_style_issue",
    "floating",
    "penetration",
}

RECOMMENDED_ACTIONS = {
    "accept",
    "deterministic_resize_snap",
    "rerun_t2i",
    "rerun_geometry",
    "rerun_texture",
    "rerun_t2i_geometry_texture",
    "manual_review",
}


Vector3 = Tuple[float, float, float]
Bounds = Tuple[Vector3, Vector3]


@dataclass(frozen=True)
class ObjMesh:
    vertices: List[Vector3]
    faces: List[Tuple[int, ...]]


@dataclass(frozen=True)
class ObjMeshSummary:
    vertex_count: int
    face_count: int
    bounds: Optional[Bounds]
    sample_vertices: List[Vector3]
    sample_stride: int
    sample_limit: int
    file_size_bytes: int


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output_path)


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from strict, fenced, or prefaced model output."""
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("expected evaluator response to be a JSON object")
    return payload


def load_openai_api_key(auth_path: Optional[str | Path] = None) -> str:
    """Load an OpenAI-compatible API key without logging or persisting it."""
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    candidates: List[Path] = []
    if auth_path:
        candidates.append(Path(auth_path))
    else:
        candidates.extend([
            Path.home() / ".codex" / "provider_snapshots" / "auth_aicodemirror.json",
            Path.home() / ".codex-live" / "provider_snapshots" / "auth_aicodemirror.json",
            Path.home() / ".codex" / "auth.json",
        ])

    for path in candidates:
        if not path.exists():
            continue
        payload = load_json(path)
        api_key = payload.get("OPENAI_API_KEY") if isinstance(payload, Mapping) else None
        if isinstance(api_key, str) and api_key.strip() and api_key.strip().lower() != "none":
            return api_key.strip()
    raise RuntimeError(
        "OpenAI-compatible credentials not found. Set OPENAI_API_KEY or keep a valid key in ~/.codex/provider_snapshots/auth_aicodemirror.json."
    )


def _as_float3(values: Sequence[Any]) -> Vector3:
    if len(values) != 3:
        raise ValueError("expected 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


def bbox_bounds(part: Mapping[str, Any]) -> Bounds:
    bbox = part.get("bbox") or {}
    center = _as_float3(bbox.get("center") or part.get("actor_location") or [])
    size = _as_float3(bbox.get("size") or part.get("actor_size") or [])
    half = tuple(value / 2.0 for value in size)
    return (
        (center[0] - half[0], center[1] - half[1], center[2] - half[2]),
        (center[0] + half[0], center[1] + half[1], center[2] + half[2]),
    )


def bounds_extent(bounds: Bounds) -> Vector3:
    mn, mx = bounds
    return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


def bounds_center(bounds: Bounds) -> Vector3:
    mn, mx = bounds
    return ((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0)


def _volume(extent: Sequence[float]) -> float:
    value = 1.0
    for item in extent:
        value *= max(float(item), 0.0)
    return value


def read_obj_mesh(path: str | Path) -> ObjMesh:
    vertices: List[Vector3] = []
    faces: List[Tuple[int, ...]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith("v "):
                parts = raw.strip().split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif raw.startswith("f "):
                face: List[int] = []
                for token in raw.strip().split()[1:]:
                    index_text = token.split("/")[0]
                    if not index_text:
                        continue
                    index = int(index_text)
                    if index < 0:
                        index = len(vertices) + index + 1
                    face.append(index - 1)
                if len(face) >= 3:
                    faces.append(tuple(face))
    return ObjMesh(vertices=vertices, faces=faces)


def read_obj_mesh_summary(
    path: str | Path,
    *,
    max_sample_vertices: int = DEFAULT_MAX_SAMPLE_VERTICES,
) -> ObjMeshSummary:
    """Stream OBJ bounds/count metrics without materializing dense faces.

    Generated Hunyuan meshes can be tens or hundreds of MB per part.  The V9 QA
    loop only needs first-pass bounds, fill ratios, and face counts; loading all
    faces to build adjacency makes this diagnostic path CPU/memory-bound.  This
    parser keeps exact min/max bounds and a deterministic evenly-spaced vertex
    sample for approximate percentile bounds.
    """
    path = Path(path)
    sample_limit = max(0, int(max_sample_vertices))
    sample_stride = 1
    samples: List[Vector3] = []
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    vertex_count = 0
    face_count = 0
    file_size = path.stat().st_size if path.exists() else 0

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith("v "):
                parts = raw.split(maxsplit=4)
                if len(parts) < 4:
                    continue
                try:
                    vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError:
                    continue
                vertex_count += 1
                for axis, value in enumerate(vertex):
                    if value < mins[axis]:
                        mins[axis] = value
                    if value > maxs[axis]:
                        maxs[axis] = value
                if sample_limit and vertex_count % sample_stride == 0:
                    samples.append(vertex)
                    if len(samples) > sample_limit * 2:
                        samples[:] = samples[::2]
                        sample_stride *= 2
            elif raw.startswith("f "):
                face_count += 1

    if sample_limit and len(samples) > sample_limit:
        keep_every = int(math.ceil(len(samples) / float(sample_limit)))
        samples = samples[::keep_every][:sample_limit]

    bounds = None
    if vertex_count:
        bounds = ((mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2]))
    return ObjMeshSummary(
        vertex_count=vertex_count,
        face_count=face_count,
        bounds=bounds,
        sample_vertices=samples,
        sample_stride=sample_stride,
        sample_limit=sample_limit,
        file_size_bytes=file_size,
    )


def point_bounds(points: Sequence[Sequence[float]]) -> Bounds:
    if not points:
        raise ValueError("cannot compute bounds for empty point list")
    return (
        tuple(min(float(point[axis]) for point in points) for axis in range(3)),  # type: ignore[return-value]
        tuple(max(float(point[axis]) for point in points) for axis in range(3)),  # type: ignore[return-value]
    )


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("empty percentile values")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def percentile_point_bounds(points: Sequence[Sequence[float]], low: float = 0.01, high: float = 0.99) -> Bounds:
    if not points:
        raise ValueError("cannot compute percentile bounds for empty point list")
    mins = []
    maxs = []
    for axis in range(3):
        values = [float(point[axis]) for point in points]
        mins.append(percentile(values, low))
        maxs.append(percentile(values, high))
    return (tuple(mins), tuple(maxs))  # type: ignore[return-value]


def face_component_stats(mesh: ObjMesh) -> Dict[str, Any]:
    if not mesh.faces:
        return {
            "component_count": 0,
            "largest_component_face_ratio": 0.0,
            "largest_component_faces": 0,
        }
    vertex_to_faces: Dict[int, List[int]] = {}
    for face_index, face in enumerate(mesh.faces):
        for vertex_index in face:
            vertex_to_faces.setdefault(vertex_index, []).append(face_index)
    neighbors: List[set[int]] = [set() for _ in mesh.faces]
    for face_indices in vertex_to_faces.values():
        for face_index in face_indices:
            neighbors[face_index].update(face_indices)

    seen: set[int] = set()
    component_sizes: List[int] = []
    for start in range(len(mesh.faces)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nxt in neighbors[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        component_sizes.append(size)
    largest = max(component_sizes) if component_sizes else 0
    return {
        "component_count": len(component_sizes),
        "largest_component_face_ratio": round(largest / max(len(mesh.faces), 1), 6),
        "largest_component_faces": largest,
        "component_face_counts": sorted(component_sizes, reverse=True),
    }


def _round3(values: Sequence[float]) -> List[float]:
    return [round(float(value), 6) for value in values]


def mesh_metrics_for_part(
    part: Mapping[str, Any],
    mesh_path: str | Path | None,
    *,
    max_sample_vertices: int = DEFAULT_MAX_SAMPLE_VERTICES,
    deep_mesh_components: bool = False,
) -> Dict[str, Any]:
    part_id = str(part.get("part_id", ""))
    bbox = bbox_bounds(part)
    bbox_extent = bounds_extent(bbox)
    metrics: Dict[str, Any] = {
        "part_id": part_id,
        "bbox": {
            "min": _round3(bbox[0]),
            "max": _round3(bbox[1]),
            "extent": _round3(bbox_extent),
            "center": _round3(bounds_center(bbox)),
        },
        "mesh_path": str(mesh_path) if mesh_path else None,
        "mesh_exists": bool(mesh_path and Path(mesh_path).exists()),
    }
    if not metrics["mesh_exists"]:
        metrics.update({
            "vertex_count": 0,
            "face_count": 0,
            "mesh_parser_mode": "missing",
            "axis_fill_ratio": [0.0, 0.0, 0.0],
            "percentile_axis_fill_ratio": [0.0, 0.0, 0.0],
            "issue_flags": ["missing_mesh"],
        })
        return metrics

    summary = read_obj_mesh_summary(
        Path(str(mesh_path)),
        max_sample_vertices=max_sample_vertices,
    )
    metrics["vertex_count"] = summary.vertex_count
    metrics["face_count"] = summary.face_count
    metrics["mesh_file_size_bytes"] = summary.file_size_bytes
    metrics["mesh_parser_mode"] = "streaming_vertex_bounds"
    metrics["percentile_sample_count"] = len(summary.sample_vertices)
    metrics["percentile_sample_limit"] = summary.sample_limit
    metrics["percentile_sample_stride"] = summary.sample_stride
    if summary.bounds is None:
        metrics.update({
            "axis_fill_ratio": [0.0, 0.0, 0.0],
            "percentile_axis_fill_ratio": [0.0, 0.0, 0.0],
            "issue_flags": ["empty_mesh"],
        })
        return metrics

    mesh_bounds = summary.bounds
    mesh_extent = bounds_extent(mesh_bounds)
    p_bounds = percentile_point_bounds(summary.sample_vertices) if summary.sample_vertices else mesh_bounds
    p_extent = bounds_extent(p_bounds)
    axis_fill = [mesh_extent[i] / max(bbox_extent[i], 1e-9) for i in range(3)]
    percentile_fill = [p_extent[i] / max(bbox_extent[i], 1e-9) for i in range(3)]
    center_delta = [bounds_center(mesh_bounds)[i] - bounds_center(bbox)[i] for i in range(3)]
    flags: List[str] = []
    if any(value < 0.70 for value in percentile_fill):
        flags.append("low_percentile_bbox_fill")
    if any(value > 1.18 for value in axis_fill):
        flags.append("mesh_exceeds_bbox")
    if any(abs(center_delta[i]) > max(bbox_extent[i] * 0.12, 1e-4) for i in range(3)):
        flags.append("mesh_center_shift")

    if deep_mesh_components:
        mesh = read_obj_mesh(Path(str(mesh_path)))
        component_stats = face_component_stats(mesh)
        component_stats["component_analysis"] = "full_face_adjacency"
        if component_stats["component_count"] > 1 and component_stats["largest_component_face_ratio"] < 0.92:
            flags.append("fragmented_mesh")
    else:
        component_stats = {
            "component_analysis": "skipped_fast_metrics",
            "component_count": None,
            "largest_component_face_ratio": None,
            "largest_component_faces": None,
        }

    metrics.update({
        "mesh": {
            "min": _round3(mesh_bounds[0]),
            "max": _round3(mesh_bounds[1]),
            "extent": _round3(mesh_extent),
            "center": _round3(bounds_center(mesh_bounds)),
        },
        "percentile_mesh": {
            "min": _round3(p_bounds[0]),
            "max": _round3(p_bounds[1]),
            "extent": _round3(p_extent),
        },
        "axis_fill_ratio": _round3(axis_fill),
        "percentile_axis_fill_ratio": _round3(percentile_fill),
        "center_delta": _round3(center_delta),
        **component_stats,
        "issue_flags": flags,
    })
    return metrics


def bbox_contact_decision(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    contact_tolerance: float = DEFAULT_CONTACT_TOLERANCE,
    min_contact_area_ratio: float = DEFAULT_MIN_CONTACT_AREA_RATIO,
) -> Dict[str, Any]:
    a_min, a_max = bbox_bounds(a)
    b_min, b_max = bbox_bounds(b)
    overlaps = [min(a_max[i], b_max[i]) - max(a_min[i], b_min[i]) for i in range(3)]
    gaps = []
    directions = []
    for axis in range(3):
        if a_max[axis] < b_min[axis]:
            gaps.append(b_min[axis] - a_max[axis])
            directions.append(1)
        elif b_max[axis] < a_min[axis]:
            gaps.append(a_min[axis] - b_max[axis])
            directions.append(-1)
        else:
            gaps.append(0.0)
            directions.append(0)

    separating_axes = [axis for axis, gap in enumerate(gaps) if gap > 1e-9]
    positive_overlap_axes = [axis for axis, overlap in enumerate(overlaps) if overlap > 1e-9]
    axis_names = ("x", "y", "z")
    if not separating_axes and len(positive_overlap_axes) == 3:
        overlap_volume = _volume(overlaps)
        ratio = overlap_volume / max(min(_volume(bounds_extent((a_min, a_max))), _volume(bounds_extent((b_min, b_max)))), 1e-9)
        return {
            "connected": True,
            "contact_type": "overlap",
            "axis": None,
            "gap": 0.0,
            "gap_vector": [0.0, 0.0, 0.0],
            "direction_from_a_to_b": [0, 0, 0],
            "contact_area_ratio": round(min(1.0, ratio), 6),
            "overlap_volume": round(overlap_volume, 6),
        }

    if len(separating_axes) <= 1:
        axis = separating_axes[0] if separating_axes else next((i for i, overlap in enumerate(overlaps) if abs(overlap) <= 1e-9), None)
        if axis is not None:
            other_axes = [i for i in range(3) if i != axis]
            if all(overlaps[i] > 1e-9 for i in other_axes):
                contact_area = overlaps[other_axes[0]] * overlaps[other_axes[1]]
                a_face = (a_max[other_axes[0]] - a_min[other_axes[0]]) * (a_max[other_axes[1]] - a_min[other_axes[1]])
                b_face = (b_max[other_axes[0]] - b_min[other_axes[0]]) * (b_max[other_axes[1]] - b_min[other_axes[1]])
                area_ratio = contact_area / max(min(a_face, b_face), 1e-9)
                gap = gaps[axis]
                if gap <= contact_tolerance and area_ratio >= min_contact_area_ratio:
                    return {
                        "connected": True,
                        "contact_type": "near_face" if gap > 1e-9 else "face_touch",
                        "axis": axis_names[axis],
                        "gap": round(gap, 6),
                        "gap_vector": _round3(gaps),
                        "direction_from_a_to_b": list(directions),
                        "contact_area_ratio": round(min(1.0, area_ratio), 6),
                        "overlap_volume": 0.0,
                    }

    gap_distance = math.sqrt(sum(value * value for value in gaps))
    return {
        "connected": False,
        "contact_type": "gap",
        "axis": axis_names[separating_axes[0]] if len(separating_axes) == 1 else None,
        "gap": round(gap_distance, 6),
        "gap_vector": _round3(gaps),
        "direction_from_a_to_b": list(directions),
        "contact_area_ratio": 0.0,
        "overlap_volume": 0.0,
    }


def expected_bbox_contacts(
    contract: Mapping[str, Any],
    *,
    contact_tolerance: float = DEFAULT_CONTACT_TOLERANCE,
    min_contact_area_ratio: float = DEFAULT_MIN_CONTACT_AREA_RATIO,
) -> List[Dict[str, Any]]:
    parts = list(contract.get("parts", []) or [])
    edges: List[Dict[str, Any]] = []
    for i, part_a in enumerate(parts):
        for j in range(i + 1, len(parts)):
            part_b = parts[j]
            decision = bbox_contact_decision(
                part_a,
                part_b,
                contact_tolerance=contact_tolerance,
                min_contact_area_ratio=min_contact_area_ratio,
            )
            if decision["connected"]:
                edges.append({
                    "part_a": str(part_a.get("part_id")),
                    "part_b": str(part_b.get("part_id")),
                    "index_a": i,
                    "index_b": j,
                    **decision,
                })
    return edges


def _bounds_from_metric(metric: Mapping[str, Any], key: str) -> Optional[Bounds]:
    payload = metric.get(key)
    if not isinstance(payload, Mapping):
        return None
    mn = payload.get("min")
    mx = payload.get("max")
    if not isinstance(mn, Sequence) or not isinstance(mx, Sequence) or len(mn) != 3 or len(mx) != 3:
        return None
    return (_as_float3(mn), _as_float3(mx))


def bounds_gap(a: Bounds, b: Bounds) -> Tuple[float, List[float]]:
    a_min, a_max = a
    b_min, b_max = b
    gaps = []
    for axis in range(3):
        if a_max[axis] < b_min[axis]:
            gaps.append(b_min[axis] - a_max[axis])
        elif b_max[axis] < a_min[axis]:
            gaps.append(a_min[axis] - b_max[axis])
        else:
            gaps.append(0.0)
    return math.sqrt(sum(value * value for value in gaps)), gaps


def mesh_contact_gaps(
    part_metrics: Mapping[str, Mapping[str, Any]],
    expected_contacts: Sequence[Mapping[str, Any]],
    *,
    mesh_gap_tolerance: float = DEFAULT_MESH_GAP_TOLERANCE,
) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = []
    for edge in expected_contacts:
        part_a = str(edge["part_a"])
        part_b = str(edge["part_b"])
        metric_a = part_metrics.get(part_a, {})
        metric_b = part_metrics.get(part_b, {})
        bounds_a = _bounds_from_metric(metric_a, "percentile_mesh") or _bounds_from_metric(metric_a, "mesh")
        bounds_b = _bounds_from_metric(metric_b, "percentile_mesh") or _bounds_from_metric(metric_b, "mesh")
        if bounds_a is None or bounds_b is None:
            gaps.append({
                "part_a": part_a,
                "part_b": part_b,
                "status": "missing_mesh_bounds",
                "gap": None,
                "above_tolerance": True,
            })
            continue
        distance, gap_vector = bounds_gap(bounds_a, bounds_b)
        gaps.append({
            "part_a": part_a,
            "part_b": part_b,
            "status": "measured",
            "bbox_contact_type": edge.get("contact_type"),
            "bbox_contact_axis": edge.get("axis"),
            "gap": round(distance, 6),
            "gap_vector": _round3(gap_vector),
            "above_tolerance": distance > mesh_gap_tolerance,
            "mesh_gap_tolerance": mesh_gap_tolerance,
        })
    return gaps


def resolve_contract_from_run(run_dir: str | Path, manifest: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "contract" / "layout_mesh_contract.json",
        run_dir / "layout_mesh_contract.json",
        run_dir / "input" / "layout_mesh_contract.original.json",
    ]
    if manifest:
        for part in manifest.get("parts", []) or []:
            path = part.get("contract_path")
            if path:
                candidates.append(Path(path))
    for candidate in candidates:
        if candidate.exists():
            payload = load_json(candidate)
            if isinstance(payload, dict) and payload.get("parts"):
                return payload
    raise FileNotFoundError(f"unable to locate layout_mesh_contract.json for {run_dir}")


def resolve_part_mesh_path(run_dir: str | Path, part_id: str, manifest_part: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    run_dir = Path(run_dir)
    manifest_part = dict(manifest_part or {})
    candidates = [
        manifest_part.get("normalized_output_path"),
        run_dir / "assemblies" / "parts" / f"{part_id}.obj",
        manifest_part.get("placeholder_output_path"),
        run_dir / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj",
        manifest_part.get("raw_output_path"),
        run_dir / "parts" / part_id / f"{part_id}.obj",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def load_texture_manifest(texture_run_dir: str | Path | None) -> Dict[str, Any]:
    if not texture_run_dir:
        return {}
    path = Path(texture_run_dir) / "texture_manifest.json"
    if path.exists():
        payload = load_json(path)
        return payload if isinstance(payload, dict) else {}
    return {}


def texture_status_by_part(texture_manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(part.get("part_id")): dict(part)
        for part in texture_manifest.get("parts", []) or []
        if part.get("part_id")
    }


def collect_quality_metrics(
    run_dir: str | Path,
    texture_run_dir: str | Path | None = None,
    *,
    contact_tolerance: float = DEFAULT_CONTACT_TOLERANCE,
    min_contact_area_ratio: float = DEFAULT_MIN_CONTACT_AREA_RATIO,
    mesh_gap_tolerance: float = DEFAULT_MESH_GAP_TOLERANCE,
    max_sample_vertices: int = DEFAULT_MAX_SAMPLE_VERTICES,
    deep_mesh_components: bool = False,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    contract = resolve_contract_from_run(run_dir, manifest)
    manifest_by_part = {
        str(part.get("part_id")): dict(part)
        for part in manifest.get("parts", []) or []
        if part.get("part_id")
    }
    texture_manifest = load_texture_manifest(texture_run_dir)
    texture_by_part = texture_status_by_part(texture_manifest)
    part_metrics: Dict[str, Dict[str, Any]] = {}
    for index, part in enumerate(contract.get("parts", []) or []):
        part_id = str(part.get("part_id"))
        mesh_path = resolve_part_mesh_path(run_dir, part_id, manifest_by_part.get(part_id))
        metric = mesh_metrics_for_part(
            part,
            mesh_path,
            max_sample_vertices=max_sample_vertices,
            deep_mesh_components=deep_mesh_components,
        )
        metric.update({
            "index": index,
            "target_class": part.get("target_class", ""),
            "display_label": (
                part.get("part_description")
                or part.get("open_vocab_label")
                or part.get("source_part_name")
                or part.get("source_actor_label")
                or part.get("target_class", "")
            ),
            "semantic_role": part.get("semantic_role", ""),
            "detail_level": part.get("detail_level", ""),
            "spatial_relations": list(part.get("spatial_relations", []) or []),
            "prompt": part.get("generation_prompt") or part.get("part_prompt") or part.get("target_prompt") or "",
            "texture_status": texture_by_part.get(part_id, {}).get("status"),
            "texture_record": texture_by_part.get(part_id, {}),
        })
        part_metrics[part_id] = metric

    contacts = expected_bbox_contacts(
        contract,
        contact_tolerance=contact_tolerance,
        min_contact_area_ratio=min_contact_area_ratio,
    )
    contact_gaps = mesh_contact_gaps(part_metrics, contacts, mesh_gap_tolerance=mesh_gap_tolerance)
    part_issue_flags = {
        part_id: metric.get("issue_flags", [])
        for part_id, metric in part_metrics.items()
        if metric.get("issue_flags")
    }
    return {
        "schema_version": "archstudio_s2_quality_metrics.v1",
        "run_dir": str(run_dir),
        "texture_run_dir": str(Path(texture_run_dir).resolve()) if texture_run_dir else None,
        "layout_id": contract.get("layout_id"),
        "num_parts": len(contract.get("parts", []) or []),
        "num_meshes_present": sum(1 for metric in part_metrics.values() if metric.get("mesh_exists")),
        "num_texture_records": len(texture_by_part),
        "scene_bounds": contract.get("scene_bounds"),
        "coordinate_frame": contract.get("coordinate_frame"),
        "geometry_continuity_contract": contract.get("geometry_continuity_contract"),
        "contact_policy": {
            "contact_tolerance": contact_tolerance,
            "min_contact_area_ratio": min_contact_area_ratio,
            "mesh_gap_tolerance": mesh_gap_tolerance,
            "source": "s2_recomputed_bbox_contact_graph_with_s1_contract_when_present",
        },
        "mesh_metric_policy": {
            "parser_mode": "streaming_vertex_bounds",
            "max_sample_vertices": max_sample_vertices,
            "deep_mesh_components": deep_mesh_components,
            "component_analysis": "full_face_adjacency" if deep_mesh_components else "skipped_fast_metrics",
        },
        "expected_bbox_contacts": contacts,
        "mesh_contact_gaps": contact_gaps,
        "parts": list(part_metrics.values()),
        "part_issue_flags": part_issue_flags,
        "summary": {
            "expected_contact_count": len(contacts),
            "mesh_gap_count": sum(1 for item in contact_gaps if item.get("above_tolerance")),
            "missing_mesh_count": sum(1 for metric in part_metrics.values() if not metric.get("mesh_exists")),
            "flagged_part_count": len(part_issue_flags),
            "texture_failure_count": sum(1 for item in texture_by_part.values() if item.get("status") != "succeeded"),
        },
    }


def deterministic_findings_from_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    issue_index = 1
    for part in metrics.get("parts", []) or []:
        part_id = str(part.get("part_id"))
        label = int(part.get("index", 0)) + 1
        flags = set(part.get("issue_flags", []) or [])
        if "missing_mesh" in flags or "empty_mesh" in flags:
            issues.append({
                "issue_id": f"d{issue_index:03d}",
                "part_ids": [part_id],
                "labels": [label],
                "issue_type": "missing_part",
                "severity": 5,
                "confidence": 1.0,
                "evidence": ["metrics_summary"],
                "diagnosis": "No usable mesh was found for this part.",
                "recommended_action": "rerun_geometry",
                "repair_instruction": "Regenerate geometry for this exact part using the existing reference image and bbox.",
            })
            issue_index += 1
        if flags.intersection({"low_percentile_bbox_fill", "mesh_exceeds_bbox", "mesh_center_shift", "fragmented_mesh"}):
            issues.append({
                "issue_id": f"d{issue_index:03d}",
                "part_ids": [part_id],
                "labels": [label],
                "issue_type": "scale_mismatch" if "mesh_exceeds_bbox" in flags or "mesh_center_shift" in flags else "low_geometry_quality",
                "severity": 3,
                "confidence": 0.78,
                "evidence": ["metrics_summary", "partboard"],
                "diagnosis": "Mesh statistics suggest poor bbox fill, center shift, or fragmented geometry.",
                "recommended_action": "manual_review",
                "repair_instruction": "Inspect the partboard; choose resize/snap if semantics are acceptable, otherwise rerun T2I+geometry.",
            })
            issue_index += 1

    for gap in metrics.get("mesh_contact_gaps", []) or []:
        if not gap.get("above_tolerance"):
            continue
        part_ids = [str(gap.get("part_a")), str(gap.get("part_b"))]
        labels = []
        index_by_id = {str(part.get("part_id")): int(part.get("index", 0)) + 1 for part in metrics.get("parts", []) or []}
        for part_id in part_ids:
            if part_id in index_by_id:
                labels.append(index_by_id[part_id])
        issues.append({
            "issue_id": f"d{issue_index:03d}",
            "part_ids": part_ids,
            "labels": labels,
            "issue_type": "gap_or_seam",
            "severity": 4,
            "confidence": 0.82,
            "evidence": ["assembly_geometry_views", "metrics_summary"],
            "diagnosis": "Layout bboxes are expected to touch or overlap, but generated mesh percentile bounds leave a measurable gap.",
            "recommended_action": "deterministic_resize_snap",
            "repair_instruction": "Resize or snap the affected mesh faces toward the expected bbox contact axis before considering a full rerun.",
        })
        issue_index += 1

    scene_score = max(0.0, 1.0 - min(1.0, len(issues) / max(float(metrics.get("num_parts", 1)), 1.0)))
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "scene_score": round(scene_score, 3),
        "global_summary": "Deterministic mock evaluator generated findings from mesh metrics.",
        "global_style_diagnosis": "Style coherence is not judged in mock mode.",
        "issues": issues,
    }


def validate_quality_findings(payload: Mapping[str, Any], valid_part_ids: Iterable[str]) -> Dict[str, Any]:
    valid_ids = set(str(item) for item in valid_part_ids)
    result = dict(payload)
    if result.get("schema_version") != QUALITY_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {QUALITY_SCHEMA_VERSION}")
    issues = result.get("issues")
    if not isinstance(issues, list):
        raise ValueError("quality findings must contain an issues list")
    try:
        result["scene_score"] = max(0.0, min(1.0, float(result.get("scene_score", 0.0))))
    except (TypeError, ValueError) as exc:
        raise ValueError("scene_score must be numeric") from exc
    for index, issue in enumerate(issues):
        if not isinstance(issue, Mapping):
            raise ValueError(f"issue {index} must be an object")
        part_ids = issue.get("part_ids")
        if not isinstance(part_ids, list):
            raise ValueError(f"issue {index} part_ids must be a list")
        unknown = [str(part_id) for part_id in part_ids if str(part_id) not in valid_ids]
        if unknown:
            raise ValueError(f"issue {index} references unknown part_ids: {unknown}")
        issue_type = issue.get("issue_type")
        if issue_type not in ISSUE_TYPES:
            raise ValueError(f"issue {index} has unknown issue_type: {issue_type}")
        action = issue.get("recommended_action")
        if action not in RECOMMENDED_ACTIONS:
            raise ValueError(f"issue {index} has unknown recommended_action: {action}")
        severity = issue.get("severity")
        try:
            numeric_severity = int(severity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"issue {index} severity must be an integer") from exc
        if numeric_severity < 1 or numeric_severity > 5:
            raise ValueError(f"issue {index} severity must be in [1, 5]")
        try:
            confidence = float(issue.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"issue {index} confidence must be numeric") from exc
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"issue {index} confidence must be in [0, 1]")
    return result


def _copy_if_exists(source: str | Path | None, destination: Path) -> Optional[str]:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination)


def build_part_index_map(contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for index, part in enumerate(contract.get("parts", []) or []):
        rows.append({
            "label": index + 1,
            "index": index,
            "part_id": part.get("part_id"),
            "display_label": (
                part.get("part_description")
                or part.get("open_vocab_label")
                or part.get("source_part_name")
                or part.get("source_actor_label")
                or part.get("target_class", "")
            ),
            "target_class": part.get("target_class", ""),
            "semantic_role": part.get("semantic_role", ""),
            "detail_level": part.get("detail_level", ""),
            "spatial_relations": list(part.get("spatial_relations", []) or []),
            "prompt": part.get("generation_prompt") or part.get("part_prompt") or part.get("target_prompt") or "",
            "bbox": part.get("bbox", {}),
            "reference_image": part.get("reference_image_path") or part.get("reference_image"),
        })
    return rows


def _manifest_parts_by_id(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(part.get("part_id")): part
        for part in manifest.get("parts", []) or []
        if part.get("part_id")
    }


def _assembly_part_mesh_path(run_dir: Path, part_id: str, manifest_part: Mapping[str, Any] | None = None) -> Optional[Path]:
    candidates: List[Path] = []
    if manifest_part:
        for key in ("normalized_output_path", "raw_output_path", "placeholder_output_path"):
            value = manifest_part.get(key)
            if value:
                candidates.append(Path(str(value)))
    candidates.extend([
        run_dir / "assemblies" / "parts" / f"{part_id}.obj",
        run_dir / "parts" / part_id / f"{part_id}.obj",
        run_dir / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_evidence_render_notes(
    run_dir: Path,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    max_faces_per_mesh: int,
) -> Dict[str, Any]:
    """Describe evidence-render sampling so the visual judge does not over-trust raster artifacts.

    Matplotlib is used for offline evidence rendering because this repository
    should stay dependency-light.  Hunyuan outputs often contain hundreds of
    thousands of tiny triangles per part, so PNG evidence is necessarily a
    deterministic sampled surface view.  The interactive GLB and deterministic
    mesh metrics remain the geometry ground truth.
    """
    manifest_by_id = _manifest_parts_by_id(manifest)
    part_notes: List[Dict[str, Any]] = []
    for index, part in enumerate(contract.get("parts", []) or []):
        part_id = str(part.get("part_id"))
        mesh_path = _assembly_part_mesh_path(run_dir, part_id, manifest_by_id.get(part_id))
        summary = None
        if mesh_path and mesh_path.exists():
            try:
                mesh_summary = read_obj_mesh_summary(mesh_path, max_sample_vertices=0)
                summary = {
                    "mesh_path": str(mesh_path),
                    "vertex_count": mesh_summary.vertex_count,
                    "face_count": mesh_summary.face_count,
                    "rendered_face_cap": int(max_faces_per_mesh),
                    "render_face_fraction": round(
                        min(1.0, float(max_faces_per_mesh) / float(max(mesh_summary.face_count, 1))),
                        6,
                    ),
                }
            except Exception as exc:
                summary = {"mesh_path": str(mesh_path), "error": repr(exc)}
        part_notes.append({
            "label": index + 1,
            "part_id": part_id,
            "display_label": (
                part.get("part_description")
                or part.get("open_vocab_label")
                or part.get("source_part_name")
                or part.get("target_class", "")
            ),
            "mesh": summary,
        })
    return {
        "schema_version": "archstudio_s2_quality_evidence_render_notes.v1",
        "raster_renderer": "filled projected polygon surface preview (not point-cloud scatter)",
        "rendered_face_cap_per_mesh": int(max_faces_per_mesh),
        "interpretation_warning": (
            "Assembly PNGs are rendered as filled mesh surfaces with a higher face cap for VLM readability. "
            "Do not interpret dark renderer backgrounds as point clouds. Cross-check part-level mesh renders, "
            "interactive GLB assets, vertex/face counts, bbox fill metrics, and contact-gap metrics."
        ),
        "ground_truth_geometry_assets": [
            "report/assets/assembly_complete_per_part_color_yup.glb",
            "report/assets/parts/<part_id>.glb",
            "assemblies/parts/<part_id>.obj",
        ],
        "parts": part_notes,
    }


def write_partboard_images(part_rows: Sequence[Mapping[str, Any]], output_dir: Path, *, rows_per_page: int = 8) -> List[str]:
    """Write compact PNG part boards when Pillow is available."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    pages: List[str] = []
    width = 1600
    row_h = 170
    for page_index in range(0, len(part_rows), rows_per_page):
        rows = part_rows[page_index : page_index + rows_per_page]
        image = Image.new("RGB", (width, max(1, len(rows)) * row_h + 40), "#f7f7f5")
        draw = ImageDraw.Draw(image)
        draw.text((18, 12), "ArchStudio-S2 V9 partboard evidence", fill="#222", font=font)
        for local_index, row in enumerate(rows):
            y = 38 + local_index * row_h
            draw.rounded_rectangle((12, y, width - 12, y + row_h - 10), radius=10, fill="#ffffff", outline="#d8d8d8")
            label = row.get("label")
            part_id = str(row.get("part_id", ""))
            display = str(row.get("display_label") or row.get("target_class") or "")
            role = str(row.get("semantic_role") or "")
            prompt = str(row.get("prompt") or "")
            bbox = row.get("bbox") or {}
            draw.text((28, y + 12), f"#{label} {display}", fill="#111", font=font)
            draw.text((28, y + 32), part_id[:120], fill="#555", font=font)
            draw.text((28, y + 52), f"role={role or 'n/a'} bbox={bbox}", fill="#555", font=font)
            wrapped = []
            line = ""
            for word in prompt.split():
                if len(line) + len(word) > 150:
                    wrapped.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                wrapped.append(line)
            for offset, text in enumerate(wrapped[:4]):
                draw.text((28, y + 74 + offset * 16), text, fill="#333", font=font)

            ref = row.get("reference_image")
            if ref and Path(str(ref)).exists():
                try:
                    ref_img = Image.open(str(ref)).convert("RGB")
                    ref_img.thumbnail((145, 145))
                    image.paste(ref_img, (width - 175, y + 10))
                except Exception:
                    pass
        page_path = output_dir / f"partboard_{len(pages):03d}.png"
        image.save(page_path)
        pages.append(str(page_path))
    return pages


def _paste_thumbnail(
    canvas: Any,
    draw: Any,
    image_path: str | Path | None,
    box: Tuple[int, int, int, int],
    *,
    label: str,
    font: Any,
) -> None:
    """Paste an image thumbnail into a labeled box; tolerate missing assets."""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill="#111318", outline="#2d3440")
    draw.text((x0 + 10, y0 + 8), label, fill="#f4efe5", font=font)
    inner = (x0 + 10, y0 + 28, x1 - 10, y1 - 10)
    if not image_path or not Path(str(image_path)).exists():
        draw.text((inner[0], inner[1] + 20), "missing", fill="#d7b56d", font=font)
        return
    try:
        from PIL import Image
        image = Image.open(str(image_path)).convert("RGB")
        image.thumbnail((inner[2] - inner[0], inner[3] - inner[1]))
        px = inner[0] + (inner[2] - inner[0] - image.width) // 2
        py = inner[1] + (inner[3] - inner[1] - image.height) // 2
        canvas.paste(image, (px, py))
    except Exception as exc:  # noqa: BLE001 - evidence should diagnose bad assets.
        draw.text((inner[0], inner[1] + 20), f"load failed: {exc!r}"[:80], fill="#ffb4ab", font=font)


def write_part_visual_eval_boards(
    part_rows: Sequence[Mapping[str, Any]],
    report_assets_dir: str | Path,
    output_dir: str | Path,
    *,
    rows_per_page: int = 4,
) -> List[str]:
    """Write mesh-first part QA boards: reference image -> generated mesh -> layout bbox.

    These are the primary VLM inputs for S2 because Stage 2 is judged on the
    generated geometry.  Layout appears only as the rightmost constraint panel.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return []

    report_assets_dir = Path(report_assets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    pages: List[str] = []
    width = 1900
    row_h = 330
    header_h = 72
    for page_index in range(0, len(part_rows), rows_per_page):
        rows = part_rows[page_index : page_index + rows_per_page]
        image = Image.new("RGB", (width, header_h + max(1, len(rows)) * row_h + 24), "#eee6d6")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, header_h), fill="#20251f")
        draw.text(
            (24, 18),
            "ArchStudio-S2 mesh-first VLM evidence: reference image → generated mesh render → layout constraint",
            fill="#fff8e8",
            font=font,
        )
        for local_index, row in enumerate(rows):
            y = header_h + local_index * row_h + 12
            draw.rounded_rectangle((16, y, width - 16, y + row_h - 12), radius=14, fill="#fffdf7", outline="#c7b996")
            part_id = str(row.get("part_id", ""))
            label = row.get("label")
            display = str(row.get("display_label") or row.get("target_class") or "")
            role = str(row.get("semantic_role") or "")
            bbox = row.get("bbox") or {}
            prompt = str(row.get("prompt") or "")
            text_x = 34
            draw.text((text_x, y + 18), f"#{label} {display}", fill="#141414", font=font)
            draw.text((text_x, y + 38), part_id[:95], fill="#4b5563", font=font)
            draw.text((text_x, y + 58), f"role={role or 'n/a'} bbox={bbox}", fill="#4b5563", font=font)
            wrapped = _wrap_text_for_image(draw, prompt, font, 520)
            for offset, text in enumerate(wrapped[:7]):
                draw.text((text_x, y + 86 + offset * 17), text, fill="#263238", font=font)

            parts_dir = report_assets_dir / "parts"
            reference = parts_dir / f"{part_id}_reference.png"
            mesh = parts_dir / f"{part_id}_mesh.png"
            layout = parts_dir / f"{part_id}_layout.png"
            _paste_thumbnail(image, draw, reference, (640, y + 18, 990, y + row_h - 28), label="T2I reference image", font=font)
            _paste_thumbnail(image, draw, mesh, (1010, y + 18, 1430, y + row_h - 28), label="Generated part mesh render", font=font)
            _paste_thumbnail(image, draw, layout, (1450, y + 18, 1870, y + row_h - 28), label="Layout bbox constraint", font=font)
        page_path = output_dir / f"part_mesh_eval_board_{len(pages):03d}.png"
        image.save(page_path)
        pages.append(str(page_path))
    return pages


def build_evidence_pack(
    run_dir: str | Path,
    output_dir: str | Path,
    texture_run_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Create report-ready visual evidence from an existing run."""
    from .visualization import (
        S2_VLM_SIX_VIEWS,
        compose_image_grid,
        export_contract_part_assembly_glb,
        export_layout_3d_boxes_glb,
        render_obj_mesh_views,
        render_contract_part_assembly_views,
        render_layout_3d_box_views,
        render_layout_projection,
        render_layout_text_callouts,
        render_layout_xy,
        render_s1_style_layout_multiview,
    )

    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(run_dir / "manifest.json")
    contract = resolve_contract_from_run(run_dir, manifest)
    part_rows = build_part_index_map(contract)
    evidence_max_faces_per_mesh = int(
        os.environ.get("S2_QUALITY_EVIDENCE_MAX_FACES_PER_MESH", DEFAULT_EVIDENCE_MAX_FACES_PER_MESH)
    )

    assets: Dict[str, Any] = {}
    assets["layout_overview"] = render_layout_xy(contract, evidence_dir / "layout_overview.png")
    assets["layout_xz"] = render_layout_projection(contract, evidence_dir / "layout_xz.png", axes=("x", "z"))
    assets["layout_yz"] = render_layout_projection(contract, evidence_dir / "layout_yz.png", axes=("y", "z"))
    assets["layout_text_callouts"] = render_layout_text_callouts(
        contract,
        evidence_dir / "layout_text_callouts_xz.png",
        axes=("x", "z"),
        title="V9 quality evidence: layout XZ part-to-text callouts",
    )
    try:
        assets["layout_s1_style_multiview"] = render_s1_style_layout_multiview(
            contract,
            evidence_dir / "layout_s1_style_visual_critic_multiview.png",
            title="ArchStudio-S2 layout evidence · S1-style six-panel numeric labels",
        )
        assets["layout_s1_style_label_coverage"] = str(
            Path(assets["layout_s1_style_multiview"]).with_name("layout_s1_style_visual_critic_multiview_label_coverage.json")
        )
    except Exception as exc:
        assets["layout_s1_style_multiview_error"] = repr(exc)
    try:
        assets["layout_3d_model"] = export_layout_3d_boxes_glb(contract, evidence_dir / "layout_3d_boxes.glb")
    except Exception as exc:
        assets["layout_3d_model_error"] = repr(exc)
    assets["layout_3d_views"] = render_layout_3d_box_views(contract, evidence_dir, "layout_3d_view", title="V9 quality evidence layout")
    assets["assembly_geometry_views"] = render_contract_part_assembly_views(
        contract,
        run_dir,
        evidence_dir,
        "assembly_geometry_view",
        title="V9 quality evidence assembly",
        max_faces_per_mesh=evidence_max_faces_per_mesh,
        include_placeholders=True,
        placeholder_alpha=0.18,
    )
    try:
        six_views = render_contract_part_assembly_views(
            contract,
            run_dir,
            evidence_dir,
            "assembly_geometry_six_view",
            title="V9 quality evidence assembly six-view",
            max_faces_per_mesh=evidence_max_faces_per_mesh,
            include_placeholders=True,
            placeholder_alpha=0.18,
            views=S2_VLM_SIX_VIEWS,
        )
        if six_views:
            assets["assembly_geometry_six_views"] = six_views
            six_board = compose_image_grid(
                [(f"assembly_{name}", path) for name, path in six_views.items()],
                evidence_dir / "assembly_geometry_six_view_board.png",
                "ArchStudio-S2 generated mesh assembly · six-view VLM input",
                columns=3,
            )
            if six_board:
                assets["assembly_geometry_six_view_board"] = six_board
    except Exception as exc:
        assets["assembly_geometry_six_view_error"] = repr(exc)
    try:
        assets["assembly_geometry_model"] = export_contract_part_assembly_glb(
            contract,
            run_dir,
            evidence_dir / "assembly_geometry_per_part_color.glb",
            max_vertices_per_part=65000,
            include_placeholders=True,
            placeholder_alpha=80,
        )
    except Exception as exc:
        assets["assembly_geometry_model_error"] = repr(exc)

    texture_manifest = load_texture_manifest(texture_run_dir)
    if texture_manifest.get("assembly_glb"):
        copied = _copy_if_exists(texture_manifest.get("assembly_glb"), evidence_dir / "textured_assembly.glb")
        if copied:
            assets["textured_assembly_model"] = copied
    if texture_manifest.get("assembly_obj"):
        texture_assembly_obj = Path(str(texture_manifest.get("assembly_obj")))
        if texture_assembly_obj.exists():
            assets["textured_assembly_views"] = render_obj_mesh_views(
                texture_assembly_obj,
                evidence_dir,
                "textured_assembly_view",
                title="V9 quality evidence textured assembly",
                max_faces=evidence_max_faces_per_mesh,
                use_material_colors=True,
            )
    if texture_run_dir and (Path(texture_run_dir) / "report" / "index.html").exists():
        assets["texture_report"] = str(Path(texture_run_dir) / "report" / "index.html")

    partboard_paths = write_partboard_images(part_rows, evidence_dir)
    assets["partboards"] = partboard_paths
    part_eval_boards = write_part_visual_eval_boards(part_rows, run_dir / "report" / "assets", evidence_dir)
    if part_eval_boards:
        assets["part_mesh_eval_boards"] = part_eval_boards
    render_notes = build_evidence_render_notes(
        run_dir,
        contract,
        manifest,
        max_faces_per_mesh=evidence_max_faces_per_mesh,
    )
    assets["render_notes"] = str(evidence_dir / "render_notes.json")
    write_json(evidence_dir / "part_index_map.json", part_rows)
    write_json(evidence_dir / "render_notes.json", render_notes)
    write_json(evidence_dir / "evidence_manifest.json", assets)
    return {
        "schema_version": "archstudio_s2_quality_evidence.v1",
        "run_dir": str(run_dir),
        "texture_run_dir": str(Path(texture_run_dir).resolve()) if texture_run_dir else None,
        "evidence_dir": str(evidence_dir),
        "part_index_map": str(evidence_dir / "part_index_map.json"),
        "assets": assets,
    }


def build_report_asset_evidence_pack(
    run_dir: str | Path,
    output_dir: str | Path,
    texture_run_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Create a lightweight evidence pack by reusing existing report assets.

    Full evidence rendering samples every mesh and can be expensive for dense
    Hunyuan outputs.  Agent iterations usually run after a normal S2 report has
    already produced layout/assembly PNGs and GLBs, so this pack references
    those assets directly and only writes cheap JSON/partboard artifacts.
    """
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(run_dir / "manifest.json")
    contract = resolve_contract_from_run(run_dir, manifest)
    part_rows = build_part_index_map(contract)
    report_assets = run_dir / "report" / "assets"

    def existing(name: str) -> Optional[str]:
        path = report_assets / name
        return str(path) if path.exists() else None

    def view_group(prefix: str) -> Dict[str, str]:
        views = {}
        for name in ("iso", "front", "side", "top"):
            value = existing(f"{prefix}_{name}.png")
            if value:
                views[name] = value
        return views

    assets: Dict[str, Any] = {
        "layout_overview": existing("layout_overview.png"),
        "layout_xz": existing("layout_xz.png"),
        "layout_yz": existing("layout_yz.png"),
        "layout_text_callouts": existing("layout_text_callouts_xz.png"),
        "layout_s1_style_multiview": existing("layout_s1_style_visual_critic_multiview.png"),
        "layout_s1_style_label_coverage": existing("layout_s1_style_visual_critic_multiview_label_coverage.json"),
        "layout_3d_model": existing("layout_3d_boxes.glb"),
        "layout_3d_views": view_group("layout_3d_view"),
        "assembly_geometry_model": (
            existing("assembly_complete_per_part_color_yup.glb")
            or existing("assembly_raw_per_part_color_yup.glb")
        ),
        "assembly_geometry_views": (
            view_group("assembly_complete_view")
            or view_group("assembly_raw_view")
        ),
    }
    try:
        from .visualization import S2_VLM_SIX_VIEWS, compose_image_grid, render_contract_part_assembly_views, render_s1_style_layout_multiview
        if not assets.get("layout_s1_style_multiview"):
            assets["layout_s1_style_multiview"] = render_s1_style_layout_multiview(
                contract,
                evidence_dir / "layout_s1_style_visual_critic_multiview.png",
                title="ArchStudio-S2 layout evidence · S1-style six-panel numeric labels",
            )
            assets["layout_s1_style_label_coverage"] = str(
                Path(assets["layout_s1_style_multiview"]).with_name("layout_s1_style_visual_critic_multiview_label_coverage.json")
            )
        six_views = render_contract_part_assembly_views(
            contract,
            run_dir,
            evidence_dir,
            "assembly_geometry_six_view",
            title="ArchStudio-S2 generated mesh assembly six-view",
            max_faces_per_mesh=int(os.environ.get("S2_QUALITY_EVIDENCE_MAX_FACES_PER_MESH", DEFAULT_EVIDENCE_MAX_FACES_PER_MESH)),
            include_placeholders=True,
            placeholder_alpha=0.18,
            views=S2_VLM_SIX_VIEWS,
        )
        if six_views:
            assets["assembly_geometry_six_views"] = six_views
            six_board = compose_image_grid(
                [(f"assembly_{name}", path) for name, path in six_views.items()],
                evidence_dir / "assembly_geometry_six_view_board.png",
                "ArchStudio-S2 generated mesh assembly · six-view VLM input",
                columns=3,
            )
            if six_board:
                assets["assembly_geometry_six_view_board"] = six_board
    except Exception as exc:
        assets["s1_style_or_six_view_evidence_error"] = repr(exc)

    assets = {key: value for key, value in assets.items() if value}

    texture_manifest = load_texture_manifest(texture_run_dir)
    if texture_manifest.get("assembly_glb"):
        assets["textured_assembly_model"] = texture_manifest.get("assembly_glb")
    if texture_run_dir and (Path(texture_run_dir) / "report" / "index.html").exists():
        assets["texture_report"] = str(Path(texture_run_dir) / "report" / "index.html")

    partboard_paths = write_partboard_images(part_rows, evidence_dir)
    if partboard_paths:
        assets["partboards"] = partboard_paths
    part_eval_boards = write_part_visual_eval_boards(part_rows, report_assets, evidence_dir)
    if part_eval_boards:
        assets["part_mesh_eval_boards"] = part_eval_boards
    write_json(evidence_dir / "part_index_map.json", part_rows)
    write_json(evidence_dir / "evidence_manifest.json", assets)
    return {
        "schema_version": "archstudio_s2_quality_evidence.v1",
        "run_dir": str(run_dir),
        "texture_run_dir": str(Path(texture_run_dir).resolve()) if texture_run_dir else None,
        "evidence_dir": str(evidence_dir),
        "part_index_map": str(evidence_dir / "part_index_map.json"),
        "assets": assets,
        "source": "existing_report_assets_fast",
    }


def evaluate_with_mock(
    metrics: Mapping[str, Any],
    mock_response: str | Path | Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if mock_response is None:
        payload = deterministic_findings_from_metrics(metrics)
    elif isinstance(mock_response, Mapping):
        payload = dict(mock_response)
    else:
        text = Path(mock_response).read_text(encoding="utf-8") if Path(str(mock_response)).exists() else str(mock_response)
        payload = parse_json_object(text)
    part_ids = [str(part.get("part_id")) for part in metrics.get("parts", []) or []]
    return validate_quality_findings(payload, part_ids)


def _image_data_url(path: str | Path) -> str:
    path = Path(path)
    compress = os.environ.get("S2_VLM_COMPRESS_IMAGES", "1").strip().lower() not in {"0", "false", "no", "off"}
    if compress:
        try:
            from io import BytesIO
            from PIL import Image

            max_side = max(256, int(os.environ.get("S2_VLM_IMAGE_MAX_SIDE", "1280")))
            quality = max(40, min(95, int(os.environ.get("S2_VLM_IMAGE_JPEG_QUALITY", "82"))))
            image = Image.open(path).convert("RGB")
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            return "data:image/jpeg;base64,{}".format(base64.b64encode(buffer.getvalue()).decode("ascii"))
        except Exception:
            pass
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    return "data:{};base64,{}".format(
        mime_type,
        base64.b64encode(path.read_bytes()).decode("ascii"),
    )


def _compact_metrics_for_evaluator(metrics: Mapping[str, Any], max_parts: int = 80) -> Dict[str, Any]:
    parts = []
    for part in list(metrics.get("parts", []) or [])[:max_parts]:
        parts.append({
            "label": int(part.get("index", 0)) + 1,
            "part_id": part.get("part_id"),
            "display_label": part.get("display_label"),
            "semantic_role": part.get("semantic_role"),
            "detail_level": part.get("detail_level"),
            "spatial_relations": part.get("spatial_relations", []),
            "bbox_extent": (part.get("bbox") or {}).get("extent"),
            "mesh_extent": (part.get("mesh") or {}).get("extent"),
            "axis_fill_ratio": part.get("axis_fill_ratio"),
            "percentile_axis_fill_ratio": part.get("percentile_axis_fill_ratio"),
            "center_delta": part.get("center_delta"),
            "issue_flags": part.get("issue_flags", []),
            "texture_status": part.get("texture_status"),
        })
    return {
        "schema_version": metrics.get("schema_version"),
        "layout_id": metrics.get("layout_id"),
        "num_parts": metrics.get("num_parts"),
        "num_meshes_present": metrics.get("num_meshes_present"),
        "summary": metrics.get("summary", {}),
        "scene_bounds": metrics.get("scene_bounds"),
        "coordinate_frame": metrics.get("coordinate_frame"),
        "geometry_continuity_contract": metrics.get("geometry_continuity_contract"),
        "mesh_contact_gaps": [
            gap for gap in metrics.get("mesh_contact_gaps", [])
            if gap.get("above_tolerance")
        ][:80],
        "parts": parts,
    }


def _resolve_evidence_image_paths(evidence: Mapping[str, Any] | None, *, max_images: int = 12) -> List[Tuple[str, Path]]:
    if not evidence:
        return []
    assets = evidence.get("assets", {}) if isinstance(evidence, Mapping) else {}
    if not isinstance(assets, Mapping):
        return []

    preferred: List[Tuple[str, str | Path]] = []
    # S2 is judged on generated 3D assets.  Mesh/part evidence must dominate the
    # VLM context; layout is only a constraint reference near the end.
    if assets.get("assembly_geometry_six_view_board"):
        preferred.append(("assembly_geometry_six_view_board", assets["assembly_geometry_six_view_board"]))
    for group_key in ("assembly_geometry_six_views", "assembly_geometry_views", "textured_assembly_views"):
        value = assets.get(group_key)
        if isinstance(value, Mapping):
            for name in ("iso", "front", "back", "left", "right", "side", "top"):
                if value.get(name):
                    preferred.append((f"{group_key}.{name}", value[name]))
    part_eval_boards = assets.get("part_mesh_eval_boards")
    if isinstance(part_eval_boards, list):
        for index, path in enumerate(part_eval_boards):
            preferred.append((f"part_mesh_eval_board_{index:03d}", path))
    partboards = assets.get("partboards")
    if isinstance(partboards, list):
        for index, path in enumerate(partboards):
            preferred.append((f"partboard_{index:03d}", path))
    if assets.get("layout_s1_style_multiview"):
        preferred.append(("layout_s1_style_multiview", assets["layout_s1_style_multiview"]))
    for group_key in ("layout_3d_views",):
        value = assets.get(group_key)
        if isinstance(value, Mapping):
            for name in ("iso", "front", "back", "left", "right", "side", "top"):
                if value.get(name):
                    preferred.append((f"{group_key}.{name}", value[name]))
    for key in ("layout_overview", "layout_xz", "layout_yz"):
        value = assets.get(key)
        if value:
            preferred.append((key, value))

    resolved: List[Tuple[str, Path]] = []
    for label, path in preferred:
        candidate = Path(path)
        if candidate.exists() and candidate.is_file():
            resolved.append((label, candidate))
        if len(resolved) >= max_images:
            break
    return resolved


def _safe_artifact_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    text = "_".join(part for part in text.split("_") if part)
    return text[:120] or "artifact"


def _wrap_text_for_image(draw: Any, text: str, font: Any, max_width: int) -> List[str]:
    """Wrap text using Pillow's measured width when available."""
    words = str(text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        try:
            width = draw.textlength(trial, font=font)
        except Exception:
            width = len(trial) * 7
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def compose_visual_critic_board(
    evidence: Mapping[str, Any] | None,
    output_path: str | Path,
    *,
    title: str = "ArchStudio-S2 GPT-5.5 visual critic evidence board",
    max_images: int = 8,
) -> Optional[str]:
    """Compose S1-style single-board visual evidence from S2 report assets.

    Stage1's visual critic asks GPT-5.5 with one readable numbered multiview
    image.  S2 naturally has several evidence images (layout callouts, layout
    views, assembly views, partboards).  This helper preserves the same
    inspectability by tiling the ordered VLM input images into a single board and
    labeling each tile with the exact evidence name used in the prompt.
    """
    images = _resolve_evidence_image_paths(evidence, max_images=max_images)
    if not images:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    title_font = font
    tile_w = 620
    tile_h = 390
    label_h = 44
    margin = 24
    gap = 18
    columns = 2
    rows = int(math.ceil(len(images) / float(columns)))
    width = columns * tile_w + (columns - 1) * gap + 2 * margin
    height = 72 + rows * (tile_h + label_h) + (rows - 1) * gap + margin
    board = Image.new("RGB", (width, height), "#f4efe5")
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, width, 64), fill="#22251f")
    draw.text((margin, 18), title, fill="#fff8e8", font=title_font)

    for index, (label, source_path) in enumerate(images, start=1):
        row = (index - 1) // columns
        col = (index - 1) % columns
        x = margin + col * (tile_w + gap)
        y = 72 + row * (tile_h + label_h + gap)
        draw.rounded_rectangle((x, y, x + tile_w, y + tile_h + label_h), radius=12, fill="#fffdf7", outline="#c9bda5")
        try:
            image = Image.open(source_path).convert("RGB")
            image.thumbnail((tile_w - 20, tile_h - 20), Image.LANCZOS)
            px = x + (tile_w - image.width) // 2
            py = y + 10 + (tile_h - 20 - image.height) // 2
            draw.rectangle((x + 10, y + 10, x + tile_w - 10, y + tile_h - 10), fill="#111318")
            board.paste(image, (px, py))
        except Exception as exc:  # noqa: BLE001 - board should still diagnose bad assets.
            draw.rectangle((x + 10, y + 10, x + tile_w - 10, y + tile_h - 10), fill="#fff1f1")
            draw.text((x + 18, y + 22), f"failed to load image: {exc!r}", fill="#8c1d18", font=font)
        label_text = f"{index:02d}. {label}"
        draw.rounded_rectangle((x + 12, y + tile_h + 8, x + tile_w - 12, y + tile_h + label_h - 8), radius=7, fill="#2f4b67")
        draw.text((x + 22, y + tile_h + 18), label_text[:90], fill="#ffffff", font=font)

    board.save(output_path)
    return str(output_path)


def extract_visual_evaluator_questions(prompt_text: str) -> List[str]:
    try:
        payload = parse_json_object(prompt_text)
    except Exception:
        return list(S2_VISUAL_CRITIC_QUESTIONS)
    instructions = payload.get("instructions") if isinstance(payload, Mapping) else None
    questions = list(S2_VISUAL_CRITIC_QUESTIONS)
    if isinstance(instructions, list):
        # Keep S1-style explicit questions while preserving the actual request
        # constraints that were sent to the VLM.
        questions.extend(str(item) for item in instructions[:8])
    return questions


def write_s2_visual_critic_sidecars(
    *,
    run_dir: str | Path,
    iteration: int,
    provider: str,
    model: str | None,
    base_url: str | None,
    request_manifest: Mapping[str, Any] | None,
    response_manifest: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Write S1-compatible QA and repair-plan JSON sidecars for S2."""
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    question_path = Path(str((request_manifest or {}).get("prompt_path") or output_dir / "question_prompt.txt"))
    answer_path = output_dir / "answer.json"
    prompt_text = question_path.read_text(encoding="utf-8", errors="ignore") if question_path.exists() else ""
    findings = (response_manifest or {}).get("findings") if isinstance(response_manifest, Mapping) else None
    if not isinstance(findings, Mapping):
        findings = {}
    issues = findings.get("issues", []) if isinstance(findings.get("issues"), list) else []
    verdict = "accept" if not issues and findings else "revise" if issues else "skipped"
    critic_board = compose_visual_critic_board(
        evidence,
        output_dir / "s2_visual_critic_readable_label_multiview.png",
        title=f"ArchStudio-S2 visual critic · iteration {iteration:03d}",
        max_images=int((request_manifest or {}).get("max_images") or 8),
    )
    image_rel = None
    if critic_board:
        try:
            image_rel = os.path.relpath(critic_board, run_dir)
        except Exception:
            image_rel = critic_board
    qa = {
        "schema": "archstudio_s2.visual_critic_qa.v1",
        "critic_mode": provider,
        "model": model,
        "base_url": str(base_url).split("?", 1)[0] if base_url else None,
        "run_id": run_dir.name,
        "layout_id": metrics.get("layout_id"),
        "iteration": int(iteration),
        "image": image_rel,
        "image_type": "mesh_six_view_board_and_s1_style_layout_multiview",
        "images": list((request_manifest or {}).get("input_images", []) or []),
        "questions": extract_visual_evaluator_questions(prompt_text),
        "question_prompt_path": str(question_path) if question_path else None,
        "answer_path": str(answer_path) if answer_path.exists() else None,
        "raw_text": None,
        "answer": {
            "score": findings.get("scene_score"),
            "verdict": verdict,
            "grounded": verdict == "accept",
            "physically_plausible": verdict == "accept",
            "summary": findings.get("global_summary") or findings.get("global_style_diagnosis") or "",
            "global_style_diagnosis": findings.get("global_style_diagnosis"),
            "findings": [
                {
                    "question": "S2 part/geometry visual quality",
                    "answer": issue.get("diagnosis"),
                    "severity": issue.get("severity"),
                    "part_numbers": issue.get("labels", []),
                    "part_ids": issue.get("part_ids", []),
                    "issue_type": issue.get("issue_type"),
                    "evidence": issue.get("evidence", []),
                    "recommended_action": issue.get("recommended_action"),
                    "repair_instruction": issue.get("repair_instruction"),
                }
                for issue in issues
            ],
            "recommended_repairs": [
                {
                    "action": issue.get("recommended_action"),
                    "part_numbers": issue.get("labels", []),
                    "part_ids": issue.get("part_ids", []),
                    "rationale": issue.get("diagnosis"),
                    "instruction": issue.get("repair_instruction"),
                    "issue_type": issue.get("issue_type"),
                    "severity": issue.get("severity"),
                }
                for issue in issues
            ],
        },
        "metrics_summary": dict(metrics.get("summary", {})),
        "response_manifest_path": str(answer_path) if answer_path.exists() else None,
    }
    repair_plan = {
        "schema": "archstudio_s2.visual_critic_repair_plan.v1",
        "run_id": run_dir.name,
        "layout_id": metrics.get("layout_id"),
        "iteration": int(iteration),
        "critic_mode": provider,
        "source_image": qa.get("image"),
        "image_type": qa.get("image_type"),
        "status": "proposed_for_agent_execution" if issues else "no_repair_needed" if findings else "skipped_or_error",
        "note": "S2 sidecar mirrors Stage1 visual_critic_qa/repair_plan so 9999 can show the asked image, questions, answer, and local repair intent.",
        "recommended_repairs": qa["answer"]["recommended_repairs"],
    }
    qa_path = output_dir / f"visual_critic_qa_iteration_{iteration:03d}.json"
    plan_path = output_dir / f"visual_critic_repair_plan_iteration_{iteration:03d}.json"
    write_json(qa_path, qa)
    write_json(plan_path, repair_plan)
    if iteration == 0:
        write_json(output_dir / "visual_critic_qa.json", qa)
        write_json(output_dir / "visual_critic_repair_plan.json", repair_plan)
    return {
        "qa_path": str(qa_path),
        "repair_plan_path": str(plan_path),
        "critic_board_path": critic_board or "",
    }


def write_visual_evaluator_request_artifacts(
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    output_dir: str | Path,
    *,
    max_images: int = 12,
) -> Dict[str, Any]:
    """Persist the exact visual-evaluator question and image set.

    The VLM request was previously implicit inside ``evaluate_quality``.  The S2
    agent report needs the same evidence humans want to inspect: the prompt
    sent to the judge, the ordered images, and copy-stable paths that remain
    valid even if downstream report assets are later rebuilt.
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "input_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_visual_evaluator_prompt(metrics, evidence)
    prompt_path = output_dir / "question_prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    images = []
    for index, (label, source_path) in enumerate(_resolve_evidence_image_paths(evidence, max_images=max_images), start=1):
        suffix = source_path.suffix or ".png"
        copied = image_dir / f"{index:02d}_{_safe_artifact_name(label)}{suffix}"
        try:
            shutil.copy2(source_path, copied)
            copied_path = str(copied)
            copy_status = "copied"
        except Exception as exc:  # noqa: BLE001 - request manifest should diagnose missing images.
            copied_path = None
            copy_status = f"copy_failed:{repr(exc)}"
        images.append(
            {
                "index": index,
                "label": label,
                "source_path": str(source_path),
                "copied_path": copied_path,
                "copy_status": copy_status,
            }
        )
    critic_board_path = compose_visual_critic_board(
        evidence,
        output_dir / "s2_visual_critic_readable_label_multiview.png",
        title="ArchStudio-S2 visual critic input board",
        max_images=max_images,
    )

    manifest = {
        "schema_version": "archstudio_s2_visual_evaluator_request.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_path": str(prompt_path),
        "questions": extract_visual_evaluator_questions(prompt),
        "critic_board_path": critic_board_path,
        "image_type": "mesh_six_view_board_and_s1_style_layout_multiview",
        "max_images": int(max_images),
        "image_count": len(images),
        "input_images": images,
        "evidence_dir": evidence.get("evidence_dir") if isinstance(evidence, Mapping) else None,
        "metrics_summary": dict(metrics.get("summary", {})),
    }
    write_json(output_dir / "request_manifest.json", manifest)
    return manifest


def write_visual_evaluator_response_artifacts(
    output_dir: str | Path,
    *,
    findings: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "archstudio_s2_visual_evaluator_response.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "error" if error else "completed" if findings else "skipped",
        "error": error,
        "findings": dict(findings) if isinstance(findings, Mapping) else None,
        "issue_count": len(findings.get("issues", []) or []) if isinstance(findings, Mapping) else 0,
    }
    write_json(output_dir / "answer.json", payload)
    return payload


def _evidence_render_notes_for_prompt(evidence: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not evidence or not isinstance(evidence, Mapping):
        return None
    assets = evidence.get("assets", {})
    if not isinstance(assets, Mapping):
        return None
    notes_path = assets.get("render_notes")
    if not notes_path:
        return None
    try:
        notes = load_json(notes_path)
    except Exception:
        return None
    parts = []
    for part in notes.get("parts", [])[:80]:
        mesh = part.get("mesh") or {}
        parts.append({
            "label": part.get("label"),
            "part_id": part.get("part_id"),
            "display_label": part.get("display_label"),
            "vertex_count": mesh.get("vertex_count"),
            "face_count": mesh.get("face_count"),
            "render_face_fraction": mesh.get("render_face_fraction"),
        })
    return {
        "raster_renderer": notes.get("raster_renderer"),
        "rendered_face_cap_per_mesh": notes.get("rendered_face_cap_per_mesh"),
        "interpretation_warning": notes.get("interpretation_warning"),
        "ground_truth_geometry_assets": notes.get("ground_truth_geometry_assets"),
        "parts": parts,
    }


def _copy_named_image(source: str | Path | None, destination: Path) -> Optional[str]:
    if not source:
        return None
    source_path = Path(str(source))
    if not source_path.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination)



def _extract_sse_chat_content(text: str) -> str:
    """Extract assistant content from OpenAI-compatible SSE chat chunks.

    Some gateways return a raw ``data: ...`` stream even for non-streaming
    client calls.  The final usage chunk can itself be valid JSON but contains
    no assistant answer; treating that object as the model answer silently marks
    failed VLM calls as completed.
    """
    pieces: List[str] = []
    saw_sse = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        saw_sse = True
        data = stripped[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta") if isinstance(choice.get("delta"), Mapping) else {}
            message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
            content = delta.get("content") if delta else None
            if content is None and message:
                content = message.get("content")
            if isinstance(content, str):
                pieces.append(content)
    if pieces:
        return "".join(pieces).strip()
    return "" if saw_sse else str(text or "")


def _completion_raw_and_message_text(completion: Any) -> Tuple[str, str]:
    """Return (raw_debug_text, assistant_message_text) from SDK/dict/string responses."""
    if isinstance(completion, str):
        raw_text = completion
        return raw_text, _extract_sse_chat_content(raw_text)

    if isinstance(completion, Mapping):
        raw_text = json.dumps(completion, ensure_ascii=False)
        choices = completion.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") if isinstance(first_choice, Mapping) else {}
        delta = first_choice.get("delta") if isinstance(first_choice, Mapping) else {}
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            return raw_text, message.get("content", "")
        if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
            return raw_text, delta.get("content", "")
        return raw_text, _extract_sse_chat_content(raw_text)

    if hasattr(completion, "model_dump_json"):
        try:
            raw_text = completion.model_dump_json()
        except Exception:
            raw_text = str(completion)
    else:
        raw_text = str(completion)
    try:
        choices = getattr(completion, "choices", None) or []
        first_choice = choices[0] if choices else None
        message = getattr(first_choice, "message", None)
        if message is not None and isinstance(getattr(message, "content", None), str):
            return raw_text, getattr(message, "content") or ""
        delta = getattr(first_choice, "delta", None) if first_choice is not None else None
        if delta is not None and isinstance(getattr(delta, "content", None), str):
            return raw_text, getattr(delta, "content") or ""
    except Exception:
        pass
    return raw_text, _extract_sse_chat_content(raw_text)


def _validate_focused_visual_critic_answer(payload: Mapping[str, Any]) -> Dict[str, Any]:
    required = {"score", "verdict", "summary", "issues"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"focused visual critic response missing required keys: {', '.join(missing)}")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("focused visual critic response field 'issues' must be a list")
    return dict(payload)

def _call_openai_json_with_images(
    *,
    prompt: Mapping[str, Any],
    image_paths: Sequence[Tuple[str, str | Path]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    auth_path: Optional[str | Path] = None,
    max_tokens: int = 4096,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("openai and httpx packages are required for focused visual critics") from exc

    model = model or os.environ.get("S2_QUALITY_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MULTIMODAL_MODEL
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_AICODEMIRROR_BASE_URL).rstrip("/")
    api_key = load_openai_api_key(auth_path)
    trust_env = os.environ.get("S2_OPENAI_TRUST_ENV", "0").strip().lower() in {"1", "true", "yes", "on"}
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=trust_env, timeout=180.0),
    )
    compact_instruction = {
        "response_contract": "Return one compact JSON object only. Do not include markdown. Keep summary and repair_hint short.",
        "prompt": prompt,
    }
    content: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(compact_instruction, ensure_ascii=False, separators=(",", ":"))}]
    image_detail = os.environ.get("S2_VLM_IMAGE_DETAIL", "low").strip().lower() or "low"
    for label, path in image_paths:
        content.append({"type": "text", "text": f"Focused critic input image: {label}"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path), "detail": image_detail}})
    max_tokens = max(max_tokens, int(os.environ.get("S2_FOCUSED_CRITIC_MAX_TOKENS", str(max_tokens))))
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a focused ArchStudio-S2 visual critic. Return only a compact valid JSON object. No markdown, no prose outside JSON."},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    attempts: List[Tuple[str, Dict[str, Any]]] = [("json_object", kwargs)]
    fallback_kwargs = dict(kwargs)
    fallback_kwargs.pop("response_format", None)
    attempts.append(("plain", fallback_kwargs))
    completion_kwargs = dict(fallback_kwargs)
    completion_kwargs.pop("max_tokens", None)
    completion_kwargs["max_completion_tokens"] = max_tokens
    attempts.append(("plain_max_completion_tokens", completion_kwargs))

    errors: List[str] = []
    last_raw_text = ""
    retry_count = max(0, int(os.environ.get("S2_FOCUSED_CRITIC_REQUEST_RETRIES", "3")))
    retry_sleep = max(0.0, float(os.environ.get("S2_FOCUSED_CRITIC_RETRY_SLEEP", "4")))
    for attempt_name, attempt_kwargs in attempts:
        for request_index in range(retry_count + 1):
            try:
                completion = client.chat.completions.create(**attempt_kwargs)
                raw_response_text, message_text = _completion_raw_and_message_text(completion)
                last_raw_text = raw_response_text
                if not str(message_text or "").strip():
                    raise ValueError("provider returned no assistant message content")
                payload = _validate_focused_visual_critic_answer(parse_json_object(message_text))
                metadata = {
                    "provider": "openai_multimodal",
                    "model": model,
                    "base_url": base_url.split("?", 1)[0],
                    "image_count": len(image_paths),
                    "attempt": attempt_name,
                    "request_retry_index": request_index,
                    "request_retry_count": retry_count,
                    "raw_text": message_text,
                    "raw_response_text": raw_response_text,
                }
                return payload, metadata
            except Exception as exc:  # noqa: BLE001 - retry transient empty/invalid provider responses.
                snippet = str(last_raw_text or "")[:500].replace("\n", " ")
                errors.append(f"{attempt_name}[{request_index}/{retry_count}]: {type(exc).__name__}: {exc}; raw={snippet}")
                if request_index < retry_count:
                    time.sleep(retry_sleep * min(3, request_index + 1))

    raise RuntimeError("Focused OpenAI-compatible visual critic failed validation after retries: " + " | ".join(errors))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 3) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _focused_issue_type(scope: str, issue_type: Any, summary: Any = "") -> str:
    """Map focused-critic local labels into the shared S2 quality taxonomy."""
    raw = str(issue_type or "").strip().lower()
    text = f"{raw} {summary}".lower()
    if "penetrat" in text or "intersect" in text or "cut through" in text:
        return "penetration"
    if "overlap" in text:
        return "overlap"
    if "float" in text or "floating" in text or "hang" in text:
        return "floating"
    if "gap" in text or "seam" in text or "detach" in text or "sink" in text:
        return "gap_or_seam"
    if "geometry_contract" in text or "i23d_prior" in text or "layout_contract" in text:
        return "scale_mismatch"
    if "aspect" in text or "scale" in text or "bbox" in text or "size" in text or "prior" in text:
        return "scale_mismatch"
    if "whole" in text or "unrelated" in text or "semantic" in text or "wrong" in text:
        return "wrong_semantics"
    if "background" in text or "base" in text or "panel" in text or "pedestal" in text or "floor" in text:
        return "artifact_base_or_panel"
    if "orient" in text or "gravity" in text or "sideways" in text or "upside" in text or "rotat" in text:
        return "wrong_orientation"
    if "style" in text or "coherent" in text:
        return "style_inconsistent" if scope != "scene_assembly" else "global_style_issue"
    if "missing" in text:
        return "missing_part"
    if scope == "part_image":
        return "wrong_semantics"
    if scope == "part_mesh":
        return "low_geometry_quality"
    return "global_style_issue"


def _focused_recommended_action(scope: str, issue_type: str, repair_route: Any = None) -> str:
    route = str(repair_route or "").strip().lower().replace("-", "_")
    if route in {"deterministic_snap", "snap", "resize_snap", "deterministic_resize_snap", "assembly_snap"}:
        return "deterministic_resize_snap"
    if route in {"rerun_i3d", "rerun_i23d", "rerun_geometry", "i3d"}:
        return "rerun_geometry"
    if route in {"rerun_t2i_i3d", "rerun_t2i_geometry", "rerun_t2i", "t2i_i3d"}:
        return "rerun_t2i_geometry_texture"
    if route in {"manual", "manual_review", "review"}:
        return "manual_review"
    if scope == "part_image":
        return "rerun_t2i_geometry_texture"
    if issue_type in {"wrong_semantics", "artifact_base_or_panel"}:
        return "rerun_t2i_geometry_texture"
    if issue_type in {"scale_mismatch", "gap_or_seam", "overlap", "floating", "penetration"}:
        return "deterministic_resize_snap"
    if issue_type in {"wrong_orientation", "low_geometry_quality", "missing_part"}:
        return "rerun_geometry"
    if issue_type in {"style_inconsistent", "global_style_issue"}:
        return "manual_review"
    return "manual_review"


def _focused_severity(score: float, raw_severity: Any = None) -> int:
    severity = _safe_int(raw_severity, 0)
    if severity > 0:
        return max(1, min(5, severity))
    if score < 0.30:
        return 5
    if score < 0.50:
        return 4
    if score < 0.72:
        return 3
    return 2


def _load_focused_answer(result: Mapping[str, Any]) -> Dict[str, Any]:
    path = result.get("answer_path")
    if path:
        try:
            payload = load_json(path)
            nested = payload.get("answer") if isinstance(payload, Mapping) else None
            if isinstance(nested, Mapping):
                return dict(nested)
        except Exception:
            pass
    answer = result.get("answer")
    if isinstance(answer, Mapping):
        return dict(answer)
    path = None
    if path:
        try:
            payload = load_json(path)
            nested = payload.get("answer") if isinstance(payload, Mapping) else None
            if isinstance(nested, Mapping):
                return dict(nested)
        except Exception:
            return {}
    return {}


def quality_findings_from_focused_visual_critics(
    focused: Mapping[str, Any],
    *,
    valid_part_ids: Optional[Iterable[str]] = None,
    score_threshold: float = 0.72,
) -> Dict[str, Any]:
    """Convert focused per-stage VLM answers into the shared quality schema.

    The focused critic requests deliberately use local, stage-specific issue
    labels (for example ``whole_building`` in the image critic).  The repair
    loop already consumes ``archstudio_s2_quality_findings.v1``.  This adapter
    keeps the VLM calls small and single-purpose while preserving the existing
    reject/accept operation planner.
    """
    valid = set(str(item) for item in valid_part_ids or [])
    issues: List[Dict[str, Any]] = []
    scores: List[float] = []
    counter = 1
    label_to_part_id: Dict[int, str] = {}
    for row in focused.get("part_index", []) or []:
        if not isinstance(row, Mapping):
            continue
        label = _safe_int(row.get("label"), 0)
        part_id_value = row.get("part_id")
        if label > 0 and part_id_value:
            label_to_part_id[label] = str(part_id_value)

    for result in focused.get("results", []) or []:
        if not isinstance(result, Mapping):
            continue
        scope = str(result.get("scope") or "")
        part_id = str(result.get("part_id") or "")
        case_id = str(result.get("case_id") or "")
        answer = _load_focused_answer(result)
        score = _safe_float(answer.get("score", result.get("score")), 0.0)
        scores.append(score)
        verdict = str(answer.get("final_verdict") or answer.get("verdict") or result.get("verdict", "")).strip().lower()
        sub_verdict_keys = ("semantic_verdict", "geometry_contract_verdict", "i23d_prior_verdict")
        sub_verdicts = {
            key: str(answer.get(key) or "").strip().lower()
            for key in sub_verdict_keys
            if answer.get(key) is not None
        }
        blocking_sub_verdicts = {
            key: value
            for key, value in sub_verdicts.items()
            if value in {"revise", "reject", "failed", "fail"}
        }
        if blocking_sub_verdicts and verdict in {"", "accept", "accepted", "pass", "passed"}:
            verdict = "revise"
        answer_issues = answer.get("issues") if isinstance(answer.get("issues"), list) else []
        if blocking_sub_verdicts:
            existing_text = json.dumps(answer_issues, ensure_ascii=False).lower()
            if "geometry_contract" not in existing_text and "i23d_prior" not in existing_text:
                answer_issues = list(answer_issues) + [{
                    "issue_type": "geometry_contract_mismatch" if "geometry_contract_verdict" in blocking_sub_verdicts else "i23d_prior_mismatch",
                    "severity": 4,
                    "evidence": f"sub_verdicts={blocking_sub_verdicts}; semantic_verdict={sub_verdicts.get('semantic_verdict', '')}",
                    "repair_route": "rerun_t2i_i3d" if scope == "part_image" else "rerun_t2i_i3d",
                    "repair_hint": answer.get("repair_hint") or answer.get("summary") or "Regenerate a geometry-contract-compliant T2I prior, then rerun geometry.",
                }]
        needs_issue = (
            result.get("status") != "completed"
            or verdict in {"revise", "reject", "failed", "fail"}
            or score < float(score_threshold)
        )
        if needs_issue and not answer_issues:
            answer_issues = [{
                "issue_type": "focused_stage_rejected",
                "severity": _focused_severity(score),
                "evidence": answer.get("summary") or result.get("summary") or case_id,
                "repair_route": answer.get("repair_route") or ("rerun_t2i_i3d" if scope == "part_image" else ""),
                "repair_hint": answer.get("repair_hint") or answer.get("summary") or "rerun or repair this focused stage",
            }]
        for raw_issue in answer_issues:
            if not isinstance(raw_issue, Mapping):
                raw_issue = {"issue_type": str(raw_issue)}
            raw_severity = _focused_severity(score, raw_issue.get("severity"))
            if verdict in {"accept", "accepted", "pass", "passed"} and score >= float(score_threshold) and raw_severity < 3:
                continue
            issue_type = _focused_issue_type(
                scope,
                raw_issue.get("issue_type"),
                raw_issue.get("evidence") or answer.get("summary") or result.get("summary") or "",
            )
            root_cause = str(raw_issue.get("root_cause") or answer.get("root_cause") or "").strip().lower().replace("-", "_")
            repair_route_override = raw_issue.get("repair_route") or answer.get("repair_route")
            if scope == "part_mesh" and root_cause in {"source_image", "bad_source_image", "t2i", "t2i_prior", "layout_contract"}:
                repair_route_override = repair_route_override or "rerun_t2i_i3d"
                if issue_type == "low_geometry_quality":
                    issue_type = "wrong_semantics"
            part_ids: List[str] = []
            raw_part_ids = raw_issue.get("part_ids")
            if isinstance(raw_part_ids, list):
                part_ids = [str(item) for item in raw_part_ids if not valid or str(item) in valid]
            if not part_ids and part_id and (not valid or part_id in valid):
                part_ids = [part_id]
            labels: List[int] = []
            raw_labels = raw_issue.get("labels")
            if isinstance(raw_labels, list):
                labels = [_safe_int(label, 0) for label in raw_labels if _safe_int(label, 0) > 0]
            if not labels and _safe_int(result.get("label"), 0) > 0:
                labels = [_safe_int(result.get("label"), 0)]
            if not part_ids and labels:
                part_ids = [label_to_part_id[label] for label in labels if label in label_to_part_id and (not valid or label_to_part_id[label] in valid)]
            evidence_items = [
                f"focused_scope:{scope}",
                f"case:{case_id}",
            ]
            issue_evidence = raw_issue.get("evidence")
            if isinstance(issue_evidence, list):
                evidence_items.extend(str(item) for item in issue_evidence)
            elif issue_evidence:
                evidence_items.append(str(issue_evidence))
            severity = raw_severity
            repair_hint = (
                raw_issue.get("repair_hint")
                or raw_issue.get("repair_instruction")
                or answer.get("repair_hint")
                or answer.get("summary")
                or result.get("summary")
                or "Repair this rejected focused stage and re-run the same focused critic."
            )
            issues.append({
                "issue_id": f"fv{counter:03d}",
                "part_ids": part_ids,
                "labels": labels,
                "issue_type": issue_type,
                "severity": severity,
                "confidence": round(max(0.05, min(0.98, 1.0 - score if score < score_threshold else 0.55)), 3),
                "evidence": evidence_items,
                "diagnosis": str(answer.get("summary") or result.get("summary") or raw_issue.get("evidence") or issue_type),
                "recommended_action": _focused_recommended_action(scope, issue_type, repair_route_override),
                "repair_route": str(repair_route_override or ""),
                "repair_instruction": str(repair_hint),
            })
            counter += 1

    scene_score = sum(scores) / len(scores) if scores else 1.0
    if issues:
        scene_score = min(scene_score, max(0.0, 1.0 - min(1.0, len(issues) / max(1.0, len(focused.get("results", []) or [])))))
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "scene_score": round(max(0.0, min(1.0, scene_score)), 3),
        "global_summary": (
            "Focused single-scope visual critics converted to S2 repair findings. "
            "Each VLM call judged only one stage/scope instead of an omnibus image set."
        ),
        "global_style_diagnosis": "Scene style is judged only by scene_assembly focused calls; part_image/part_mesh calls remain local.",
        "issues": issues,
        "provider_metadata": {
            "provider": "focused_visual_critics",
            "model": focused.get("model"),
            "base_url": focused.get("base_url"),
            "case_count": len(focused.get("results", []) or []),
            "score_threshold": score_threshold,
            "focused_schema_version": focused.get("schema_version"),
        },
    }
    return validate_quality_findings(payload, valid or [str(issue_part) for issue in issues for issue_part in issue.get("part_ids", [])])




def _round_number(value: Any, digits: int = 4) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _bbox_size_only(part: Mapping[str, Any]) -> List[Any]:
    bbox = part.get("bbox") if isinstance(part.get("bbox"), Mapping) else {}
    size = bbox.get("size") or bbox.get("extent") or []
    if not isinstance(size, (list, tuple)):
        return []
    return [_round_number(item) for item in list(size)[:3]]




def build_part_geometry_contract(part: Mapping[str, Any], *, label: Optional[int] = None) -> Dict[str, Any]:
    """Create a compact VLM-facing geometry contract from the layout bbox.

    This is intentionally not a full metric dump.  It translates the S1 layout
    bbox into a small set of geometric obligations for the T2I image and I23D
    mesh: what silhouette/aspect the image must make easy for I23D, and which
    common misleading priors should be rejected before mesh generation.
    """
    size = _bbox_size_only(part)
    dims: List[float] = []
    for value in size[:3]:
        try:
            dims.append(abs(float(value)))
        except (TypeError, ValueError):
            dims.append(0.0)
    while len(dims) < 3:
        dims.append(0.0)
    x, y, z = dims[:3]
    eps = 1e-6
    horizontal_max = max(x, y, eps)
    horizontal_min = max(min(v for v in (x, y) if v > eps) if (x > eps or y > eps) else 0.0, eps)
    max_dim = max(x, y, z, eps)
    min_dim = max(min(v for v in (x, y, z) if v > eps) if (x > eps or y > eps or z > eps) else 0.0, eps)
    description = _part_display_label(part).lower()
    # Use only positive layout semantics for geometry-family classification.
    # Do not read part_prompt/target_prompt: those contain long negative prompts
    # such as "no base/no plinth/no floor", which can falsely classify a tall
    # gallery wall as a low slab.
    positive_fields = [
        part.get("part_description"),
        part.get("open_vocab_label"),
        part.get("source_actor_label"),
        part.get("generation_prompt"),
        part.get("semantic_role"),
    ]
    relations = part.get("spatial_relations")
    if isinstance(relations, list):
        positive_fields.extend(relations[:6])
    text = " ".join(str(value or "") for value in positive_fields).lower()
    combined_text = f"{description} {text}"
    wallish_text = any(token in combined_text for token in ["wall", "gallery", "bar", "wing", "facade", "enclosing", "enclosure"])
    slab_strong_text = any(token in combined_text for token in ["paving", "ground", "floor", "slab", "plinth", "low paving", "courtyard void"])
    base_text = "base" in combined_text and not wallish_text
    slab_text = slab_strong_text or base_text
    column_text = any(token in combined_text for token in ["column", "pillar", "post", "tower", "fin"])

    thin_ratio = min_dim / max_dim
    depth_ratio = min(x, y) / max(horizontal_max, eps)
    height_to_width = z / max(x, eps)
    height_to_depth = z / max(y, eps)
    width_to_depth = x / max(y, eps)

    if (wallish_text and z >= 1.4 * horizontal_min and depth_ratio <= 0.45 and not slab_strong_text) or (z >= 0.55 * max_dim and depth_ratio <= 0.35 and horizontal_max >= 1.5 * horizontal_min and not slab_text):
        shape_family = "tall_upright_thin_wall_or_gallery_bar"
        must_be = [
            "read as a tall upright architectural wall/gallery-bar matching bbox height",
            "show thin depth and vertical faces; it should not collapse into a low curb",
            "make front/side silhouette suitable for I23D to recover a high wall-like mesh",
        ]
        must_not = [
            "low wall, curb, plinth, bench, paving strip, or squat base",
            "whole building/composite scene instead of one isolated part",
            "thick block that ignores the thin bbox depth",
        ]
    elif z <= 0.28 * horizontal_max or slab_text:
        shape_family = "low_horizontal_slab_or_ground_plane"
        must_be = [
            "read as a low, horizontal element with height much smaller than plan footprint",
            "make the top/plan footprint clear enough for I23D, not a vertical tower",
        ]
        must_not = [
            "tall freestanding wall/tower",
            "whole building mass with facade context",
            "decorative background/base unrelated to this part",
        ]
    elif column_text or (z >= 1.6 * horizontal_max and max(x, y) <= 0.45 * z):
        shape_family = "vertical_column_fin_or_tower"
        must_be = [
            "read as a vertical upright part with height dominant over footprint",
            "keep a clean isolated silhouette with visible gravity direction",
        ]
        must_not = [
            "flat ground slab", "wide low wall", "whole building image"]
    elif depth_ratio <= 0.35 or wallish_text:
        shape_family = "thin_linear_architectural_bar_or_wall"
        must_be = [
            "read as one elongated thin architectural component",
            "preserve the thin bbox depth and clear long-axis silhouette",
        ]
        must_not = ["deep bulky mass", "whole building", "decorative base/pedestal"]
    else:
        shape_family = "block_like_mass"
        must_be = [
            "read as one isolated architectural mass fitting the bbox proportions",
            "keep scale/aspect compatible with width/depth/height from layout",
        ]
        must_not = ["whole building/composite scene", "unrelated object", "background pedestal/base artifact"]

    return {
        "layout_label": label,
        "bbox_size_xyz": size,
        "axis_semantics": "x=layout width, y=layout depth/thickness, z=height/up",
        "aspect_ratios": {
            "width_to_depth_x_over_y": _round_number(width_to_depth, 3),
            "height_to_width_z_over_x": _round_number(height_to_width, 3),
            "height_to_depth_z_over_y": _round_number(height_to_depth, 3),
            "thin_axis_over_max": _round_number(thin_ratio, 3),
        },
        "shape_family": shape_family,
        "must_be": must_be,
        "must_not_be": must_not,
        "i23d_prior_acceptance_rule": (
            "Accept the T2I image only if it is a clean isolated geometry prior whose visible silhouette can plausibly generate "
            "a mesh matching this bbox family; semantic match alone is insufficient."
        ),
    }

def _part_core_prompt_context(part: Mapping[str, Any], *, label: Optional[int] = None) -> Dict[str, Any]:
    """Small VLM-facing part context: only facts needed for focused critique."""
    bbox_size = _bbox_size_only(part)
    description = _part_display_label(part)
    prompt_text = (
        part.get("generation_prompt")
        or part.get("target_prompt")
        or part.get("part_prompt")
        or description
        or ""
    )
    context: Dict[str, Any] = {
        "layout_label": label,
        "part_id": str(part.get("part_id") or ""),
        "part_description": description,
        "bbox_size_xyz": bbox_size,
        "bbox_aspect_hint": "x-width, y-depth/thickness, z-height/up from layout bbox; center is intentionally omitted",
        "text_description": str(prompt_text),
        "geometry_contract": build_part_geometry_contract(part, label=label),
    }
    semantic_role = part.get("semantic_role")
    if semantic_role:
        context["semantic_role"] = semantic_role
    relations = part.get("spatial_relations")
    if isinstance(relations, list) and relations:
        context["layout_relations"] = [str(item) for item in relations[:4]]
    return context

def _part_by_id(contract: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(part.get("part_id")): part for part in contract.get("parts", []) or [] if part.get("part_id")}


def _manifest_part_by_id(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(part.get("part_id")): part for part in manifest.get("parts", []) or [] if part.get("part_id")}


def _part_display_label(part: Mapping[str, Any]) -> str:
    return str(
        part.get("part_description")
        or part.get("open_vocab_label")
        or part.get("source_actor_label")
        or part.get("target_class")
        or part.get("part_id")
        or "part"
    )


def run_focused_visual_critics(
    run_dir: str | Path,
    *,
    provider: str = "gpt55",
    model: str = DEFAULT_MULTIMODAL_MODEL,
    base_url: Optional[str] = None,
    auth_path: Optional[str | Path] = None,
    output_dir: str | Path | None = None,
    max_parts: int = 0,
    part_ids: Optional[Sequence[str]] = None,
    scopes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run small, single-question visual critics instead of one omnibus VLM call."""
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir) if output_dir else run_dir / "agent_loop" / "focused_visual_critics"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(run_dir / "manifest.json")
    contract = resolve_contract_from_run(run_dir, manifest)
    metrics = collect_quality_metrics(run_dir, max_sample_vertices=DEFAULT_MAX_SAMPLE_VERTICES, deep_mesh_components=False)
    part_map = _part_by_id(contract)
    metric_by_id = {str(part.get("part_id")): part for part in metrics.get("parts", []) or []}
    all_part_ids = [str(part.get("part_id")) for part in contract.get("parts", []) or [] if part.get("part_id")]
    max_part_count = int(max_parts or 0)
    requested_part_ids = [str(part_id) for part_id in (part_ids or all_part_ids) if str(part_id) in part_map]
    selected_part_ids = requested_part_ids if max_part_count <= 0 else requested_part_ids[:max_part_count]
    scopes_set = set(scopes or ["scene_assembly", "part_image", "part_mesh"])
    report_assets = run_dir / "report" / "assets"
    render_dir = output_dir / "rendered_inputs"
    render_dir.mkdir(parents=True, exist_ok=True)
    focused_views = {
        "iso": (24, 42),
        "front": (8, 0),
        "back": (8, 180),
        "left": (8, -90),
        "right": (8, 90),
        "top": (90, -90),
    }
    max_faces_per_mesh = int(
        os.environ.get("S2_FOCUSED_CRITIC_MAX_FACES_PER_MESH", str(DEFAULT_EVIDENCE_MAX_FACES_PER_MESH))
    )
    try:
        from .visualization import (
            S2_VLM_SIX_VIEWS,
            compose_image_grid,
            render_contract_part_assembly_views,
            render_obj_mesh_views,
            render_s1_style_layout_multiview,
        )
    except Exception:
        S2_VLM_SIX_VIEWS = {}
        compose_image_grid = None
        render_contract_part_assembly_views = None
        render_obj_mesh_views = None
        render_s1_style_layout_multiview = None
    results: List[Dict[str, Any]] = []

    def write_case(case_id: str, prompt: Mapping[str, Any], images: Sequence[Tuple[str, str | Path]]) -> Dict[str, Any]:
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        copied_images = []
        call_images: List[Tuple[str, Path]] = []
        for index, (label, source) in enumerate(images, start=1):
            source_path = Path(str(source))
            copied = _copy_named_image(source_path, case_dir / "input_images" / f"{index:02d}_{_safe_artifact_name(label)}{source_path.suffix or '.png'}")
            if copied:
                copied_images.append({"index": index, "label": label, "source_path": str(source_path), "copied_path": copied})
                call_images.append((label, Path(copied)))
        request = {
            "schema_version": "archstudio_s2.focused_visual_critic_request.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case_id": case_id,
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "input_images": copied_images,
        }
        write_json(case_dir / "request.json", request)
        provider_norm = str(provider or "").lower().replace("-", "_")
        if provider_norm in {"mock", "metrics", "metrics_only"}:
            gaps = prompt.get("mesh_contact_gaps") if isinstance(prompt.get("mesh_contact_gaps"), list) else []
            label_by_part = {part_id: idx + 1 for idx, part_id in enumerate(all_part_ids)}
            if prompt.get("critic_scope") == "scene_assembly" and gaps:
                first_gap = gaps[0] if isinstance(gaps[0], Mapping) else {}
                part_ids = [str(first_gap.get("part_a")), str(first_gap.get("part_b"))]
                labels = [label_by_part[part_id] for part_id in part_ids if part_id in label_by_part]
                answer = {
                    "score": 0.42,
                    "verdict": "revise",
                    "summary": "Mock focused assemble critic found a measurable contact gap in mesh metrics and preserved the true six-view mesh inputs for review.",
                    "issues": [{
                        "issue_type": "gap_or_seam",
                        "part_ids": part_ids,
                        "labels": labels,
                        "severity": 4,
                        "evidence": "mesh_contact_gaps plus scene assembly six-view board",
                        "repair_route": "deterministic_snap",
                        "repair_hint": "snap or resize the named parts toward their expected bbox contact faces, then rerun the assemble critic",
                    }],
                }
            else:
                answer = {
                    "score": 0.92,
                    "verdict": "accept",
                    "summary": "Mock focused critic accepted this scope; inspect recorded input images for visual audit.",
                    "issues": [],
                }
            provider_metadata = {"provider": "mock_focused", "model": model, "base_url": None}
            status = "completed"
            error = None
        else:
            try:
                answer, provider_metadata = _call_openai_json_with_images(
                    prompt=prompt,
                    image_paths=call_images,
                    model=model,
                    base_url=base_url,
                    auth_path=auth_path,
                )
                status = "completed"
                error = None
            except Exception as exc:  # noqa: BLE001 - focused run should preserve each failure.
                answer = {}
                provider_metadata = {"provider": provider, "model": model, "base_url": str(base_url or DEFAULT_AICODEMIRROR_BASE_URL).split("?", 1)[0]}
                status = "error"
                error = repr(exc)
        response = {
            "schema_version": "archstudio_s2.focused_visual_critic_response.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case_id": case_id,
            "status": status,
            "error": error,
            "answer": answer,
            "provider_metadata": provider_metadata,
        }
        write_json(case_dir / "answer.json", response)
        result = {
            "case_id": case_id,
            "scope": prompt.get("critic_scope"),
            "part_id": (
                prompt.get("part_id")
                or ((prompt.get("part_context") or {}).get("part_id") if isinstance(prompt.get("part_context"), Mapping) else None)
            ),
            "label": (prompt.get("part_context") or {}).get("layout_label") if isinstance(prompt.get("part_context"), Mapping) else prompt.get("label"),
            "status": status,
            "score": answer.get("score"),
            "verdict": answer.get("verdict"),
            "summary": answer.get("summary"),
            "error": error,
            "request_path": str(case_dir / "request.json"),
            "answer_path": str(case_dir / "answer.json"),
            "input_images": copied_images,
        }
        results.append(result)
        return result

    if "scene_assembly" in scopes_set:
        scene_images = []
        rendered_scene_views = {}
        if render_contract_part_assembly_views is not None:
            try:
                rendered_scene_views = render_contract_part_assembly_views(
                    contract,
                    run_dir,
                    render_dir,
                    "scene_assembly_true_mesh",
                    title="Focused critic true colored assembled mesh",
                    max_faces_per_mesh=max_faces_per_mesh,
                    include_placeholders=True,
                    placeholder_alpha=0.18,
                    views=S2_VLM_SIX_VIEWS or focused_views,
                )
            except Exception:
                rendered_scene_views = {}
        for name in ("iso", "front", "back", "left", "right", "top"):
            path = Path(rendered_scene_views.get(name, "")) if rendered_scene_views.get(name) else None
            fallback_name = "side" if name in {"left", "right"} else name
            if not path or not path.exists():
                path = report_assets / f"assembly_complete_view_{fallback_name}.png"
            if not path.exists():
                path = report_assets / f"assembly_raw_view_{fallback_name}.png"
            if path.exists():
                scene_images.append((f"true_colored_assembly_mesh_{name}", path))
        if compose_image_grid is not None and scene_images:
            scene_board = compose_image_grid(
                scene_images,
                render_dir / "scene_assembly_mesh_six_view_board.png",
                "ArchStudio-S2 scene assembly mesh · six-view focused critic input",
                columns=3,
            )
            if scene_board:
                scene_images = [("true_colored_assembly_mesh_six_view_board", scene_board)]
        write_case(
            "scene_assembly",
            {
                "task": "Focused S2 scene-assembly critic",
                "critic_scope": "scene_assembly",
                "question": "Only judge the overall assembled generated mesh from these multi-view true mesh surface renders. Is the assembled geometry physically coherent with no visible gaps, seams, floating parts, penetrations, impossible overlaps, wrong scale/orientation, or missing parts? Every rejection must name concrete numbered labels/part_ids when visible.",
                "do_not_judge": ["individual T2I reference quality", "texture", "layout aesthetics", "style preference unless it causes geometry incoherence"],
                "required_json": {"score": "0-1", "verdict": "accept|revise", "summary": "short", "issues": [{"issue_type": "gap_or_seam|overlap|floating|penetration|scale_mismatch|wrong_orientation|missing_part|low_geometry_quality|wrong_semantics", "part_ids": ["optional exact ids from metrics/context"], "labels": [1], "severity": 1, "evidence": "which view and what numbered labels show it", "repair_route": "deterministic_snap|rerun_i3d|rerun_t2i_i3d|manual_review", "repair_hint": "specific part-level fix"}]},
                "metrics_summary": metrics.get("summary", {}),
                "mesh_contact_gaps": [gap for gap in metrics.get("mesh_contact_gaps", []) if gap.get("above_tolerance")][:20],
                "part_index": [{"label": int(part.get("index", 0)) + 1, "part_id": part.get("part_id"), "display_label": part.get("display_label"), "semantic_role": part.get("semantic_role")} for part in metrics.get("parts", [])],
            },
            scene_images,
        )

    part_scopes_requested = bool(scopes_set.intersection({"part_image", "part_mesh"}))
    if part_scopes_requested:
        for part_id in selected_part_ids:
            part = part_map[part_id]
            label = all_part_ids.index(part_id) + 1 if part_id in all_part_ids else None
            live_reference = run_dir / "parts" / part_id / "reference.png"
            reference = live_reference if live_reference.exists() else report_assets / "parts" / f"{part_id}_reference.png"
            mesh_path = resolve_part_mesh_path(run_dir, part_id)
            part_render_dir = render_dir / _safe_artifact_name(part_id)
            mesh_views = {}
            layout_board = None
            if mesh_path and render_obj_mesh_views is not None:
                try:
                    mesh_views = render_obj_mesh_views(
                        mesh_path,
                        part_render_dir,
                        "part_true_mesh",
                        title=f"Focused critic true mesh #{label}",
                        max_faces=max_faces_per_mesh,
                        views=focused_views,
                    )
                except Exception:
                    mesh_views = {}
            if render_s1_style_layout_multiview is not None:
                try:
                    layout_board = render_s1_style_layout_multiview(
                        contract,
                        part_render_dir / "layout_s1_style_visual_critic_multiview.png",
                        title=f"ArchStudio-S2 layout context #{label} · S1-style numeric labels · highlighted current part",
                        highlight_part_id=part_id,
                    )
                except Exception:
                    layout_board = None
            mesh_render = Path(mesh_views.get("iso", "")) if mesh_views.get("iso") else report_assets / "parts" / f"{part_id}_mesh.png"
            layout_render = report_assets / "parts" / f"{part_id}_layout.png"
            prompt_common = {
                "part_id": part_id,
                "label": label,
                "part_context": _part_core_prompt_context(part, label=label),
            }
            geometry_contract = prompt_common["part_context"].get("geometry_contract", {})
            if "part_image" in scopes_set and reference.exists():
                write_case(
                    f"part_image_{label:03d}_{_safe_artifact_name(part_id)}",
                    {
                        "task": "Focused S2 part T2I-reference critic",
                        "critic_scope": "part_image",
                        "question": "Judge this T2I reference for layout label #{label} as a geometry-grounded I23D prior, not just as a pretty semantic image. First locate the numbered part in the highlighted S1-style layout board; then compare the image against part_context.text_description and part_context.geometry_contract. Reject if the image looks semantically related but violates the bbox-derived shape family, e.g. a low wall/plinth for a tall upright thin wall/gallery bar, a bulky block for a thin-depth part, or any whole-building/composite/background/base artifact. Accept only if semantic_verdict, geometry_contract_verdict, and i23d_prior_verdict all pass.".format(label=label),
                        "do_not_judge": ["generated 3D mesh quality", "assembly seams", "texture transfer", "style preference unless it hides geometry"],
                        "geometry_contract": geometry_contract,
                        "required_json": {"score": "0-1", "semantic_verdict": "accept|revise", "geometry_contract_verdict": "accept|revise", "i23d_prior_verdict": "accept|revise", "final_verdict": "accept|revise", "verdict": "same as final_verdict", "summary": "short VLM audit opinion including accept/revise reason", "issues": [{"issue_type": "wrong_semantics|whole_building|background_or_base_artifact|bad_aspect|scale_mismatch|geometry_contract_mismatch|i23d_prior_mismatch", "severity": 1, "evidence": "what you see and which text/bbox/layout/geometry_contract constraint it violates", "repair_route": "rerun_t2i|rerun_t2i_i3d|manual_review", "repair_hint": "specific prompt rewrite hint for a clean isolated part image that obeys geometry_contract.must_be and avoids must_not_be"}]},
                        **prompt_common,
                    },
                    [("part_reference_image", reference)]
                    + ([("layout_s1_style_multiview_context", layout_board)] if layout_board and Path(layout_board).exists() else [])
                    + ([("part_layout_bbox_context_xy_fallback", layout_render)] if (not layout_board or not Path(layout_board).exists()) and layout_render.exists() else []),
                )
            if "part_mesh" in scopes_set and mesh_render.exists():
                mesh_images = []
                for name in ("iso", "front", "back", "left", "right", "top"):
                    path = Path(mesh_views.get(name, "")) if mesh_views.get(name) else None
                    if path and path.exists():
                        mesh_images.append((f"generated_true_part_mesh_{name}", path))
                if not mesh_images:
                    mesh_images = [("generated_true_part_mesh_iso", mesh_render)]
                images = list(mesh_images)
                if compose_image_grid is not None and len(mesh_images) > 1:
                    mesh_board = compose_image_grid(
                        mesh_images,
                        part_render_dir / "part_mesh_six_view_board.png",
                        f"ArchStudio-S2 part mesh #{label} · six-view VLM input",
                        columns=3,
                    )
                    if mesh_board:
                        images = [("generated_true_part_mesh_six_view_board", mesh_board)]
                if reference.exists():
                    images.append(("source_T2I_reference_image", reference))
                if layout_board and Path(layout_board).exists():
                    images.append(("layout_s1_style_multiview_context", layout_board))
                elif layout_render.exists():
                    images.append(("part_layout_bbox_context_xy_fallback", layout_render))
                write_case(
                    f"part_mesh_{label:03d}_{_safe_artifact_name(part_id)}",
                    {
                        "task": "Focused S2 single-part mesh critic",
                        "critic_scope": "part_mesh",
                        "question": "Only judge this generated mesh for layout label #{label}. The primary image is a six-view mesh board (iso/front/back/left/right/top) when available; use the source T2I image, S1-style numeric layout board, and part_context.geometry_contract as constraints. Decide whether failure is caused by I23D generation or by a bad source image that already violates the layout geometry contract. Does the mesh match text_description, bbox_size_xyz, shape_family, thin/depth/height ratios, and gravity/orientation?".format(label=label),
                        "do_not_judge": ["overall scene assembly", "other unrelated parts", "texture quality unless it blocks geometry reading"],
                        "geometry_contract": geometry_contract,
                        "required_json": {"score": "0-1", "verdict": "accept|revise", "root_cause": "source_image|i23d_generation|layout_contract|unclear", "summary": "short", "issues": [{"issue_type": "wrong_semantics|low_geometry_quality|scale_mismatch|wrong_orientation|artifact_base_or_panel|geometry_contract_mismatch", "severity": 1, "root_cause": "source_image|i23d_generation|layout_contract|unclear", "evidence": "image label", "repair_route": "rerun_i3d|rerun_t2i_i3d|deterministic_snap|manual_review", "repair_hint": "specific I23D prompt/geometry rerun hint, or T2I prompt fallback if the source image is the root cause"}]},
                        **prompt_common,
                    },
                    images,
                )

    summary = {
        "schema_version": "archstudio_s2.focused_visual_critics.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "provider": provider,
        "model": model,
        "base_url": str(base_url or DEFAULT_AICODEMIRROR_BASE_URL).split("?", 1)[0],
        "selected_part_ids": selected_part_ids,
        "part_index": [{"label": index + 1, "part_id": part_id, "display_label": _part_display_label(part_map[part_id])} for index, part_id in enumerate(all_part_ids) if part_id in part_map],
        "scopes": sorted(scopes_set),
        "results": results,
    }
    write_json(output_dir / "focused_visual_critics_summary.json", summary)
    return summary

def build_visual_evaluator_prompt(metrics: Mapping[str, Any], evidence: Mapping[str, Any] | None) -> str:
    payload = {
        "task": "ArchStudio-S2 multimodal quality evaluation",
        "instructions": [
            "You are evaluating a layout-guided 3D architectural asset generated part-by-part.",
            "Primary evidence is the generated mesh: assembly_geometry_views and part_mesh_eval_board images. Layout images are only constraints, not the generated result.",
            "For each part_mesh_eval_board row, compare the T2I reference image and generated part mesh render against the row's text prompt and bbox; reject if the mesh is a box, wrong part, whole building, badly oriented, or detached.",
            "Use the numbered labels and part_id values from the evidence and metrics; do not invent parts.",
            "Judge visual plausibility, part-to-text semantic match, style coherence, geometry seams/gaps, scale mismatch, wrong orientation, bad texture, and artifact bases/panels.",
            "Respect open vocabulary: do not rely on closed wall/window/roof classes; use part_description, semantic_role, spatial_relations, and bbox facts.",
            "Use textured_assembly_views when judging colors/materials and whether dark glass/shadow openings read correctly; use assembly_geometry_views for shape/scale/contact.",
            "If deterministic metrics show a mesh gap or missing mesh, include it unless the visual evidence clearly proves it is intentional.",
            "Important evidence caveat: assembly PNGs are sampled surface previews of dense meshes, not screenshots from the interactive GLB. If they look sparse/dotted, cross-check render_notes, vertex/face counts, part-level views, and metrics before calling the whole asset a point cloud.",
            "Return ONLY a JSON object matching the schema. Keep issues actionable and part-level whenever possible.",
        ],
        "valid_issue_types": sorted(ISSUE_TYPES),
        "valid_recommended_actions": sorted(RECOMMENDED_ACTIONS),
        "required_schema": {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "scene_score": "float in [0,1]",
            "global_summary": "short overall diagnosis",
            "global_style_diagnosis": "short style/coherence note",
            "issues": [
                {
                    "issue_id": "v001",
                    "part_ids": ["must be existing part_id strings"],
                    "labels": ["matching numeric labels"],
                    "issue_type": "one valid_issue_types value",
                    "severity": "integer 1-5",
                    "confidence": "float in [0,1]",
                    "evidence": ["image labels or metrics_summary"],
                    "diagnosis": "what is wrong and why",
                    "recommended_action": "one valid_recommended_actions value",
                    "repair_instruction": "concrete next action for this exact part or pair",
                }
            ],
        },
        "metrics": _compact_metrics_for_evaluator(metrics),
        "evidence_render_notes": _evidence_render_notes_for_prompt(evidence),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def evaluate_with_openai_multimodal(
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    auth_path: Optional[str | Path] = None,
    max_images: int = 12,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible multimodal evaluator and validate findings."""
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("openai and httpx packages are required for multimodal evaluation") from exc

    model = model or os.environ.get("S2_QUALITY_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MULTIMODAL_MODEL
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_AICODEMIRROR_BASE_URL).rstrip("/")
    api_key = api_key or load_openai_api_key(auth_path)
    trust_env = os.environ.get("S2_OPENAI_TRUST_ENV", "0").strip().lower() in {"1", "true", "yes", "on"}
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=trust_env, timeout=180.0),
    )

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": build_visual_evaluator_prompt(metrics, evidence)}
    ]
    for label, image_path in _resolve_evidence_image_paths(evidence, max_images=max_images):
        content.append({"type": "text", "text": f"Evidence image: {label}"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(image_path),
                "detail": "high",
            },
        })

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict multimodal QA evaluator for ArchStudio-S2. Return only JSON."},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as first_exc:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("response_format", None)
        try:
            completion = client.chat.completions.create(**fallback_kwargs)
        except Exception:
            fallback_kwargs.pop("max_tokens", None)
            fallback_kwargs["max_completion_tokens"] = max_tokens
            try:
                completion = client.chat.completions.create(**fallback_kwargs)
            except Exception as final_exc:  # pragma: no cover - provider/network dependent
                raise RuntimeError(f"OpenAI-compatible multimodal quality request failed: {final_exc}") from first_exc

    text = completion.choices[0].message.content or ""
    payload = parse_json_object(text)
    part_ids = [str(part.get("part_id")) for part in metrics.get("parts", []) or []]
    result = validate_quality_findings(payload, part_ids)
    result["provider_metadata"] = {
        "provider": "openai_multimodal",
        "model": model,
        "base_url": base_url.split("?", 1)[0],
        "max_images": max_images,
        "image_count": len(_resolve_evidence_image_paths(evidence, max_images=max_images)),
    }
    return result


def evaluate_quality(
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    *,
    provider: str,
    mock_response: str | Path | Mapping[str, Any] | None = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    auth_path: Optional[str | Path] = None,
    max_images: int = 12,
) -> Optional[Dict[str, Any]]:
    normalized = provider.lower().replace("-", "_")
    if normalized in {"metrics", "metrics_only"}:
        return None
    if normalized == "mock":
        return evaluate_with_mock(metrics, mock_response)
    if normalized in {"openai", "aimirror", "aicodemirror", "gpt", "gpt55", "multimodal"}:
        return evaluate_with_openai_multimodal(
            metrics,
            evidence,
            model=model,
            base_url=base_url,
            auth_path=auth_path,
            max_images=max_images,
        )
    raise ValueError(f"unknown quality evaluator provider: {provider}")


def action_for_issue(issue: Mapping[str, Any]) -> str:
    recommended = str(issue.get("recommended_action") or "")
    if recommended in RECOMMENDED_ACTIONS and recommended != "accept":
        return recommended
    issue_type = str(issue.get("issue_type") or "")
    if issue_type == "gap_or_seam":
        return "deterministic_resize_snap"
    if issue_type in {"texture_bad", "texture_direction_bad", "low_texture_quality", "style_inconsistent"}:
        return "rerun_texture"
    if issue_type in {"wrong_semantics", "artifact_base_or_panel"}:
        return "rerun_t2i_geometry_texture"
    if issue_type in {"scale_mismatch", "low_geometry_quality", "wrong_orientation", "missing_part"}:
        return "rerun_geometry"
    return "manual_review"


def build_repair_plan(
    findings: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    max_parts: int = 5,
    max_attempts_per_part: int = 2,
    dry_run: bool = True,
) -> Dict[str, Any]:
    issues = sorted(
        list(findings.get("issues", []) or []),
        key=lambda item: (-int(item.get("severity", 0)), -float(item.get("confidence", 0.0)), str(item.get("issue_id", ""))),
    )
    planned_parts: set[str] = set()
    actions: List[Dict[str, Any]] = []
    for issue in issues:
        part_ids = [str(part_id) for part_id in issue.get("part_ids", [])]
        if not part_ids:
            continue
        if len(planned_parts.union(part_ids)) > max_parts:
            continue
        action = action_for_issue(issue)
        if action == "accept":
            continue
        planned_parts.update(part_ids)
        actions.append({
            "issue_id": issue.get("issue_id"),
            "part_ids": part_ids,
            "labels": list(issue.get("labels", []) or []),
            "issue_type": issue.get("issue_type"),
            "severity": issue.get("severity"),
            "confidence": issue.get("confidence"),
            "planned_action": action,
            "dry_run": dry_run,
            "max_attempts_per_part": max_attempts_per_part,
            "repair_instruction": issue.get("repair_instruction", ""),
            "source_diagnosis": issue.get("diagnosis", ""),
            "attempt_output_policy": "write additive attempt assets under quality/v9_agent/repairs; never overwrite source run",
        })
    return {
        "schema_version": REPAIR_PLAN_VERSION,
        "run_dir": metrics.get("run_dir"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "max_parts": max_parts,
        "max_attempts_per_part": max_attempts_per_part,
        "num_input_issues": len(issues),
        "num_planned_actions": len(actions),
        "planned_part_ids": sorted(planned_parts),
        "actions": actions,
        "stop_condition": "dry_run_plan_only" if dry_run else "execute_selected_local_repairs_then_reevaluate",
    }


def make_quality_summary(
    run_dir: str | Path,
    quality_dir: str | Path,
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
    findings: Mapping[str, Any] | None = None,
    repair_plan: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    quality_dir = Path(quality_dir).resolve()
    def rel(path: str | Path | None) -> Optional[str]:
        if not path:
            return None
        try:
            return os.path.relpath(str(path), str(run_dir / "report"))
        except Exception:
            return str(path)

    evidence_assets = dict((evidence or {}).get("assets", {}) if isinstance(evidence, Mapping) else {})
    summary = {
        "schema_version": "archstudio_s2_quality_summary.v1",
        "run_dir": str(run_dir),
        "quality_dir": str(quality_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics_path": rel(quality_dir / "metrics_summary.json"),
        "findings_path": rel(quality_dir / "quality_findings.json") if findings else None,
        "repair_plan_path": rel(quality_dir / "repair_plan.json") if repair_plan else None,
        "trace_path": rel(quality_dir / "quality_agent_trace.json"),
        "scene_score": findings.get("scene_score") if findings else None,
        "global_summary": findings.get("global_summary") if findings else "metrics-only; no visual evaluator findings",
        "metric_summary": dict(metrics.get("summary", {})),
        "issue_count": len(findings.get("issues", []) or []) if findings else 0,
        "planned_action_count": len(repair_plan.get("actions", []) or []) if repair_plan else 0,
        "evidence": {
            key: (
                {name: rel(path) for name, path in value.items()}
                if isinstance(value, Mapping)
                else [rel(item) for item in value]
                if isinstance(value, list)
                else rel(value)
            )
            for key, value in evidence_assets.items()
        },
    }
    return summary


def write_quality_trace(
    quality_dir: str | Path,
    *,
    run_dir: str | Path,
    texture_run_dir: str | Path | None,
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    findings: Mapping[str, Any] | None,
    repair_plan: Mapping[str, Any] | None = None,
    provider: str = "metrics-only",
) -> Dict[str, Any]:
    trace = {
        "schema_version": QUALITY_TRACE_VERSION,
        "run_dir": str(Path(run_dir).resolve()),
        "texture_run_dir": str(Path(texture_run_dir).resolve()) if texture_run_dir else None,
        "provider": provider,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "timeline": [
            {
                "phase": "collect_deterministic_metrics",
                "status": "completed",
                "summary": dict(metrics.get("summary", {})),
            },
            {
                "phase": "build_evidence_pack",
                "status": "completed" if evidence else "skipped",
                "evidence_dir": evidence.get("evidence_dir") if evidence else None,
            },
            {
                "phase": "visual_or_mock_evaluator",
                "status": "completed" if findings else "skipped",
                "issue_count": len(findings.get("issues", []) or []) if findings else 0,
            },
            {
                "phase": "repair_plan",
                "status": "completed" if repair_plan else "not_planned",
                "planned_action_count": len(repair_plan.get("actions", []) or []) if repair_plan else 0,
            },
        ],
        "stop_condition": {
            "metrics_collected": True,
            "findings_available": bool(findings),
            "repair_plan_available": bool(repair_plan),
            "gpu_or_model_generation_invoked": False,
        },
    }
    write_json(Path(quality_dir) / "quality_agent_trace.json", trace)
    return trace
