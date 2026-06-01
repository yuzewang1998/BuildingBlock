"""Contract emission utilities for the BuildingBlock layout-to-mesh pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


SCHEMA_VERSION = "layout_mesh_contract.v1"
UNITS = "normalized_scene_units"
ORIENTATION_POLICY = "axis_aligned_no_extra_rotation"
BBOX_OWNERSHIP = "full_part_bbox"
COORDINATE_FRAME = {
    "handedness": "right-handed",
    "axes": {
        "x": "right",
        "y": "forward",
        "z": "up",
    },
    "actor_location": "axis_aligned_bbox_center",
    "actor_size": "full_extent_xyz",
    "bbox_encoding": {
        "center": "scene_xyz",
        "size": "full_extent_xyz",
    },
    "global_rotation": "none",
}
CONTRACT_FILENAME = "layout_mesh_contract.json"
RAW_LAYOUT_FILENAME = "layout.json"
HUNYUAN_DEMO_ROOT = Path(
    "/home/wangyz/project/0working/Hunyuan3D-Omni-main/Hunyuan3D-Omni-main/demos/bbox"
)
DEFAULT_REFERENCE_IMAGE_MAP = {
    "wall": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "window": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "door": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "floor": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "pillar": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "pipe": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "railing": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "stair": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "accessory": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
    "roof": str(HUNYUAN_DEMO_ROOT / "jianying0.png"),
    "balcony": str(HUNYUAN_DEMO_ROOT / "jianying0.png"),
    "awning": str(HUNYUAN_DEMO_ROOT / "jianying0.png"),
    "chimney": str(HUNYUAN_DEMO_ROOT / "jianbihua0.png"),
    "object": str(HUNYUAN_DEMO_ROOT / "furniture0.png"),
}
OPEN_SEMANTIC_LAYOUT_VERSION = "open_semantic_parts.v0"
OPEN_SEMANTIC_TARGET_CLASS = "open_semantic_part"
OPEN_SEMANTIC_MIN_BBOX_EXTENT = 0.02


def is_open_semantic_layout(raw_layout: Any) -> bool:
    version = raw_layout.get("s2_contract_version") if isinstance(raw_layout, dict) else None
    return (
        isinstance(raw_layout, dict)
        and (
            version == OPEN_SEMANTIC_LAYOUT_VERSION
            or str(version or "").startswith("open_semantic_parts.v1")
        )
        and isinstance(raw_layout.get("parts"), list)
    )


def _as_text_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        text = str(values).strip()
        return [text] if text else []
    if isinstance(values, Sequence):
        return [str(item).strip() for item in values if str(item).strip()]
    return [str(values).strip()] if str(values).strip() else []


def _open_semantic_source_label(part: Dict[str, Any], index: int) -> str:
    for key in ("part_description", "generation_prompt", "part_name", "part_id"):
        value = str(part.get(key, "")).strip()
        if value:
            return value
    return "open semantic architectural component {}".format(index + 1)


def _open_semantic_target_prompt(part: Dict[str, Any], scene_prompt: str = "") -> str:
    description = str(part.get("part_description") or part.get("part_name") or "architectural component").strip()
    generation_prompt = str(part.get("generation_prompt") or "").strip()
    semantic_role = str(part.get("semantic_role") or "").strip()
    detail_level = str(part.get("detail_level") or "").strip()
    spatial_relations = _as_text_list(part.get("spatial_relations"))
    material_hint = str(part.get("material_hint") or "").strip()

    bbox = part.get("bbox") or {}
    size = bbox.get("size") or part.get("actor_size") or []
    scale_clause = ""
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) == 3:
        try:
            sx, sy, sz = [float(value) for value in size]
            scale_clause = (
                "layout bbox full extent XYZ: x-width {:.4f}, y-depth {:.4f}, z-height {:.4f}; "
                "respect these proportions as a single component"
            ).format(sx, sy, sz)
        except Exception:
            scale_clause = ""

    clauses = [
        "open-vocabulary architectural component: {}".format(description),
    ]
    if generation_prompt and generation_prompt.lower() != description.lower():
        clauses.append(generation_prompt)
    if semantic_role:
        clauses.append("semantic role: {}".format(semantic_role))
    if detail_level:
        clauses.append("detail level: {}".format(detail_level))
    if spatial_relations:
        clauses.append("layout relations: {}".format("; ".join(spatial_relations)))
    if scale_clause:
        clauses.append(scale_clause)
    if material_hint:
        clauses.append("material hint: {}".format(material_hint))
    if scene_prompt:
        clauses.append("overall building context for style only: {}".format(scene_prompt))
    return " | ".join(clauses)


def _open_semantic_part_id(layout_id: str, index: int, part: Dict[str, Any]) -> str:
    existing = str(part.get("part_id") or "").strip()
    if existing:
        return _sanitize_part_id(existing)
    name = str(part.get("part_name") or part.get("part_description") or "part").strip()
    slug = _sanitize_target_class(name)[:72] or "part"
    return "{}__{:04d}__{}".format(layout_id, index, slug)


def _repair_nonpositive_bbox_size(size: Sequence[float]) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Return a valid bbox size and an audit record when a layout has a degenerate axis.

    Stage-1 open-semantic layouts may intentionally include very thin semantic
    markers, but downstream T2I/geometry code requires strictly positive bbox
    extents.  We do not mutate the source layout file; instead we expand any
    non-positive axis to a small generic thickness and preserve the original
    size plus the repair note in the emitted S2 contract.
    """

    original = [float(value) for value in size]
    if all(value > 0 for value in original):
        return [round(value, 6) for value in original], None

    positive_values = [value for value in original if value > 0]
    if positive_values:
        inferred_extent = min(positive_values) * 0.25
    else:
        inferred_extent = OPEN_SEMANTIC_MIN_BBOX_EXTENT
    replacement = round(max(OPEN_SEMANTIC_MIN_BBOX_EXTENT, inferred_extent), 6)
    repaired = [round(value if value > 0 else replacement, 6) for value in original]
    return repaired, {
        "type": "non_positive_bbox_axis_repaired",
        "original_size": [round(value, 6) for value in original],
        "repaired_size": repaired,
        "replacement_extent": replacement,
        "policy": (
            "generic S2 compatibility repair: non-positive bbox axes are expanded "
            "to max(0.02, 0.25 * smallest positive extent); source layout is unchanged"
        ),
    }


def _sanitize_part_id(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_\-]+", "_", value.strip())
    return sanitized.strip("_") or "part"


def normalize_open_semantic_layout(raw_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not is_open_semantic_layout(raw_layout):
        raise ValueError("Expected an open_semantic_parts.v0/v1 layout object")

    normalized_parts: List[Dict[str, Any]] = []
    scene_prompt = str(raw_layout.get("prompt") or "").strip()
    layout_id = str(raw_layout.get("layout_id") or "").strip()
    for index, part in enumerate(raw_layout.get("parts", [])):
        if not isinstance(part, dict):
            raise ValueError("Open semantic layout part {} must be a JSON object".format(index))
        bbox = part.get("bbox")
        if not isinstance(bbox, dict):
            raise ValueError("Open semantic layout part {} is missing bbox".format(index))
        actor_size_raw = _rounded_xyz(bbox.get("size", part.get("actor_size", [])))
        actor_size, bbox_repair_warning = _repair_nonpositive_bbox_size(actor_size_raw)
        actor_location = _rounded_xyz(bbox.get("center", part.get("actor_location", [])))

        part_description = str(part.get("part_description") or part.get("part_name") or "").strip()
        if not part_description:
            raise ValueError("Open semantic layout part {} is missing part_description".format(index))

        open_prompt = _open_semantic_target_prompt(part, scene_prompt=scene_prompt)
        normalized_parts.append(
            {
                "source_actor_label": _open_semantic_source_label(part, index),
                "source_materials": [],
                "actor_size": actor_size,
                "actor_location": actor_location,
                "source_part_id": str(part.get("part_id") or "").strip(),
                "source_part_name": str(part.get("part_name") or "").strip(),
                "source_bbox_size": actor_size_raw,
                "bbox_repair_warning": bbox_repair_warning,
                "open_vocab_label": part_description,
                "part_description": part_description,
                "generation_prompt": str(part.get("generation_prompt") or "").strip(),
                "semantic_role": str(part.get("semantic_role") or "").strip(),
                "spatial_relations": _as_text_list(part.get("spatial_relations")),
                "detail_level": str(part.get("detail_level") or "").strip(),
                "material_hint": str(part.get("material_hint") or "").strip(),
                "legacy_compatibility_hint": str(part.get("legacy_compatibility_hint") or "").strip(),
                "target_class_override": OPEN_SEMANTIC_TARGET_CLASS,
                "target_prompt_override": open_prompt,
                "part_id_override": _open_semantic_part_id(layout_id or "open_semantic_layout", index, part),
                "open_semantic_layout_id": layout_id,
            }
        )
    if not normalized_parts:
        raise ValueError("Expected at least one open semantic layout part")
    return normalized_parts


def normalize_layout_payload(raw_layout: Any) -> List[Dict[str, Any]]:
    if is_open_semantic_layout(raw_layout):
        return normalize_open_semantic_layout(raw_layout)
    return normalize_raw_layout(raw_layout)


def _rounded_xyz(values: Sequence[Any], precision: int = 6) -> List[float]:
    if len(values) != 3:
        raise ValueError("Expected 3D geometry values")
    return [round(float(value), precision) for value in values]


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(payload: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def _sanitize_target_class(value: str) -> str:
    sanitized = re.sub(r"[^0-9a-z]+", "_", value.lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "object"


def _normalize_material_label(material: str) -> str:
    label = str(material).strip()
    if label.lower().endswith("material"):
        label = label[: -len("material")]
    return label.strip() or "object"


def _derive_target_class(materials: Sequence[str]) -> str:
    if not materials:
        return "object"
    return _sanitize_target_class(_normalize_material_label(materials[0]))


def _derive_target_prompt(target_class: str) -> str:
    return "3d mesh of {}".format(target_class.replace("_", " "))


def _derive_reference_image(target_class: str) -> str:
    return DEFAULT_REFERENCE_IMAGE_MAP.get(target_class, DEFAULT_REFERENCE_IMAGE_MAP["object"])


def _looks_normalized_part(part: Dict[str, Any]) -> bool:
    return {
        "source_actor_label",
        "source_materials",
        "actor_size",
        "actor_location",
    }.issubset(part.keys())


def normalize_raw_layout(raw_layout: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_layout, list) or not raw_layout:
        raise ValueError("Expected a non-empty raw BuildingBlock layout list")

    normalized_parts: List[Dict[str, Any]] = []
    for index, part in enumerate(raw_layout):
        if not isinstance(part, dict):
            raise ValueError("Layout part {} must be a JSON object".format(index))

        if _looks_normalized_part(part):
            source_actor_label = str(part.get("source_actor_label", "")).strip()
            source_materials = [str(item) for item in part.get("source_materials", [])]
        else:
            source_actor_label = str(part.get("actor_label", "")).strip()
            source_materials = [str(item) for item in part.get("materials", [])]
        actor_size = _rounded_xyz(part.get("actor_size", []))
        actor_location = _rounded_xyz(part.get("actor_location", []))

        if not source_actor_label:
            raise ValueError("Layout part {} is missing actor_label".format(index))

        normalized_parts.append(
            {
                "source_actor_label": source_actor_label,
                "source_materials": source_materials,
                "actor_size": actor_size,
                "actor_location": actor_location,
            }
        )

    return normalized_parts


def load_raw_layout(layout_path: Path) -> List[Dict[str, Any]]:
    return normalize_layout_payload(json.loads(layout_path.read_text()))


def derive_layout_id(raw_layout_payload: Any) -> str:
    if is_open_semantic_layout(raw_layout_payload):
        layout_id = str(raw_layout_payload.get("layout_id") or "").strip()
        if layout_id:
            return _sanitize_target_class(layout_id)
        normalized_parts = normalize_open_semantic_layout(raw_layout_payload)
    else:
        normalized_parts = normalize_raw_layout(raw_layout_payload)
    canonical_parts = [_part_canonical_tuple(part) for part in normalized_parts]
    canonical_parts.sort()
    return "layout_{}".format(_content_hash(canonical_parts))


def _part_canonical_tuple(part: Dict[str, Any]) -> Tuple[Any, ...]:
    target_class = part.get("target_class_override") or _derive_target_class(part["source_materials"])
    target_prompt = part.get("target_prompt_override", "")
    part_id_override = part.get("part_id_override", "")
    return (
        part_id_override,
        part["source_actor_label"],
        tuple(sorted(part["source_materials"])),
        tuple(part["actor_location"]),
        tuple(part["actor_size"]),
        target_class,
        target_prompt,
    )


def build_layout_mesh_contract(
    raw_layout: List[Dict[str, Any]],
    contract_path: Path,
    layout_id: Optional[str] = None,
) -> Dict[str, Any]:
    layout_id = layout_id or derive_layout_id(raw_layout)
    sorted_parts = sorted(raw_layout, key=_part_canonical_tuple)

    contract_parts: List[Dict[str, Any]] = []
    seen_part_ids = set()
    for index, part in enumerate(sorted_parts):
        target_class = part.get("target_class_override") or _derive_target_class(part["source_materials"])
        target_prompt = part.get("target_prompt_override") or _derive_target_prompt(target_class)
        canonical_tuple = _part_canonical_tuple(part)
        short_hash = _content_hash(canonical_tuple, length=10)
        part_id = part.get("part_id_override") or "{}__{:04d}__{}__{}".format(
            layout_id,
            index,
            target_class,
            short_hash,
        )
        part_id = _sanitize_part_id(str(part_id))

        duplicate_index = 1
        base_part_id = part_id
        while part_id in seen_part_ids:
            part_id = "{}__dup{}".format(base_part_id, duplicate_index)
            duplicate_index += 1

        part_for_prompt = {
            **part,
            "target_class": target_class,
            "target_prompt": target_prompt,
            "bbox": {
                "center": list(part["actor_location"]),
                "size": list(part["actor_size"]),
            },
        }
        part_prompt = build_part_prompt(part_for_prompt)
        negative_prompt = negative_prompt_for_part(part_for_prompt)
        part_description_source = component_source_description(part_for_prompt)
        part_description_core = component_core_description(part_for_prompt)
        part_visual_subject = component_visual_subject(part_for_prompt)
        part_context_tail = component_context_tail(part_for_prompt)
        reference_image_path = str(
            contract_path.resolve().parents[1] / "parts" / part_id / "reference.png"
        )

        contract_part = {
            "schema_version": SCHEMA_VERSION,
            "layout_id": layout_id,
            "part_id": part_id,
            "source_actor_label": part["source_actor_label"],
            "source_materials": list(part["source_materials"]),
            "target_prompt": target_prompt,
            "target_class": target_class,
            "part_prompt": part_prompt,
            "negative_prompt": negative_prompt,
            "prompt_hash": prompt_hash(part_prompt, negative_prompt),
            "prompt_policy_version": PROMPT_POLICY_VERSION,
            "part_description_source": part_description_source,
            "part_description_core": part_description_core,
            "part_visual_subject": part_visual_subject,
            "part_context_tail": part_context_tail,
            "reference_image_mode": "t2i_part_level",
            "reference_image_path": reference_image_path,
            "reference_image": reference_image_path,
            "t2i_provider": "siliconflow_qwen_image",
            "t2i_seed": 1234,
            "t2i_status": "pending",
            "actor_size": list(part["actor_size"]),
            "actor_location": list(part["actor_location"]),
            "bbox": {
                "center": list(part["actor_location"]),
                "size": list(part["actor_size"]),
            },
            "units": UNITS,
            "coordinate_frame": COORDINATE_FRAME,
            "orientation_policy": ORIENTATION_POLICY,
            "bbox_ownership": BBOX_OWNERSHIP,
            "contract_path": str(contract_path.resolve()),
            "lifecycle_state": "contract_emitted",
        }
        for metadata_key in (
            "source_part_id",
            "source_part_name",
            "open_vocab_label",
            "part_description",
            "generation_prompt",
            "semantic_role",
            "spatial_relations",
            "detail_level",
            "material_hint",
            "legacy_compatibility_hint",
            "open_semantic_layout_id",
            "source_bbox_size",
            "bbox_repair_warning",
        ):
            if metadata_key in part and part[metadata_key] not in (None, "", []):
                contract_part[metadata_key] = part[metadata_key]

        seen_part_ids.add(part_id)
        contract_parts.append(contract_part)

    return {
        "schema_version": SCHEMA_VERSION,
        "layout_id": layout_id,
        "units": UNITS,
        "coordinate_frame": COORDINATE_FRAME,
        "orientation_policy": ORIENTATION_POLICY,
        "bbox_ownership": BBOX_OWNERSHIP,
        "contract_path": str(contract_path.resolve()),
        "parts": contract_parts,
    }


def prepare_run_directories(
    raw_layout_path: Path,
    run_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "Run directory {} already exists and is not empty".format(run_dir)
        )

    input_dir = run_dir / "input"
    contract_dir = run_dir / "contract"
    parts_dir = run_dir / "parts"
    raw_assembly_dir = run_dir / "assemblies" / "raw_hunyuan_assembly"
    placeholder_assembly_dir = run_dir / "assemblies" / "placeholder_filled_assembly"

    for directory in (
        input_dir,
        contract_dir,
        parts_dir,
        raw_assembly_dir,
        placeholder_assembly_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    raw_layout_payload = json.loads(raw_layout_path.read_text())
    normalized_layout = normalize_layout_payload(raw_layout_payload)
    copied_layout_path = input_dir / RAW_LAYOUT_FILENAME
    copied_layout_path.write_text(json.dumps(raw_layout_payload, indent=2) + "\n")

    contract_path = contract_dir / CONTRACT_FILENAME
    contract = build_layout_mesh_contract(
        normalized_layout,
        contract_path,
        layout_id=derive_layout_id(raw_layout_payload),
    )
    if isinstance(raw_layout_payload, dict) and is_open_semantic_layout(raw_layout_payload):
        for metadata_key in (
            "scene_bounds",
            "coordinate_frame",
            "unit_scale",
            "geometry_continuity_contract",
            "geometry_continuity_metrics",
            "gravity_support_contract",
            "gravity_support_metrics",
            "ground_plane_alignment_contract",
            "ground_plane_alignment_metrics",
            "coordinate_space",
            "units",
            "archetype_axes",
            "contract_note",
            "title",
        ):
            if metadata_key in raw_layout_payload and raw_layout_payload[metadata_key] not in (None, "", []):
                contract[metadata_key] = raw_layout_payload[metadata_key]
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")

    for part in contract["parts"]:
        part_dir = parts_dir / part["part_id"]
        (part_dir / "raw_hunyuan").mkdir(parents=True, exist_ok=True)
        (part_dir / "placeholder").mkdir(parents=True, exist_ok=True)
        (part_dir / "request.json").write_text(json.dumps(part, indent=2) + "\n")
        (part_dir / "stdout.log").touch()
        (part_dir / "stderr.log").touch()

    return {
        "run_dir": str(run_dir),
        "layout_path": str(copied_layout_path.resolve()),
        "contract_path": str(contract_path.resolve()),
        "layout_id": contract["layout_id"],
        "part_ids": [part["part_id"] for part in contract["parts"]],
    }
