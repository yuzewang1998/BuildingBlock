"""Contract emission utilities for the BuildingBlock layout-to-mesh pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .prompting import NEGATIVE_PROMPT, build_part_prompt, prompt_hash


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
    return normalize_raw_layout(json.loads(layout_path.read_text()))


def derive_layout_id(raw_layout_payload: Iterable[Any]) -> str:
    normalized_parts = normalize_raw_layout(list(raw_layout_payload))
    canonical_parts = [_part_canonical_tuple(part) for part in normalized_parts]
    canonical_parts.sort()
    return "layout_{}".format(_content_hash(canonical_parts))


def _part_canonical_tuple(part: Dict[str, Any]) -> Tuple[Any, ...]:
    target_class = _derive_target_class(part["source_materials"])
    return (
        part["source_actor_label"],
        tuple(sorted(part["source_materials"])),
        tuple(part["actor_location"]),
        tuple(part["actor_size"]),
        target_class,
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
        target_class = _derive_target_class(part["source_materials"])
        canonical_tuple = _part_canonical_tuple(part)
        short_hash = _content_hash(canonical_tuple, length=10)
        part_id = "{}__{:04d}__{}__{}".format(
            layout_id,
            index,
            target_class,
            short_hash,
        )

        duplicate_index = 1
        base_part_id = part_id
        while part_id in seen_part_ids:
            part_id = "{}__dup{}".format(base_part_id, duplicate_index)
            duplicate_index += 1

        part_for_prompt = {
            **part,
            "target_class": target_class,
            "bbox": {
                "center": list(part["actor_location"]),
                "size": list(part["actor_size"]),
            },
        }
        part_prompt = build_part_prompt(part_for_prompt)
        reference_image_path = str(
            contract_path.resolve().parents[1] / "parts" / part_id / "reference.png"
        )

        seen_part_ids.add(part_id)
        contract_parts.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "part_id": part_id,
                "source_actor_label": part["source_actor_label"],
                "source_materials": list(part["source_materials"]),
                "target_prompt": _derive_target_prompt(target_class),
                "target_class": target_class,
                "part_prompt": part_prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "prompt_hash": prompt_hash(part_prompt, NEGATIVE_PROMPT),
                "reference_image_mode": "t2i_part_level",
                "reference_image_path": reference_image_path,
                "reference_image": reference_image_path,
                "t2i_provider": "flux_schnell",
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
        )

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
    normalized_layout = normalize_raw_layout(raw_layout_payload)
    copied_layout_path = input_dir / RAW_LAYOUT_FILENAME
    copied_layout_path.write_text(json.dumps(raw_layout_payload, indent=2) + "\n")

    contract_path = contract_dir / CONTRACT_FILENAME
    contract = build_layout_mesh_contract(
        normalized_layout,
        contract_path,
        layout_id=derive_layout_id(raw_layout_payload),
    )
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
