#!/usr/bin/env python3
"""Apply traceable semantic primitive repairs to an ArchStudio-S2 run.

This is an additive/local repair pass for open-vocabulary S2 experiments.  It
does not change the layout contract, T2I inputs, or raw Hunyuan-Omni outputs.
Instead it rewrites selected normalized part OBJs in ``assemblies/parts`` with
small procedural meshes when the natural-language semantics are better
represented by a simple architectural primitive than by the current generated
mesh (for example a stair flight, support column, opening frame, or parapet
rail).

The classifier intentionally uses broad semantic words from S1/S2 contracts
rather than a closed building-part class list, so it remains compatible with
open-vocabulary part descriptions.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bbox_center_size(part: Mapping[str, Any]) -> Tuple[Vec3, Vec3]:
    bbox = part.get("bbox") or {}
    center = tuple(float(value) for value in bbox.get("center", []))
    size = tuple(float(value) for value in bbox.get("size", []))
    if len(center) != 3 or len(size) != 3:
        raise ValueError(f"part {part.get('part_id')} is missing bbox.center/size")
    if any(value <= 0.0 for value in size):
        raise ValueError(f"part {part.get('part_id')} has non-positive bbox size: {size}")
    return center, size


def box_vertices(center: Sequence[float], size: Sequence[float]) -> List[Vec3]:
    cx, cy, cz = (float(value) for value in center)
    sx, sy, sz = (float(value) for value in size)
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    return [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]


def add_box(vertices: List[Vec3], faces: List[Face], center: Sequence[float], size: Sequence[float]) -> None:
    base = len(vertices) + 1
    vertices.extend(box_vertices(center, size))
    faces.extend(
        [
            (base + 0, base + 3, base + 2, base + 1),
            (base + 4, base + 5, base + 6, base + 7),
            (base + 0, base + 1, base + 5, base + 4),
            (base + 1, base + 2, base + 6, base + 5),
            (base + 2, base + 3, base + 7, base + 6),
            (base + 3, base + 0, base + 4, base + 7),
        ]
    )


def add_cylinder(
    vertices: List[Vec3],
    faces: List[Face],
    center: Sequence[float],
    radius_x: float,
    radius_y: float,
    height: float,
    *,
    segments: int = 32,
) -> None:
    cx, cy, cz = (float(value) for value in center)
    z0, z1 = cz - height / 2.0, cz + height / 2.0
    bottom: List[int] = []
    top: List[int] = []
    for index in range(max(8, int(segments))):
        angle = 2.0 * math.pi * index / max(8, int(segments))
        x = cx + float(radius_x) * math.cos(angle)
        y = cy + float(radius_y) * math.sin(angle)
        bottom.append(len(vertices) + 1)
        vertices.append((x, y, z0))
        top.append(len(vertices) + 1)
        vertices.append((x, y, z1))
    faces.append(tuple(reversed(bottom)))
    faces.append(tuple(top))
    count = len(bottom)
    for index in range(count):
        nxt = (index + 1) % count
        faces.append((bottom[index], bottom[nxt], top[nxt], top[index]))


def add_support_column(vertices: List[Vec3], faces: List[Face], center: Vec3, size: Vec3) -> str:
    cx, cy, cz = center
    sx, sy, sz = size
    base_h = min(sz * 0.11, max(sz * 0.055, 0.035))
    cap_h = base_h
    shaft_h = max(sz - base_h - cap_h, sz * 0.68)
    radius_x = sx * 0.32
    radius_y = sy * 0.32
    add_cylinder(vertices, faces, (cx, cy, cz), radius_x, radius_y, shaft_h, segments=36)
    flute_w = max(min(sx, sy) * 0.055, 0.002)
    flute_d = max(min(sx, sy) * 0.055, 0.002)
    for dx, dy, box_size in [
        (radius_x * 0.93, 0.0, (flute_w, flute_d, shaft_h * 0.86)),
        (-radius_x * 0.93, 0.0, (flute_w, flute_d, shaft_h * 0.86)),
        (0.0, radius_y * 0.93, (flute_w, flute_d, shaft_h * 0.86)),
        (0.0, -radius_y * 0.93, (flute_w, flute_d, shaft_h * 0.86)),
    ]:
        add_box(vertices, faces, (cx + dx, cy + dy, cz), box_size)
    add_box(vertices, faces, (cx, cy, cz - sz / 2.0 + base_h / 2.0), (sx, sy, base_h))
    add_box(vertices, faces, (cx, cy, cz + sz / 2.0 - cap_h / 2.0), (sx, sy, cap_h))
    return "support semantics: shaft, flutes, base, and capital kept inside original bbox"


def add_frame(
    vertices: List[Vec3],
    faces: List[Face],
    center: Vec3,
    size: Vec3,
    *,
    rail_x: float,
    rail_z: float,
    include_bottom: bool,
    depth_scale: float = 0.82,
) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    rx = min(max(float(rail_x), sx * 0.06), sx * 0.34)
    rz = min(max(float(rail_z), sz * 0.04), sz * 0.30)
    depth = max(sy * float(depth_scale), 0.004)
    add_box(vertices, faces, (cx - sx / 2.0 + rx / 2.0, cy, cz), (rx, depth, sz))
    add_box(vertices, faces, (cx + sx / 2.0 - rx / 2.0, cy, cz), (rx, depth, sz))
    add_box(vertices, faces, (cx, cy, cz + sz / 2.0 - rz / 2.0), (sx, depth, rz))
    if include_bottom:
        add_box(vertices, faces, (cx, cy, cz - sz / 2.0 + rz / 2.0), (sx, depth, rz))


def add_opening_slot(vertices: List[Vec3], faces: List[Face], center: Vec3, size: Vec3) -> str:
    sx, sy, sz = size
    add_frame(
        vertices,
        faces,
        center,
        size,
        rail_x=max(sx * 0.14, 0.006),
        rail_z=max(sz * 0.075, 0.010),
        include_bottom=True,
        depth_scale=0.74,
    )
    cx, cy, cz = center
    # Thin mullions make the part read as an opening/window system without
    # filling the central void with a large slab.
    mullion_w = max(sx * 0.055, 0.0035)
    mullion_d = max(sy * 0.22, 0.004)
    add_box(vertices, faces, (cx, cy - sy * 0.10, cz), (mullion_w, mullion_d, sz * 0.70))
    add_box(vertices, faces, (cx, cy - sy * 0.11, cz), (sx * 0.54, mullion_d, max(sz * 0.035, 0.006)))
    return "opening semantics: perimeter frame plus mullions, central area left mostly empty"


def add_recessed_portal(vertices: List[Vec3], faces: List[Face], center: Vec3, size: Vec3) -> str:
    sx, sy, sz = size
    add_frame(
        vertices,
        faces,
        center,
        size,
        rail_x=max(sx * 0.13, 0.012),
        rail_z=max(sz * 0.10, 0.016),
        include_bottom=False,
        depth_scale=0.95,
    )
    cx, cy, cz = center
    threshold_h = max(sz * 0.045, 0.010)
    add_box(vertices, faces, (cx, cy - sy * 0.10, cz - sz / 2.0 + threshold_h / 2.0), (sx * 0.74, sy * 0.48, threshold_h))
    # A very narrow back reveal improves the sense of depth but avoids a solid
    # full-height panel.
    add_box(vertices, faces, (cx, cy - sy * 0.30, cz), (sx * 0.18, max(sy * 0.10, 0.004), sz * 0.58))
    return "portal semantics: jamb/lintel frame with threshold and narrow recessed reveal"


def add_stepped_flight(vertices: List[Vec3], faces: List[Face], center: Vec3, size: Vec3) -> str:
    cx, cy, cz = center
    sx, sy, sz = size
    z_min = cz - sz / 2.0
    # Use the narrower horizontal axis as travel direction for broad stair
    # flights; otherwise default to layout Y-forward travel.  This is based on
    # bbox proportions, not a closed class taxonomy.
    travel_axis = 1 if sx >= sy else 0
    travel_len = sy if travel_axis == 1 else sx
    width_len = sx if travel_axis == 1 else sy
    step_count = max(4, min(9, int(round(travel_len / max(sz * 0.28, 0.035)))))
    step_depth = travel_len / float(step_count)
    side_thickness = max(min(width_len * 0.035, step_depth * 0.25), 0.006)
    for index in range(step_count):
        height = sz * float(index + 1) / float(step_count)
        local_travel_center = -travel_len / 2.0 + step_depth * (index + 0.5)
        if travel_axis == 1:
            box_center = (cx, cy + local_travel_center, z_min + height / 2.0)
            box_size = (sx, step_depth, height)
        else:
            box_center = (cx + local_travel_center, cy, z_min + height / 2.0)
            box_size = (step_depth, sy, height)
        add_box(vertices, faces, box_center, box_size)

    # Low side cheek walls make the stair flight readable in per-part-color
    # assembly views while staying within the bbox.
    cheek_h = sz * 0.38
    cheek_z = z_min + cheek_h / 2.0
    if travel_axis == 1:
        add_box(vertices, faces, (cx - sx / 2.0 + side_thickness / 2.0, cy, cheek_z), (side_thickness, sy, cheek_h))
        add_box(vertices, faces, (cx + sx / 2.0 - side_thickness / 2.0, cy, cheek_z), (side_thickness, sy, cheek_h))
    else:
        add_box(vertices, faces, (cx, cy - sy / 2.0 + side_thickness / 2.0, cheek_z), (sx, side_thickness, cheek_h))
        add_box(vertices, faces, (cx, cy + sy / 2.0 - side_thickness / 2.0, cheek_z), (sx, side_thickness, cheek_h))
    return f"circulation semantics: {step_count} visible treads/risers plus low cheek walls inside bbox"


def add_roofline_parapet(vertices: List[Vec3], faces: List[Face], center: Vec3, size: Vec3) -> str:
    cx, cy, cz = center
    sx, sy, sz = size
    # Thin footprint, but enough Z fill to pass deterministic bbox-fill checks.
    rail_w = min(max(min(sx, sy) * 0.020, 0.018), 0.032)
    rail_h = min(max(sz * 0.74, 0.060), sz * 0.84)
    rail_z = cz - sz * 0.03
    add_box(vertices, faces, (cx, cy + sy / 2.0 - rail_w / 2.0, rail_z), (sx, rail_w, rail_h))
    add_box(vertices, faces, (cx - sx / 2.0 + rail_w / 2.0, cy, rail_z), (rail_w, sy, rail_h))
    add_box(vertices, faces, (cx + sx / 2.0 - rail_w / 2.0, cy, rail_z), (rail_w, sy, rail_h))
    # Small cap strips add a stone-rail reading without filling the U.
    cap_h = min(sz * 0.16, 0.018)
    cap_w = min(rail_w * 1.55, 0.045)
    cap_z = rail_z + rail_h / 2.0 - cap_h / 2.0
    add_box(vertices, faces, (cx, cy + sy / 2.0 - rail_w / 2.0, cap_z), (sx, cap_w, cap_h))
    add_box(vertices, faces, (cx - sx / 2.0 + rail_w / 2.0, cy, cap_z), (cap_w, sy, cap_h))
    add_box(vertices, faces, (cx + sx / 2.0 - rail_w / 2.0, cy, cap_z), (cap_w, sy, cap_h))
    return "roofline semantics: thin U-shaped parapet rail with cap strips, no filled roof body"


def add_architectural_mass(vertices: List[Vec3], faces: List[Face], center: Vec3, size: Vec3) -> str:
    cx, cy, cz = center
    sx, sy, sz = size
    # A clean rectilinear volume is a better first-order architectural prior for
    # open-semantic primary masses than noisy per-part generative shards.  Add a
    # shallow plinth and top coping, both clipped inside the bbox, so the part
    # reads as a solid gallery bar rather than a raw cube.
    add_box(vertices, faces, center, size)
    plinth_h = min(max(sz * 0.055, 0.014), sz * 0.12)
    coping_h = min(max(sz * 0.040, 0.012), sz * 0.10)
    add_box(vertices, faces, (cx, cy, cz - sz / 2.0 + plinth_h / 2.0), (sx, sy, plinth_h))
    add_box(vertices, faces, (cx, cy, cz + sz / 2.0 - coping_h / 2.0), (sx, sy, coping_h))
    return "primary-mass semantics: clean bbox-aligned gallery volume with subtle plinth/coping; layout bbox preserved"


def add_paving_plane(vertices: List[Vec3], faces: List[Face], center: Vec3, size: Vec3) -> str:
    cx, cy, cz = center
    sx, sy, sz = size
    # Keep the full bbox extent so metrics and assembly contracts remain stable,
    # but use a simple slab with a few expansion joints for visual readability.
    add_box(vertices, faces, center, size)
    joint_w = max(min(sx, sy) * 0.006, 0.002)
    joint_h = min(sz * 0.28, 0.006)
    z = cz + sz / 2.0 - joint_h / 2.0
    for frac in (-0.25, 0.0, 0.25):
        add_box(vertices, faces, (cx + sx * frac, cy, z), (joint_w, sy * 0.96, joint_h))
        add_box(vertices, faces, (cx, cy + sy * frac, z), (sx * 0.96, joint_w, joint_h))
    return "ground/paving semantics: clean low slab with shallow expansion-joint relief; bbox preserved"


def write_obj(path: Path, vertices: Sequence[Vec3], faces: Sequence[Face], *, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {header}\n")
        handle.write("# generated_by scripts/apply_archstudio_s2_semantic_geometry_repairs.py\n")
        for vertex in vertices:
            handle.write("v {:.9f} {:.9f} {:.9f}\n".format(*vertex))
        for face in faces:
            handle.write("f {}\n".format(" ".join(str(index) for index in face)))


def bounds(vertices: Sequence[Vec3]) -> Tuple[List[float], List[float]]:
    return (
        [min(vertex[axis] for vertex in vertices) for axis in range(3)],
        [max(vertex[axis] for vertex in vertices) for axis in range(3)],
    )


def semantic_text(part: Mapping[str, Any]) -> str:
    # Use only local part identity/semantics for classification.  Generation
    # prompts often contain whole-scene context or negative constraints (for
    # example other parts such as stairs/columns), so using them here causes
    # cross-part false positives.
    fields = [
        part.get("part_id", ""),
        part.get("part_description", ""),
        part.get("open_vocab_label", ""),
        part.get("source_part_name", ""),
        part.get("source_actor_label", ""),
        part.get("semantic_role", ""),
        part.get("legacy_compatibility_hint", ""),
        part.get("target_class", ""),
    ]
    return " ".join(str(value) for value in fields if value).lower()


def has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def classify_semantic_primitive(part: Mapping[str, Any]) -> str | None:
    text = semantic_text(part)
    role = str(part.get("semantic_role", "")).lower()
    # Order matters: "portal behind column row" should be an opening, not a
    # support.  "courtyard void marker" should not become an opening unless S1
    # explicitly marks it as an opening system.
    if has_any(text, ("stair", "staircase", "stairway", "tread", "riser", "step flight", "stair flight")):
        return "stepped_flight"
    if role in {"primary mass", "primary_mass"} or has_any(text, ("gallery bar", "gallery wing", "thick exhibit wall", "main mass", "building mass")):
        return "architectural_mass"
    if role in {"landscape edge", "ground plane", "paving", "courtyard"} or has_any(text, ("paving plane", "ground plane", "courtyard void marker", "low paving")):
        return "paving_plane"
    if has_any(text, ("parapet", "roofline")) or ("rail" in text and "roof" in text):
        return "roofline_parapet"
    if has_any(text, ("window", "slot")) or role == "opening system":
        if has_any(text, ("portal", "archway", "entry portal", "doorway")):
            return "recessed_portal"
        return "opening_slot"
    if has_any(text, ("portal", "archway", "doorway")):
        return "recessed_portal"
    if has_any(text, ("column", "pillar")) and not has_any(text, ("no column", "no columns", "without column")):
        return "support_column"
    return None


def build_repair_mesh(kind: str, center: Vec3, size: Vec3) -> Tuple[List[Vec3], List[Face], str]:
    vertices: List[Vec3] = []
    faces: List[Face] = []
    if kind == "support_column":
        rationale = add_support_column(vertices, faces, center, size)
    elif kind == "opening_slot":
        rationale = add_opening_slot(vertices, faces, center, size)
    elif kind == "recessed_portal":
        rationale = add_recessed_portal(vertices, faces, center, size)
    elif kind == "stepped_flight":
        rationale = add_stepped_flight(vertices, faces, center, size)
    elif kind == "roofline_parapet":
        rationale = add_roofline_parapet(vertices, faces, center, size)
    elif kind == "architectural_mass":
        rationale = add_architectural_mass(vertices, faces, center, size)
    elif kind == "paving_plane":
        rationale = add_paving_plane(vertices, faces, center, size)
    else:
        raise ValueError(f"unsupported semantic primitive: {kind}")
    return vertices, faces, rationale


def resolve_contract(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    candidates = [
        run_dir / "contract" / "layout_mesh_contract.json",
        run_dir / "layout_mesh_contract.json",
        run_dir / "input" / "layout_mesh_contract.original.json",
    ]
    for item in manifest.get("parts", []) or []:
        contract_path = item.get("contract_path")
        if contract_path:
            candidates.append(Path(contract_path))
    for candidate in candidates:
        if candidate.exists():
            payload = load_json(candidate)
            if isinstance(payload, dict) and payload.get("parts"):
                return candidate
    raise FileNotFoundError(f"no layout_mesh_contract.json found for {run_dir}")


def update_manifest_part(manifest_part: dict[str, Any], output_path: Path, repair_id: str, kind: str, rationale: str) -> None:
    manifest_part["normalized_output_path"] = str(output_path)
    manifest_part["placeholder_output_path"] = str(output_path)
    states = list(manifest_part.get("lifecycle_states") or [])
    state = f"agent_semantic_geometry_repaired_{repair_id}"
    if state not in states:
        states.append(state)
    manifest_part["lifecycle_states"] = states
    manifest_part.setdefault("repair_history", []).append(
        {
            "repair_id": repair_id,
            "kind": kind,
            "rationale": rationale,
            "output_path": str(output_path),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repair-id", default="v9repair04_semantic_geometry_refine")
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=["stepped_flight", "opening_slot", "recessed_portal", "roofline_parapet"],
        help="Semantic primitive kinds to rewrite. Add support_column if columns should be regenerated too.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-existing-backup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    contract_path = resolve_contract(run_dir, manifest)
    contract = load_json(contract_path)
    manifest_by_id: dict[str, dict[str, Any]] = {
        str(item.get("part_id")): item
        for item in manifest.get("parts", []) or []
        if item.get("part_id")
    }
    enabled_kinds = set(args.kinds or [])
    actions: list[dict[str, Any]] = []
    for part in contract.get("parts", []) or []:
        part_id = str(part.get("part_id"))
        kind = classify_semantic_primitive(part)
        if kind is None or kind not in enabled_kinds:
            continue
        center, size = bbox_center_size(part)
        vertices, faces, rationale = build_repair_mesh(kind, center, size)
        output_path = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        backup_path = output_path.with_suffix(output_path.suffix + f".pre_{args.repair_id}.bak")
        mn, mx = bounds(vertices)
        action = {
            "part_id": part_id,
            "kind": kind,
            "rationale": rationale,
            "bbox_center": list(center),
            "bbox_size": list(size),
            "output_path": str(output_path),
            "backup_path": str(backup_path),
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "bounds_min": mn,
            "bounds_max": mx,
            "dry_run": bool(args.dry_run),
        }
        actions.append(action)
        if args.dry_run:
            continue
        if output_path.exists() and (args.overwrite_existing_backup or not backup_path.exists()):
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, backup_path)
        write_obj(output_path, vertices, faces, header=f"{args.repair_id} {kind} repair for {part_id}")
        if part_id in manifest_by_id:
            update_manifest_part(manifest_by_id[part_id], output_path, args.repair_id, kind, rationale)

    note_payload = {
        "repair_id": args.repair_id,
        "run_dir": str(run_dir),
        "contract_path": str(contract_path),
        "contract_preserved": True,
        "source_run_overwritten": False,
        "enabled_kinds": sorted(enabled_kinds),
        "num_actions": len(actions),
        "actions": actions,
    }
    if not args.dry_run:
        note_path = run_dir / "repair_notes" / f"{args.repair_id}_actions.json"
        write_json(note_path, note_payload)
        manifest.setdefault("v9repair_history", []).append(
            {
                "repair_id": args.repair_id,
                "note": "Semantic primitive repair pass over normalized assembly part OBJs; layout contract and raw generation inputs preserved.",
                "num_parts": len(actions),
                "repair_notes": str(note_path),
            }
        )
        write_json(manifest_path, manifest)
    print(json.dumps(note_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
