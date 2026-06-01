#!/usr/bin/env python3
"""Rebuild ArchStudio-S2 combined assembly OBJ files from per-part OBJ meshes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.assembly import MeshData, read_obj_mesh, write_obj_mesh  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_part_obj(run_dir: Path, manifest_part: dict[str, Any]) -> Path:
    part_id = str(manifest_part.get("part_id"))
    candidates = [
        manifest_part.get("normalized_output_path"),
        manifest_part.get("placeholder_output_path"),
        run_dir / "assemblies" / "parts" / f"{part_id}.obj",
        manifest_part.get("raw_output_path"),
        run_dir / "parts" / part_id / f"{part_id}.obj",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"no part OBJ found for {part_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--header-tag", default="rebuilt_from_manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    assembly_dir = run_dir / "assemblies"
    raw_path = assembly_dir / "raw_hunyuan_assembly.obj"
    complete_path = assembly_dir / "placeholder_filled_assembly.obj"
    combined = MeshData(vertices=[], faces=[])
    resolved_parts: list[dict[str, str]] = []
    for part in manifest.get("parts", []) or []:
        part_id = str(part.get("part_id"))
        obj_path = resolve_part_obj(run_dir, part)
        combined.extend(read_obj_mesh(obj_path))
        part["normalized_output_path"] = str(obj_path)
        part["placeholder_output_path"] = str(obj_path)
        part["raw_assembly_path"] = str(raw_path)
        part["placeholder_assembly_path"] = str(complete_path)
        resolved_parts.append({"part_id": part_id, "obj": str(obj_path)})
    write_obj_mesh(combined, raw_path, header=f"{args.header_tag} raw assembly")
    write_obj_mesh(combined, complete_path, header=f"{args.header_tag} complete/fallback assembly")
    manifest["raw_assembly_path"] = str(raw_path)
    manifest["placeholder_assembly_path"] = str(complete_path)
    manifest.setdefault("assembly_rebuild_history", []).append(
        {
            "header_tag": args.header_tag,
            "num_parts": len(resolved_parts),
            "vertex_count": len(combined.vertices),
            "face_count": len(combined.faces),
            "raw_assembly_path": str(raw_path),
            "placeholder_assembly_path": str(complete_path),
        }
    )
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "num_parts": len(resolved_parts),
                "vertex_count": len(combined.vertices),
                "face_count": len(combined.faces),
                "raw_assembly_path": str(raw_path),
                "placeholder_assembly_path": str(complete_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
