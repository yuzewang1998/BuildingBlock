#!/usr/bin/env python
"""Minimal contract smoke checks for the BuildingBlock mesh pipeline."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.assembly import create_box_mesh, mesh_bounds, normalize_mesh_to_bbox  # noqa: E402


def validate_contract(contract_path: Path) -> int:
    contract = json.loads(contract_path.read_text())

    assert contract["schema_version"] == "layout_mesh_contract.v1"
    assert contract["coordinate_frame"]["handedness"] == "right-handed"
    assert contract["coordinate_frame"]["axes"] == {
        "x": "right",
        "y": "forward",
        "z": "up",
    }
    assert contract["orientation_policy"] == "axis_aligned_no_extra_rotation"
    assert contract["bbox_ownership"] == "full_part_bbox"

    parts = contract["parts"]
    assert parts, "contract must contain at least one part"

    part_ids = [part["part_id"] for part in parts]
    assert len(part_ids) == len(set(part_ids)), "part_id values must be unique"

    for part in parts:
        bbox = part["bbox"]
        assert set(bbox.keys()) == {"center", "size"}
        assert len(bbox["center"]) == 3
        assert len(bbox["size"]) == 3
        assert all(value >= 0 for value in bbox["size"])

    return 0


def validate_geometry() -> int:
    mesh = create_box_mesh(center=(0, 0, 0), size=(1, 1, 1))
    normalized = normalize_mesh_to_bbox(mesh, center=(10, 20, 30), size=(2, 4, 6))
    bounds_min, bounds_max = mesh_bounds(normalized)
    assert bounds_min == (9.0, 18.0, 27.0)
    assert bounds_max == (11.0, 22.0, 33.0)
    return 0


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("contract_json", type=Path)
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args(argv)
    if args.geometry:
        validate_geometry()
    return validate_contract(args.contract_json)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
