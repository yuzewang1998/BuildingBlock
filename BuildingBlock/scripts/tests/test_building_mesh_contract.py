#!/usr/bin/env python
"""Minimal contract smoke checks for the BuildingBlock mesh pipeline."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.assembly import (  # noqa: E402
    MeshData,
    build_assemblies,
    create_box_mesh,
    hunyuan_y_up_to_layout_z_up,
    mesh_bounds,
    normalize_mesh_to_bbox,
    shrink_wall_placeholder_size,
    write_obj_mesh,
)
from scene_synthesis.building_mesh.layout_io import (  # noqa: E402
    build_layout_mesh_contract,
    derive_layout_id,
    normalize_layout_payload,
)
from scene_synthesis.building_mesh.prompting import (  # noqa: E402
    build_part_prompt,
    negative_prompt_for_part,
    recommended_t2i_canvas_size,
    visual_ratio_for_part,
)


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

    y_up_mesh = MeshData(vertices=[(1, 2, 3), (-1, -2, -3)], faces=[(1, 2)])
    z_up_mesh = hunyuan_y_up_to_layout_z_up(y_up_mesh)
    assert z_up_mesh.vertices == [(1.0, -3.0, 2.0), (-1.0, 3.0, -2.0)]

    shrunken = shrink_wall_placeholder_size((10, 20, 5), xy_scale=0.9, z_scale=0.8)
    assert shrunken == (9.0, 18.0, 4.0)

    ratio, axes, width, height = visual_ratio_for_part("arbitrary_component", (0.005, 0.12, 0.06))
    assert axes == ("y", "z")
    assert round(ratio, 3) == 2.0
    assert (width, height) == (0.12, 0.06)
    assert recommended_t2i_canvas_size("arbitrary_component", (0.005, 0.12, 0.06)) == (1664, 928)
    prompt = build_part_prompt({
        "target_class": "arbitrary_component",
        "target_prompt": "3d mesh of arbitrary component",
        "bbox": {"size": (0.005, 0.12, 0.06)},
    })
    assert "standalone architectural component" in prompt
    assert "main visible silhouette" in prompt
    assert "2.00:1" in prompt
    assert "no support stand" in prompt
    assert "no long bottom rail" in prompt
    assert "component plus base" in prompt
    assert "top of the image is the architectural top" in prompt
    assert "visible face is the outward-facing exterior side" in prompt
    assert "closed rectangular window" not in prompt

    wall_prompt = build_part_prompt({
        "target_class": "wall",
        "target_prompt": "front_facade_wall_with_arch_supports",
        "bbox": {"size": (0.8, 0.18, 0.77)},
    })
    assert "facade elevation texture" in wall_prompt
    assert "no repeated tile pattern" in wall_prompt
    assert "projected once onto the detected outside face" in wall_prompt
    assert "top edge of the image is architectural up" in wall_prompt
    assert "horizontal facade courses and stone joints run left-to-right" in wall_prompt
    assert "cube" in negative_prompt_for_part({"target_class": "wall"})
    return 0


def _assert_bounds_close(mesh_path, expected_min, expected_max, places=6):
    from scene_synthesis.building_mesh.assembly import read_obj_mesh

    actual_min, actual_max = mesh_bounds(read_obj_mesh(mesh_path))
    assert tuple(round(value, places) for value in actual_min) == tuple(
        round(value, places) for value in expected_min
    )
    assert tuple(round(value, places) for value in actual_max) == tuple(
        round(value, places) for value in expected_max
    )


def validate_build_assemblies_layout_fit() -> int:
    """Ensure assembly normalizes each generated part into its layout bbox.

    The S2 Assemble stage must preserve the Stage1/contract layout when it
    combines independently generated meshes.  This regression covers the
    important contract: every successful Hunyuan OBJ is converted to layout
    coordinates, normalized to the part bbox, and included in both the raw and
    placeholder-filled scene assemblies.  Missing parts may only fall back to a
    placeholder after a recorded Hunyuan submission attempt.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source_a = write_obj_mesh(
            create_box_mesh(center=(100.0, -20.0, 5.0), size=(10.0, 4.0, 2.0)),
            root / "source_a.obj",
        )
        source_b = write_obj_mesh(
            create_box_mesh(center=(-9.0, 7.0, 3.0), size=(0.5, 8.0, 4.0)),
            root / "source_b.obj",
        )
        contracts = [
            {
                "schema_version": "layout_mesh_contract.v1",
                "layout_id": "assembly_layout_fit_smoke",
                "part_id": "part_a",
                "target_class": "open_semantic_part",
                "bbox": {"center": [1.0, 2.0, 3.0], "size": [2.0, 4.0, 6.0]},
            },
            {
                "schema_version": "layout_mesh_contract.v1",
                "layout_id": "assembly_layout_fit_smoke",
                "part_id": "part_b",
                "target_class": "open_semantic_part",
                "bbox": {"center": [-3.0, 0.5, 1.0], "size": [1.0, 2.0, 0.5]},
            },
        ]
        results = [
            {
                "part_id": "part_a",
                "raw_output_path": source_a,
                "lifecycle_states": ["submitted_to_hunyuan", "hunyuan_succeeded"],
            },
            {
                "part_id": "part_b",
                "raw_output_path": source_b,
                "lifecycle_states": ["submitted_to_hunyuan", "hunyuan_succeeded"],
            },
        ]
        assembly = build_assemblies(contracts, results, root / "assemblies")

        assert assembly.layout_id == "assembly_layout_fit_smoke"
        assert assembly.raw_assembly_path is not None
        assert Path(assembly.raw_assembly_path).exists()
        assert Path(assembly.placeholder_assembly_path).exists()
        assert len(assembly.parts) == 2
        by_part = {part.part_id: part for part in assembly.parts}

        _assert_bounds_close(by_part["part_a"].normalized_output_path, (0.0, 0.0, 0.0), (2.0, 4.0, 6.0))
        _assert_bounds_close(by_part["part_b"].normalized_output_path, (-3.5, -0.5, 0.75), (-2.5, 1.5, 1.25))
        _assert_bounds_close(assembly.raw_assembly_path, (-3.5, -0.5, 0.0), (2.0, 4.0, 6.0))
        _assert_bounds_close(assembly.placeholder_assembly_path, (-3.5, -0.5, 0.0), (2.0, 4.0, 6.0))

        for part in assembly.parts:
            assert "assembled_raw" in part.lifecycle_states
            assert "assembled_placeholder" in part.lifecycle_states
            assert part.placeholder_output_path == part.normalized_output_path

        missing_contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "assembly_layout_fit_smoke",
            "part_id": "part_missing",
            "target_class": "wall",
            "bbox": {"center": [0.0, 0.0, 0.0], "size": [0.2, 0.4, 0.6]},
        }
        missing_assembly = build_assemblies(
            [missing_contract],
            [{"part_id": "part_missing", "lifecycle_states": ["submitted_to_hunyuan", "hunyuan_failed"]}],
            root / "missing_assemblies",
        )
        missing_part = missing_assembly.parts[0]
        assert missing_part.normalized_output_path is None
        assert missing_part.placeholder_output_path.endswith("part_missing__placeholder.obj")
        assert "placeholder_used" in missing_part.lifecycle_states
        assert "assembled_placeholder" in missing_part.lifecycle_states
        assert Path(missing_part.placeholder_output_path).exists()

        try:
            build_assemblies(
                [missing_contract],
                [{"part_id": "part_missing", "lifecycle_states": ["hunyuan_failed"]}],
                root / "invalid_missing_assemblies",
            )
        except ValueError as exc:
            assert "placeholder fallback requires a submitted Hunyuan attempt" in str(exc)
        else:
            raise AssertionError("missing unsubmitted part must not silently produce placeholder")

    return 0


def validate_open_semantic_conversion() -> int:
    payload = {
        "layout_id": "open_vocab_smoke",
        "prompt": "a civic test building with arbitrary components",
        "s2_contract_version": "open_semantic_parts.v0",
        "parts": [
            {
                "part_id": "open_vocab_smoke__000__folded_entry_ribbon",
                "part_description": "folded bronze entry ribbon wrapping over the doorway",
                "semantic_role": "public interface",
                "detail_level": "secondary",
                "spatial_relations": ["above the entry", "in front of the main block"],
                "bbox": {"center": [0.0, -0.35, -0.12], "size": [0.5, 0.12, 0.18]},
                "generation_prompt": "Create one architectural component: folded bronze entry ribbon wrapping over the doorway.",
                "legacy_compatibility_hint": "object",
            }
        ],
    }
    normalized = normalize_layout_payload(payload)
    assert len(normalized) == 1
    assert normalized[0]["target_class_override"] == "open_semantic_part"
    assert normalized[0]["part_description"] == "folded bronze entry ribbon wrapping over the doorway"
    assert normalized[0]["spatial_relations"] == ["above the entry", "in front of the main block"]
    layout_id = derive_layout_id(payload)
    assert layout_id == "open_vocab_smoke"
    contract = build_layout_mesh_contract(normalized, Path("/tmp/open_vocab_smoke/contract/layout_mesh_contract.json"), layout_id)
    part = contract["parts"][0]
    assert part["target_class"] == "open_semantic_part"
    assert part["part_id"] == "open_vocab_smoke__000__folded_entry_ribbon"
    assert part["open_vocab_label"] == "folded bronze entry ribbon wrapping over the doorway"
    assert part["semantic_role"] == "public interface"
    assert "folded bronze entry ribbon" in part["target_prompt"]
    assert "open-vocabulary architectural component" in part["part_prompt"]
    assert "main visible silhouette" in part["part_prompt"]
    assert part["legacy_compatibility_hint"] == "object"

    payload_v1 = {
        **payload,
        "s2_contract_version": "open_semantic_parts.v1.geometry_continuity",
        "scene_bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]},
        "geometry_continuity_contract": {
            "target": "single_connected_bbox_contact_graph",
            "overlap_policy": "allowed_and_preferred_for_s2_union",
            "gap_policy": "forbid_unintended_component_gaps",
        },
    }
    normalized_v1 = normalize_layout_payload(payload_v1)
    assert normalized_v1[0]["part_description"] == "folded bronze entry ribbon wrapping over the doorway"

    degenerate_payload = {
        **payload,
        "parts": [
            {
                **payload["parts"][0],
                "part_id": "open_vocab_smoke__001__flat_market_marker",
                "part_description": "flat market stall marker visible behind arcade",
                "bbox": {"center": [0.0, 0.0, 0.0], "size": [0.22, 0.18, 0.0]},
            }
        ],
    }
    normalized_degenerate = normalize_layout_payload(degenerate_payload)
    warning = normalized_degenerate[0]["bbox_repair_warning"]
    assert normalized_degenerate[0]["actor_size"] == [0.22, 0.18, 0.045]
    assert normalized_degenerate[0]["source_bbox_size"] == [0.22, 0.18, 0.0]
    assert warning["type"] == "non_positive_bbox_axis_repaired"
    assert warning["original_size"] == [0.22, 0.18, 0.0]
    assert warning["repaired_size"] == [0.22, 0.18, 0.045]
    contract_degenerate = build_layout_mesh_contract(
        normalized_degenerate,
        Path("/tmp/open_vocab_smoke/contract/layout_mesh_contract.json"),
        "open_vocab_smoke",
    )
    degenerate_part = contract_degenerate["parts"][0]
    assert degenerate_part["bbox"]["size"] == [0.22, 0.18, 0.045]
    assert degenerate_part["bbox_repair_warning"]["replacement_extent"] == 0.045
    return 0


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("contract_json", type=Path)
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args(argv)
    if args.geometry:
        validate_geometry()
        validate_build_assemblies_layout_fit()
        validate_open_semantic_conversion()
    return validate_contract(args.contract_json)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
