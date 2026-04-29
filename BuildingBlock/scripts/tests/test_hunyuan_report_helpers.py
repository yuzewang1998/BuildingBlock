#!/usr/bin/env python
"""Smoke checks for Hunyuan command and report status helpers."""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.hunyuan_adapter import build_hunyuan_command  # noqa: E402
from scene_synthesis.building_mesh.report_bundle import (  # noqa: E402
    _failure_payload_for_report,
    _infer_part_output_paths,
    render_html,
)
from scene_synthesis.building_mesh.visualization import render_obj_mesh  # noqa: E402
from scene_synthesis.building_mesh.visualization import (  # noqa: E402
    export_contract_part_assembly_glb,
    export_layout_3d_boxes_glb,
)


def _contract_part(reference_image):
    return {
        "schema_version": "layout_mesh_contract.v1",
        "layout_id": "layout_test",
        "part_id": "layout_test__0000__window__abc",
        "target_prompt": "3d mesh of window",
        "target_class": "window",
        "contract_path": "/tmp/contract.json",
        "reference_image_path": reference_image,
        "bbox": {"center": [0, 0, 0], "size": [1, 2, 3]},
    }


def validate_command_image_repair():
    reference_image = "/tmp/reference.png"
    command = build_hunyuan_command(
        "python tool.py --image --bbox-sx $bbox_sx --save-dir $output_dir",
        contract_part=_contract_part(reference_image),
        output_dir="/tmp/out",
    )
    image_index = command.index("--image")
    assert command[image_index + 1] == reference_image
    assert "--bbox-sx" in command


def validate_reference_image_path_alias():
    reference_image = "/tmp/reference.png"
    command = build_hunyuan_command(
        "python tool.py --image $reference_image_path --save-dir $output_dir",
        contract_part=_contract_part(reference_image),
        output_dir="/tmp/out",
    )
    assert reference_image in command


def validate_report_output_inference():
    part_id = "layout_test__0000__window__abc"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        part_dir = run_dir / "parts" / part_id
        part_dir.mkdir(parents=True)
        raw_mesh = part_dir / f"{part_id}.obj"
        raw_mesh.write_text("v 0 0 0\nf 1\n")
        inferred = _infer_part_output_paths(run_dir, part_id, {})
        assert inferred["raw_output_path"] == str(raw_mesh)
        assert inferred["normalized_output_path"] == str(raw_mesh)


def validate_failure_payload_synthesizes_missing_rows():
    row = {
        "part_id": "layout_test__0001__window__def",
        "target_class": "window",
        "status": "failed",
        "raw_output_path": None,
        "normalized_output_path": None,
        "placeholder_output_path": "/tmp/placeholder.obj",
        "lifecycle_states": ["hunyuan_failed", "placeholder_used"],
    }
    payload = _failure_payload_for_report({"failures": []}, [row])
    assert len(payload["failures"]) == 1
    assert payload["failures"][0]["failure_type"] == "missing_output_mesh"


def validate_mesh_surface_render():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        obj_path = tmp_path / "triangle.obj"
        obj_path.write_text(
            "\n".join([
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
                "",
            ])
        )
        output_path = tmp_path / "triangle.png"
        rendered = render_obj_mesh(obj_path, output_path, title="triangle", max_faces=10)
        assert rendered == str(output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def validate_render_html_falls_back_to_images_without_glb():
    summary = {
        "run_dir": "/tmp/run",
        "num_parts": 1,
        "num_success": 1,
        "num_failures": 0,
        "num_failure_records": 0,
        "class_distribution": {"window": 1},
        "layout_overview": "assets/layout_overview.png",
        "layout_xz": "assets/layout_xz.png",
        "layout_yz": "assets/layout_yz.png",
        "layout_3d": "assets/layout_3d_boxes.png",
        "layout_3d_model": None,
        "layout_3d_views": {"iso": "assets/layout_3d_view_iso.png"},
        "assembly_raw_image": "assets/assembly_raw_view_iso.png",
        "assembly_raw_model": None,
        "assembly_raw_views": {"iso": "assets/assembly_raw_view_iso.png"},
        "assembly_placeholder_image": "assets/assembly_raw_view_iso.png",
        "assembly_placeholder_views": {"iso": "assets/assembly_raw_view_iso.png"},
        "assembly_raw_obj": None,
        "assembly_placeholder_obj": None,
    }
    parts = [{
        "part_id": "layout_test__0000__window__abc",
        "target_class": "window",
        "status": "success",
        "prompt": "mesh of window",
        "reference_image": "assets/ref.png",
        "mesh_render": "assets/mesh.png",
        "layout_render": "assets/layout.png",
        "model_glb": None,
        "mesh_caption": "normalized mesh",
    }]
    html_output = render_html(summary, parts, {"failures": []})
    assert '<h2>Layout 3D bbox</h2>' in html_output
    assert '<img src="assets/layout_3d_boxes.png">' in html_output
    assert '<h2>Raw assembly</h2>' in html_output
    assert '<img src="assets/assembly_raw_view_iso.png">' in html_output
    assert 'alt="layout_test__0000__window__abc mesh preview"' in html_output


def validate_layout_boxes_glb_export_without_trimesh():
    contract = {
        "parts": [{
            "part_id": "layout_test__0000__window__abc",
            "target_class": "window",
            "bbox": {"center": [0, 0, 0], "size": [1, 2, 3]},
        }]
    }
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "layout_boxes.glb"
        exported = export_layout_3d_boxes_glb(contract, output_path)
        assert exported == str(output_path)
        assert output_path.exists()
        assert output_path.read_bytes()[:4] == b"glTF"


def validate_assembly_glb_export_without_trimesh():
    contract = {
        "parts": [{
            "part_id": "layout_test__0000__window__abc",
            "target_class": "window",
            "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
        }]
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        obj_dir = tmp_path / "parts" / "layout_test__0000__window__abc"
        obj_dir.mkdir(parents=True)
        obj_path = obj_dir / "layout_test__0000__window__abc.obj"
        obj_path.write_text(
            "\n".join([
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
                "",
            ])
        )
        output_path = tmp_path / "assembly.glb"
        exported = export_contract_part_assembly_glb(contract, tmp_path, output_path)
        assert exported == str(output_path)
        assert output_path.exists()
        assert output_path.read_bytes()[:4] == b"glTF"


def main():
    validate_command_image_repair()
    validate_reference_image_path_alias()
    validate_report_output_inference()
    validate_failure_payload_synthesizes_missing_rows()
    validate_mesh_surface_render()
    validate_render_html_falls_back_to_images_without_glb()
    validate_layout_boxes_glb_export_without_trimesh()
    validate_assembly_glb_export_without_trimesh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
