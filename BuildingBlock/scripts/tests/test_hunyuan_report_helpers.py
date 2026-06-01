#!/usr/bin/env python
"""Smoke checks for Hunyuan command and report status helpers."""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.hunyuan_adapter import build_hunyuan_command  # noqa: E402
from scene_synthesis.building_mesh.reference_images import should_preserve_full_texture  # noqa: E402
from scene_synthesis.building_mesh.report_bundle import (  # noqa: E402
    _failure_payload_for_report,
    _infer_part_output_paths,
    render_html,
)
from scene_synthesis.building_mesh.visualization import render_obj_mesh  # noqa: E402
from scene_synthesis.building_mesh.visualization import (  # noqa: E402
    export_contract_part_assembly_glb,
    export_layout_3d_boxes_glb,
    layout_vertex_to_glb_viewer,
)
from scripts.texture_archstudio_s2_hunyuan_paint import (  # noqa: E402
    detect_wall_facade_axis,
    export_box_with_generated_texture_obj,
    assemble_obj,
    export_subdivided_box_like_mesh,
    infer_part_frame_axes,
    is_success_status,
    layout_to_paint_canonical_transform,
    prepare_paint_frame_input,
    restore_paint_frame_output,
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
        "layout_s1_style_multiview": "assets/layout_s1_style_visual_critic_multiview.png",
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
    assert '<img src="assets/layout_3d_boxes.png?v=' in html_output
    assert '<h2>Layout S1-style numeric labels</h2>' in html_output
    assert 'assets/layout_s1_style_visual_critic_multiview.png?v=' in html_output
    assert 'Layout text callouts' not in html_output
    assert '<h2>Raw assembly (no wall placeholder)</h2>' in html_output
    assert '<img src="assets/assembly_raw_view_iso.png?v=' in html_output
    assert 'No interactive mesh' in html_output
    assert 'normalized mesh' not in html_output


def validate_texture_swatch_detection():
    assert should_preserve_full_texture("seamless tileable architectural material texture")
    assert should_preserve_full_texture("flat texture swatch filling the entire image")
    assert should_preserve_full_texture("single non-repeating exterior facade elevation texture")
    assert not should_preserve_full_texture("one standalone object render on white background")


def validate_wall_facade_axis_detection():
    # Smallest extent is Y, so the front/back elevation faces are the facade projection target.
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 0.1, 0], [0, 0.1, 0],
        [0, 0, 2], [1, 0, 2], [1, 0.1, 2], [0, 0.1, 2],
    ]
    assert detect_wall_facade_axis(vertices) == 1


def validate_paint_frame_alignment_for_front_and_side_facades():
    import numpy as np

    scene_center = np.asarray([0.5, 0.5, 0.5], dtype=float)
    front = {
        "bbox": {
            "center": [0.5, 0.1, 0.45],
            "size": [0.8, 0.12, 0.7],
        }
    }
    side = {
        "bbox": {
            "center": [0.1, 0.5, 0.45],
            "size": [0.12, 0.8, 0.7],
        }
    }
    front_frame = infer_part_frame_axes(front, {}, scene_center)
    side_frame = infer_part_frame_axes(side, {}, scene_center)
    assert front_frame["normal_axis_index"] == 1
    assert front_frame["horizontal_axis_index"] == 0
    assert front_frame["outward_sign"] == -1
    assert side_frame["normal_axis_index"] == 0
    assert side_frame["horizontal_axis_index"] == 1
    assert side_frame["outward_sign"] == -1

    front_transform = layout_to_paint_canonical_transform(front_frame)
    side_transform = layout_to_paint_canonical_transform(side_frame)
    assert front_transform.shape == (4, 4)
    assert side_transform.shape == (4, 4)
    # Side walls and front/back walls must not share the same paint frame:
    # image-horizontal is layout Y for side walls and layout X for front/back.
    assert not (front_transform == side_transform).all()


def validate_paint_frame_roundtrip_preserves_obj_geometry_and_materials():
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "part.obj"
        source.write_text(
            "\n".join([
                "mtllib part.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 0 1",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "",
            ])
        )
        frame = {
            "u_axis": np.asarray([1.0, 0.0, 0.0]),
            "v_axis": np.asarray([0.0, 0.0, 1.0]),
            "n_axis": np.asarray([0.0, -1.0, 0.0]),
            "normal_axis_index": 1,
            "horizontal_axis_index": 0,
            "outward_sign": -1,
            "policy": "test",
        }
        canonical = tmp_path / "canonical.obj"
        info = prepare_paint_frame_input(source, canonical, frame)
        final = tmp_path / "final.obj"
        restore_paint_frame_output(canonical, final, info)
        final_text = final.read_text()
        assert "vt 0 0" in final_text
        assert "usemtl Material" in final_text
        assert "f 1/1 2/2 3/3" in final_text
        vertices = [
            tuple(round(float(value), 6) for value in line.split()[1:4])
            for line in final_text.splitlines()
            if line.startswith("v ")
        ]
        assert vertices == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def validate_wall_subdivision_uses_current_paint_frame():
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "side_wall.obj"
        source.write_text("\n".join([
            "v 0 0 0", "v 0.1 0 0", "v 0.1 1 0", "v 0 1 0",
            "v 0 0 2", "v 0.1 0 2", "v 0.1 1 2", "v 0 1 2",
            "f 1 2 3 4", "f 5 8 7 6", "f 1 5 6 2",
            "f 2 6 7 3", "f 3 7 8 4", "f 4 8 5 1", "",
        ]))
        frame = {
            "u_axis": np.asarray([0.0, -1.0, 0.0]),
            "v_axis": np.asarray([0.0, 0.0, 1.0]),
            "n_axis": np.asarray([-1.0, 0.0, 0.0]),
            "normal_axis_index": 0,
            "horizontal_axis_index": 1,
            "outward_sign": -1,
            "policy": "test",
        }
        canonical = tmp_path / "canonical.obj"
        info = prepare_paint_frame_input(source, canonical, frame)
        subdivided = tmp_path / "canonical_subdivided.obj"
        export_subdivided_box_like_mesh(canonical, subdivided, divisions=2)
        final = tmp_path / "final.obj"
        restore_paint_frame_output(subdivided, final, info)
        vertices = [
            [float(value) for value in line.split()[1:4]]
            for line in final.read_text().splitlines()
            if line.startswith("v ")
        ]
        points = np.asarray(vertices)
        assert np.allclose(points.min(axis=0), [0.0, 0.0, 0.0])
        assert np.allclose(points.max(axis=0), [0.1, 1.0, 2.0])
        assert final.read_text().count("f ") == 48


def validate_direct_wall_facade_projection_exports_reference_texture():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_obj = tmp_path / "wall.obj"
        input_obj.write_text("\n".join([
            "v 0 0 0", "v 1 0 0", "v 1 0.1 0", "v 0 0.1 0",
            "v 0 0 2", "v 1 0 2", "v 1 0.1 2", "v 0 0.1 2",
            "f 1 2 3 4", "f 5 8 7 6", "f 1 5 6 2",
            "f 2 6 7 3", "f 3 7 8 4", "f 4 8 5 1", "",
        ]))
        from PIL import Image
        ref = tmp_path / "reference.png"
        Image.new("RGB", (64, 64), (180, 160, 130)).save(ref)
        out = tmp_path / "wall_textured.obj"
        info = export_box_with_generated_texture_obj(
            input_obj, out, ref, "wall_reference", facade_projection=True
        )
        assert info["facade_projection"] is True
        assert info["facade_axis"] == 1
        assert info["facade_box_faces"] == 2
        assert out.exists()
        assert out.with_suffix(".jpg").exists()
        assert out.read_text().count("vt ") == 36



def validate_texture_stage_manifest_and_assembly_contract():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        textured_obj = tmp_path / "part_a_textured.obj"
        textured_mtl = tmp_path / "part_a_textured.mtl"
        textured_tex = tmp_path / "part_a_textured.jpg"
        textured_obj.write_text("\n".join([
            "mtllib part_a_textured.mtl",
            "v 0 0 0", "v 1 0 0", "v 0 1 0",
            "vt 0 0", "vt 1 0", "vt 0 1",
            "usemtl Painted",
            "f 1/1 2/2 3/3", "",
        ]), encoding="utf-8")
        textured_mtl.write_text("\n".join([
            "newmtl Painted",
            "Kd 1 1 1",
            "map_Kd part_a_textured.jpg",
            "",
        ]), encoding="utf-8")
        textured_tex.write_bytes(b"fake-jpeg")

        fallback_obj = tmp_path / "part_b_fallback.obj"
        fallback_obj.write_text("\n".join([
            "v 2 0 0", "v 3 0 0", "v 2 1 0",
            "f 1 2 3", "",
        ]), encoding="utf-8")

        entries = [
            {"part_id": "part_a", "status": "succeeded", "output_obj": str(textured_obj)},
            {"part_id": "part_b", "status": "failed_untextured", "fallback_obj": str(fallback_obj)},
        ]
        assembly_obj = tmp_path / "assemblies" / "textured_or_fallback_assembly.obj"
        assemble_obj(entries, assembly_obj)
        assembly_text = assembly_obj.read_text(encoding="utf-8")
        assembly_mtl = assembly_obj.with_suffix(".mtl")
        mtl_text = assembly_mtl.read_text(encoding="utf-8")

        assert is_success_status("succeeded")
        assert is_success_status("succeeded_basic_material_dense")
        assert not is_success_status("failed_untextured")
        assert "o part_a" in assembly_text
        assert "o part_b" in assembly_text
        assert "usemtl part_b_untextured" in assembly_text
        assert "newmtl part_a_Painted" in mtl_text
        assert "newmtl part_b_untextured" in mtl_text
        copied_textures = list(assembly_obj.parent.glob("tex_*.jpg"))
        assert copied_textures, "textured success parts should copy diffuse maps into the assembly bundle"

        texture_manifest = {
            "schema_version": "archstudio_s2.texture_manifest.v1",
            "source_run": str(tmp_path / "geometry_run"),
            "texture_model": "Hunyuan3D-Paint-v2-1",
            "num_parts": len(entries),
            "num_success": sum(1 for item in entries if is_success_status(str(item["status"]))),
            "num_failures": len([item for item in entries if str(item["status"]).startswith("failed")]),
            "num_wall_skipped": 0,
            "num_basic_material": sum(1 for item in entries if str(item["status"]).startswith("succeeded_basic_material")),
            "assembly_obj": str(assembly_obj),
            "assembly_glb": "",
            "assembly_status_check_glb": "",
            "parts": entries,
        }
        assert texture_manifest["num_success"] == 1
        assert texture_manifest["num_failures"] == 1
        assert json.loads(json.dumps(texture_manifest))["schema_version"] == "archstudio_s2.texture_manifest.v1"

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


def validate_glb_viewer_axis_mapping():
    assert layout_vertex_to_glb_viewer((1, 2, 3)) == (1.0, 3.0, -2.0)


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
    validate_texture_swatch_detection()
    validate_wall_facade_axis_detection()
    validate_paint_frame_alignment_for_front_and_side_facades()
    validate_paint_frame_roundtrip_preserves_obj_geometry_and_materials()
    validate_wall_subdivision_uses_current_paint_frame()
    validate_direct_wall_facade_projection_exports_reference_texture()
    validate_texture_stage_manifest_and_assembly_contract()
    validate_layout_boxes_glb_export_without_trimesh()
    validate_glb_viewer_axis_mapping()
    validate_assembly_glb_export_without_trimesh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
