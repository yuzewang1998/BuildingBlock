#!/usr/bin/env python
"""Offline regression checks for ArchStudio-S2 V9 quality-agent helpers."""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.quality_agent import (  # noqa: E402
    QUALITY_SCHEMA_VERSION,
    bbox_contact_decision,
    build_report_asset_evidence_pack,
    build_repair_plan,
    collect_quality_metrics,
    evaluate_with_mock,
    evaluate_quality,
    expected_bbox_contacts,
    mesh_contact_gaps,
    mesh_metrics_for_part,
    parse_json_object,
    quality_findings_from_focused_visual_critics,
    build_part_geometry_contract,
    _extract_sse_chat_content,
    _validate_focused_visual_critic_answer,
    run_focused_visual_critics,
    read_obj_mesh_summary,
    validate_quality_findings,
)


def write_box_obj(path: Path, min_xyz, max_xyz) -> None:
    x0, y0, z0 = min_xyz
    x1, y1, z1 = max_xyz
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"v {x0} {y0} {z0}",
                f"v {x1} {y0} {z0}",
                f"v {x1} {y1} {z0}",
                f"v {x0} {y1} {z0}",
                f"v {x0} {y0} {z1}",
                f"v {x1} {y0} {z1}",
                f"v {x1} {y1} {z1}",
                f"v {x0} {y1} {z1}",
                "f 1 2 3 4",
                "f 5 8 7 6",
                "f 1 5 6 2",
                "f 2 6 7 3",
                "f 3 7 8 4",
                "f 4 8 5 1",
                "",
            ]
        )
    )


def test_bbox_contact_and_mesh_gap_detection():
    part_a = {"part_id": "a", "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]}}
    part_b = {"part_id": "b", "bbox": {"center": [1, 0, 0], "size": [1, 1, 1]}}
    decision = bbox_contact_decision(part_a, part_b)
    assert decision["connected"]
    assert decision["contact_type"] == "face_touch"
    contract = {"parts": [part_a, part_b]}
    contacts = expected_bbox_contacts(contract)
    assert len(contacts) == 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mesh_a = tmp_path / "a.obj"
        mesh_b = tmp_path / "b.obj"
        write_box_obj(mesh_a, [-0.5, -0.5, -0.5], [0.45, 0.5, 0.5])
        write_box_obj(mesh_b, [1.08, -0.5, -0.5], [1.5, 0.5, 0.5])
        metrics = {
            "a": mesh_metrics_for_part(part_a, mesh_a),
            "b": mesh_metrics_for_part(part_b, mesh_b),
        }
        gaps = mesh_contact_gaps(metrics, contacts, mesh_gap_tolerance=0.02)
        assert gaps[0]["above_tolerance"]
        assert gaps[0]["gap"] > 0.02


def test_mock_evaluator_schema_and_repair_plan():
    metrics = {
        "run_dir": "/tmp/run",
        "num_parts": 1,
        "parts": [{"part_id": "part_a", "index": 0, "issue_flags": []}],
        "mesh_contact_gaps": [],
    }
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "scene_score": 0.4,
        "global_summary": "mock",
        "global_style_diagnosis": "mixed",
        "issues": [
            {
                "issue_id": "q001",
                "part_ids": ["part_a"],
                "labels": [1],
                "issue_type": "texture_bad",
                "severity": 3,
                "confidence": 0.75,
                "evidence": ["partboard_000"],
                "diagnosis": "texture is inconsistent",
                "recommended_action": "rerun_texture",
                "repair_instruction": "rerun paint with shared style",
            }
        ],
    }
    findings = evaluate_with_mock(metrics, payload)
    assert findings["issues"][0]["recommended_action"] == "rerun_texture"
    plan = build_repair_plan(findings, metrics, max_parts=3)
    assert plan["num_planned_actions"] == 1
    assert plan["actions"][0]["planned_action"] == "rerun_texture"
    assert evaluate_quality(metrics, None, provider="metrics") is None
    assert evaluate_quality(metrics, None, provider="mock", mock_response=payload)["scene_score"] == 0.4


def test_schema_rejects_unknown_part_and_parse_json_object():
    text = 'Here is JSON: {"schema_version":"archstudio_s2_quality_findings.v1","scene_score":0.1,"issues":[]}'
    assert parse_json_object(text)["scene_score"] == 0.1
    bad = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "scene_score": 1.0,
        "issues": [{
            "issue_id": "bad",
            "part_ids": ["missing"],
            "labels": [99],
            "issue_type": "gap_or_seam",
            "severity": 4,
            "confidence": 0.9,
            "evidence": [],
            "diagnosis": "bad",
            "recommended_action": "deterministic_resize_snap",
            "repair_instruction": "bad",
        }],
    }
    try:
        validate_quality_findings(bad, ["known"])
    except ValueError as exc:
        assert "unknown part_ids" in str(exc)
    else:
        raise AssertionError("unknown part id should fail schema validation")



def test_openai_sse_usage_chunk_is_not_accepted_as_focused_answer():
    raw = 'data: {"id":"","object":"chat.completion.chunk","choices":[],"usage":{"total_tokens":1713}}\n\ndata: [DONE]\n'
    assert _extract_sse_chat_content(raw) == ""
    try:
        _validate_focused_visual_critic_answer({"choices": [], "object": "chat.completion.chunk"})
    except ValueError as exc:
        assert "missing required keys" in str(exc)
    else:
        raise AssertionError("usage-only streaming chunk must not be accepted as VLM answer")

    content = '{"score":0.8,"verdict":"accept","summary":"ok","issues":[]}'
    raw_with_delta = 'data: {"choices":[{"delta":{"content":' + json.dumps(content) + '}}]}\n\ndata: [DONE]\n'
    assert _extract_sse_chat_content(raw_with_delta) == content
    assert _validate_focused_visual_critic_answer(json.loads(content))["verdict"] == "accept"

def test_collect_quality_metrics_reads_minimal_run():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        part_id = "layout__000__part"
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "geometry_continuity_contract": {"target": "single_connected_bbox_contact_graph"},
            "parts": [
                {
                    "part_id": part_id,
                    "target_class": "open_semantic_part",
                    "part_description": "test part",
                    "semantic_role": "primary mass",
                    "spatial_relations": ["touches the ground"],
                    "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                }
            ],
        }
        (run_dir / "contract").mkdir()
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        obj = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        write_box_obj(obj, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        manifest = {
            "layout_id": "layout",
            "schema_version": "layout_mesh_contract.v1",
            "parts": [
                {
                    "part_id": part_id,
                    "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"),
                    "normalized_output_path": str(obj),
                    "target_class": "open_semantic_part",
                }
            ],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        metrics = collect_quality_metrics(run_dir)
        assert metrics["num_parts"] == 1
        assert metrics["num_meshes_present"] == 1
        assert metrics["geometry_continuity_contract"]["target"] == "single_connected_bbox_contact_graph"
        assert metrics["parts"][0]["semantic_role"] == "primary mass"


def test_collect_quality_metrics_accepts_texture_success_records():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        texture_dir = Path(tmp) / "texture"
        part_id = "layout__000__part"
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "parts": [
                {
                    "part_id": part_id,
                    "target_class": "open_semantic_part",
                    "part_description": "test part",
                    "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                }
            ],
        }
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        obj = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        write_box_obj(obj, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        (run_dir / "manifest.json").write_text(json.dumps({
            "layout_id": "layout",
            "parts": [{
                "part_id": part_id,
                "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"),
                "normalized_output_path": str(obj),
            }],
        }), encoding="utf-8")
        texture_dir.mkdir()
        (texture_dir / "texture_manifest.json").write_text(json.dumps({
            "parts": [{"part_id": part_id, "status": "succeeded"}]
        }), encoding="utf-8")
        metrics = collect_quality_metrics(run_dir, texture_dir)
        assert metrics["num_texture_records"] == 1
        assert metrics["summary"]["texture_failure_count"] == 0
        assert metrics["parts"][0]["texture_status"] == "succeeded"


def test_streaming_obj_summary_samples_without_component_analysis():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dense.obj"
        lines = []
        for index in range(200):
            x = index / 100.0
            lines.append(f"v {x} 0 {1.0 - x}")
        for index in range(1, 198, 3):
            lines.append(f"f {index} {index + 1} {index + 2}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary = read_obj_mesh_summary(path, max_sample_vertices=25)
        assert summary.vertex_count == 200
        assert summary.face_count == 66
        assert len(summary.sample_vertices) <= 25
        assert summary.bounds[0][0] == 0.0
        assert summary.bounds[1][0] == 1.99

        part = {"part_id": "dense", "bbox": {"center": [1.0, 0.0, 0.0], "size": [2.0, 1.0, 2.0]}}
        metrics = mesh_metrics_for_part(part, path, max_sample_vertices=25)
        assert metrics["mesh_parser_mode"] == "streaming_vertex_bounds"
        assert metrics["component_analysis"] == "skipped_fast_metrics"
        assert metrics["percentile_sample_count"] <= 25




def test_part_geometry_contract_classifies_tall_thin_wall():
    part = {
        "part_id": "wall_a",
        "part_description": "north gallery bar enclosing back U-shaped museum courtyard high wall",
        "generation_prompt": "tall rear gallery wall, thin depth, high vertical facade",
        "bbox": {"center": [0, 0, 0], "size": [4.0, 0.35, 2.8]},
    }
    contract = build_part_geometry_contract(part, label=1)
    assert contract["shape_family"] == "tall_upright_thin_wall_or_gallery_bar"
    must_be = " ".join(contract["must_be"]).lower()
    must_not = " ".join(contract["must_not_be"]).lower()
    assert "tall" in must_be and "upright" in must_be and "thin" in must_be
    assert "low wall" in must_not and "plinth" in must_not
    assert contract["bbox_size_xyz"] == [4.0, 0.35, 2.8]

def test_focused_visual_critics_convert_to_repair_findings():
    focused = {
        "schema_version": "archstudio_s2.focused_visual_critics.v1",
        "model": "gpt-5.5",
        "results": [
            {
                "case_id": "part_image_001_part_a",
                "scope": "part_image",
                "part_id": "part_a",
                "label": 1,
                "status": "completed",
                "answer": {
                    "score": 0.31,
                    "verdict": "revise",
                    "summary": "Image shows an entire museum, not one isolated part.",
                    "issues": [
                        {
                            "issue_type": "whole_building",
                            "severity": 5,
                            "evidence": "full building visible",
                            "repair_hint": "regenerate a single isolated component reference image",
                        }
                    ],
                },
            },
            {
                "case_id": "part_mesh_001_part_a",
                "scope": "part_mesh",
                "part_id": "part_a",
                "label": 1,
                "status": "completed",
                "answer": {
                    "score": 0.91,
                    "verdict": "accept",
                    "summary": "Mesh is acceptable.",
                    "issues": [],
                },
            },
        ],
    }
    findings = quality_findings_from_focused_visual_critics(
        focused,
        valid_part_ids=["part_a"],
        score_threshold=0.72,
    )
    assert findings["schema_version"] == QUALITY_SCHEMA_VERSION
    assert len(findings["issues"]) == 1
    assert findings["issues"][0]["issue_type"] == "wrong_semantics"
    assert findings["issues"][0]["recommended_action"] == "rerun_t2i_geometry_texture"
    assert findings["issues"][0]["part_ids"] == ["part_a"]



def write_minimal_report_assets(run_dir: Path, part_id: str) -> None:
    from PIL import Image

    assets = run_dir / "report" / "assets"
    parts = assets / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for name, color in [
        ("assembly_complete_view_iso.png", (80, 120, 170)),
        ("assembly_complete_view_front.png", (120, 80, 170)),
        ("assembly_complete_view_side.png", (80, 170, 120)),
        ("assembly_complete_view_top.png", (170, 120, 80)),
        ("layout_overview.png", (230, 230, 230)),
        ("layout_xz.png", (220, 230, 240)),
        ("layout_yz.png", (230, 220, 240)),
        ("layout_text_callouts_xz.png", (240, 230, 220)),
    ]:
        Image.new("RGB", (80, 60), color).save(assets / name)
    for suffix, color in [("reference", (240, 240, 240)), ("mesh", (120, 180, 120)), ("layout", (180, 120, 120))]:
        Image.new("RGB", (80, 60), color).save(parts / f"{part_id}_{suffix}.png")


def test_s2_evidence_builds_mesh_six_view_and_s1_style_layout_board():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        part_id = "layout__000__part"
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "parts": [
                {
                    "part_id": part_id,
                    "target_class": "open_semantic_part",
                    "part_description": "test part",
                    "open_vocab_label": "test_part",
                    "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                }
            ],
        }
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        obj = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        write_box_obj(obj, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        (run_dir / "manifest.json").write_text(json.dumps({
            "layout_id": "layout",
            "parts": [{"part_id": part_id, "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"), "normalized_output_path": str(obj)}],
        }), encoding="utf-8")
        write_minimal_report_assets(run_dir, part_id)
        evidence = build_report_asset_evidence_pack(run_dir, run_dir / "quality" / "v9_agent" / "iteration_000")
        assets = evidence["assets"]
        assert Path(assets["assembly_geometry_six_view_board"]).exists()
        assert Path(assets["layout_s1_style_multiview"]).exists()
        assert Path(assets["layout_s1_style_label_coverage"]).exists()
        audit = json.loads(Path(assets["layout_s1_style_label_coverage"]).read_text(encoding="utf-8"))
        assert audit["identity_contract"] == "global_numeric_pixel_visible_inline_labels"
        assert len(audit["views"]) == 6
        assert audit["views"][0]["view_prefix"] == ""


def test_focused_visual_critics_use_single_mesh_board_and_layout_board():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        part_id = "layout__000__part"
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "parts": [
                {
                    "part_id": part_id,
                    "target_class": "open_semantic_part",
                    "part_description": "test part",
                    "open_vocab_label": "test_part",
                    "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                    "reference_image_path": str(run_dir / "report" / "assets" / "parts" / f"{part_id}_reference.png"),
                }
            ],
        }
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        obj = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        write_box_obj(obj, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        (run_dir / "manifest.json").write_text(json.dumps({
            "layout_id": "layout",
            "parts": [{"part_id": part_id, "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"), "normalized_output_path": str(obj), "raw_output_path": str(obj)}],
        }), encoding="utf-8")
        write_minimal_report_assets(run_dir, part_id)
        summary = run_focused_visual_critics(
            run_dir,
            provider="mock",
            output_dir=run_dir / "agent_loop" / "focused_visual_critics",
            max_parts=1,
            scopes=["part_mesh"],
        )
        result = summary["results"][0]
        labels = [image["label"] for image in result["input_images"]]
        assert "generated_true_part_mesh_six_view_board" in labels
        assert "layout_s1_style_multiview_context" in labels
        assert not any(label.startswith("generated_true_part_mesh_front") for label in labels)



def test_part_image_critic_prefers_live_reference_over_stale_report_asset():
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        part_id = "layout__000__part"
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "parts": [{
                "part_id": part_id,
                "target_class": "open_semantic_part",
                "part_description": "test live reference part",
                "open_vocab_label": "test_part",
                "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                "reference_image_path": str(run_dir / "report" / "assets" / "parts" / f"{part_id}_reference.png"),
            }],
        }
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        obj = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        write_box_obj(obj, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        (run_dir / "manifest.json").write_text(json.dumps({
            "layout_id": "layout",
            "parts": [{"part_id": part_id, "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"), "normalized_output_path": str(obj), "raw_output_path": str(obj)}],
        }), encoding="utf-8")
        write_minimal_report_assets(run_dir, part_id)
        live_reference = run_dir / "parts" / part_id / "reference.png"
        live_reference.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), (12, 34, 56)).save(live_reference)
        summary = run_focused_visual_critics(
            run_dir,
            provider="mock",
            output_dir=run_dir / "agent_loop" / "focused_visual_critics",
            max_parts=1,
            scopes=["part_image"],
        )
        result = summary["results"][0]
        ref_image = next(image for image in result["input_images"] if image["label"] == "part_reference_image")
        assert Path(ref_image["source_path"]) == live_reference
        assert Path(ref_image["copied_path"]).read_bytes() == live_reference.read_bytes()


def test_scene_assembly_labels_and_repair_route_drive_snap():
    focused = {
        "schema_version": "archstudio_s2.focused_visual_critics.v1",
        "model": "gpt-5.5",
        "part_index": [
            {"label": 1, "part_id": "part_a", "display_label": "left mass"},
            {"label": 2, "part_id": "part_b", "display_label": "right mass"},
        ],
        "results": [
            {
                "case_id": "scene_assembly",
                "scope": "scene_assembly",
                "status": "completed",
                "answer": {
                    "score": 0.41,
                    "verdict": "revise",
                    "summary": "Front view shows a visible seam between labels 1 and 2.",
                    "issues": [
                        {
                            "issue_type": "floating_gap",
                            "labels": [1, 2],
                            "severity": 4,
                            "evidence": "front view gap between labels 1 and 2",
                            "repair_route": "deterministic_snap",
                            "repair_hint": "snap both meshes to their touching bbox faces",
                        }
                    ],
                },
                "request_path": "/tmp/request.json",
                "answer_path": "/tmp/answer.json",
                "input_images": [],
            }
        ],
    }
    findings = quality_findings_from_focused_visual_critics(
        focused,
        valid_part_ids=["part_a", "part_b"],
        score_threshold=0.72,
    )
    assert len(findings["issues"]) == 1
    issue = findings["issues"][0]
    assert issue["part_ids"] == ["part_a", "part_b"]
    assert issue["issue_type"] in {"gap_or_seam", "floating"}
    assert issue["recommended_action"] == "deterministic_resize_snap"
    assert issue["repair_route"] == "deterministic_snap"



def test_part_mesh_routes_split_i3d_from_t2i_fallback():
    focused = {
        "schema_version": "archstudio_s2.focused_visual_critics.v1",
        "model": "gpt-5.5",
        "results": [
            {
                "case_id": "part_mesh_001_part_a",
                "scope": "part_mesh",
                "part_id": "part_a",
                "status": "completed",
                "answer": {
                    "score": 0.38,
                    "verdict": "revise",
                    "summary": "Mesh is blobby and misses the reference silhouette, but the source image is usable.",
                    "issues": [
                        {
                            "issue_type": "low_geometry_quality",
                            "severity": 4,
                            "evidence": "six-view mesh board versus source reference image",
                            "repair_route": "rerun_i3d",
                            "repair_hint": "rerun Hunyuan-Omni3D with a stricter prompt preserving the existing reference image and bbox",
                        }
                    ],
                },
                "input_images": [],
            },
            {
                "case_id": "part_mesh_002_part_b",
                "scope": "part_mesh",
                "part_id": "part_b",
                "status": "completed",
                "answer": {
                    "score": 0.31,
                    "verdict": "revise",
                    "summary": "The mesh follows a whole-building reference image, so geometry rerun alone will not fix it.",
                    "issues": [
                        {
                            "issue_type": "wrong_semantics",
                            "severity": 5,
                            "evidence": "source reference image and six-view mesh both depict a whole building",
                            "repair_route": "rerun_t2i_i3d",
                            "repair_hint": "rewrite the T2I prompt for one isolated part, regenerate the image, then rerun I23D",
                        }
                    ],
                },
                "input_images": [],
            },
        ],
    }
    findings = quality_findings_from_focused_visual_critics(
        focused,
        valid_part_ids=["part_a", "part_b"],
        score_threshold=0.72,
    )
    by_part = {issue["part_ids"][0]: issue for issue in findings["issues"]}
    assert by_part["part_a"]["recommended_action"] == "rerun_geometry"
    assert by_part["part_a"]["repair_route"] == "rerun_i3d"
    assert "existing reference image" in by_part["part_a"]["repair_instruction"]
    assert by_part["part_b"]["recommended_action"] == "rerun_t2i_geometry_texture"
    assert by_part["part_b"]["repair_route"] == "rerun_t2i_i3d"
    assert "regenerate the image" in by_part["part_b"]["repair_instruction"]


def test_part_image_critic_uses_layout_context_and_t2i_route():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        part_id = "layout__000__part"
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "parts": [
                {
                    "part_id": part_id,
                    "target_class": "open_semantic_part",
                    "part_description": "thin vertical entry fin sized for a narrow bbox",
                    "open_vocab_label": "entry_fin",
                    "semantic_role": "secondary facade detail",
                    "spatial_relations": ["attached to the front facade"],
                    "bbox": {"center": [0, 0, 0], "size": [0.1, 0.03, 1.0]},
                    "reference_image_path": str(run_dir / "report" / "assets" / "parts" / f"{part_id}_reference.png"),
                }
            ],
        }
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        obj = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        write_box_obj(obj, [-0.05, -0.015, -0.5], [0.05, 0.015, 0.5])
        (run_dir / "manifest.json").write_text(json.dumps({
            "layout_id": "layout",
            "parts": [{"part_id": part_id, "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"), "normalized_output_path": str(obj), "raw_output_path": str(obj)}],
        }), encoding="utf-8")
        write_minimal_report_assets(run_dir, part_id)
        summary = run_focused_visual_critics(
            run_dir,
            provider="mock",
            output_dir=run_dir / "agent_loop" / "focused_visual_critics",
            max_parts=1,
            scopes=["part_image"],
        )
        result = summary["results"][0]
        assert result["part_id"] == part_id
        assert result["label"] == 1
        labels = [image["label"] for image in result["input_images"]]
        assert "part_reference_image" in labels
        assert "layout_s1_style_multiview_context" in labels
        assert "part_geometry_contract_bbox_proxy" not in labels
        request = json.loads(Path(result["request_path"]).read_text(encoding="utf-8"))
        prompt = request["prompt"]
        assert prompt["part_id"] == part_id
        assert prompt["label"] == 1
        assert "bbox" in prompt["question"]
        assert "part_context" in prompt
        assert prompt["part_context"]["layout_label"] == 1
        assert prompt["part_context"]["bbox_size_xyz"] == [0.1, 0.03, 1.0]
        assert "text_description" in prompt["part_context"]
        assert "geometry_contract" in prompt["part_context"]
        assert prompt["geometry_contract"]["bbox_size_xyz"] == [0.1, 0.03, 1.0]
        assert "geometry_contract_verdict" in prompt["required_json"]
        assert "i23d_prior_verdict" in prompt["required_json"]
        assert "bbox" not in prompt
        assert "mesh_metrics" not in prompt
        assert "face_count" not in json.dumps(prompt)
        assert "issue_flags" not in json.dumps(prompt)
        assert prompt["required_json"]["issues"][0]["repair_route"] == "rerun_t2i|rerun_t2i_i3d|manual_review"

    focused = {
        "schema_version": "archstudio_s2.focused_visual_critics.v1",
        "model": "gpt-5.5",
        "results": [
            {
                "case_id": "part_image_001_part_a",
                "scope": "part_image",
                "part_id": "part_a",
                "label": 1,
                "status": "completed",
                "answer": {
                    "score": 0.25,
                    "verdict": "revise",
                    "summary": "Reference is a whole building and wrong scale for the part bbox.",
                    "issues": [
                        {
                            "issue_type": "bad_aspect",
                            "severity": 4,
                            "evidence": "whole building fills image instead of isolated narrow part",
                            "repair_route": "rerun_t2i_i3d",
                            "repair_hint": "rewrite prompt as one isolated narrow part on white background before rerunning I23D",
                        }
                    ],
                },
            }
        ],
    }
    findings = quality_findings_from_focused_visual_critics(focused, valid_part_ids=["part_a"], score_threshold=0.72)
    issue = findings["issues"][0]
    assert issue["recommended_action"] == "rerun_t2i_geometry_texture"
    assert issue["repair_route"] == "rerun_t2i_i3d"
    assert "isolated narrow part" in issue["repair_instruction"]



def test_part_image_geometry_contract_subverdict_blocks_semantic_accept():
    focused = {
        "schema_version": "archstudio_s2.focused_visual_critics.v1",
        "model": "gpt-5.5",
        "results": [{
            "case_id": "part_image_001_part_a",
            "scope": "part_image",
            "part_id": "part_a",
            "label": 1,
            "status": "completed",
            "answer": {
                "score": 0.86,
                "verdict": "accept",
                "semantic_verdict": "accept",
                "geometry_contract_verdict": "revise",
                "i23d_prior_verdict": "revise",
                "final_verdict": "revise",
                "summary": "Semantically a wall, but the image is a low curb/plinth and cannot serve as a tall thin wall prior.",
                "issues": [],
            },
        }],
    }
    findings = quality_findings_from_focused_visual_critics(focused, valid_part_ids=["part_a"], score_threshold=0.72)
    assert len(findings["issues"]) == 1
    issue = findings["issues"][0]
    assert issue["issue_type"] == "scale_mismatch"
    assert issue["recommended_action"] == "rerun_t2i_geometry_texture"
    assert issue["repair_route"] == "rerun_t2i_i3d"


def test_part_mesh_source_image_root_cause_routes_back_to_t2i():
    focused = {
        "schema_version": "archstudio_s2.focused_visual_critics.v1",
        "model": "gpt-5.5",
        "results": [{
            "case_id": "part_mesh_001_part_a",
            "scope": "part_mesh",
            "part_id": "part_a",
            "label": 1,
            "status": "completed",
            "answer": {
                "score": 0.34,
                "verdict": "revise",
                "root_cause": "source_image",
                "summary": "The mesh is low because the source T2I image is already a low wall, not the tall bbox wall.",
                "issues": [{
                    "issue_type": "low_geometry_quality",
                    "severity": 4,
                    "root_cause": "source_image",
                    "evidence": "source image violates geometry contract",
                    "repair_hint": "regenerate T2I as tall upright thin wall before rerunning I23D",
                }],
            },
        }],
    }
    findings = quality_findings_from_focused_visual_critics(focused, valid_part_ids=["part_a"], score_threshold=0.72)
    issue = findings["issues"][0]
    assert issue["recommended_action"] == "rerun_t2i_geometry_texture"
    assert issue["repair_route"] == "rerun_t2i_i3d"
    assert issue["issue_type"] in {"wrong_semantics", "scale_mismatch"}

def main() -> int:
    test_bbox_contact_and_mesh_gap_detection()
    test_mock_evaluator_schema_and_repair_plan()
    test_schema_rejects_unknown_part_and_parse_json_object()
    test_openai_sse_usage_chunk_is_not_accepted_as_focused_answer()
    test_collect_quality_metrics_reads_minimal_run()
    test_collect_quality_metrics_accepts_texture_success_records()
    test_streaming_obj_summary_samples_without_component_analysis()
    test_focused_visual_critics_convert_to_repair_findings()
    test_part_geometry_contract_classifies_tall_thin_wall()
    test_part_image_critic_uses_layout_context_and_t2i_route()
    test_part_image_critic_prefers_live_reference_over_stale_report_asset()
    test_scene_assembly_labels_and_repair_route_drive_snap()
    test_part_mesh_routes_split_i3d_from_t2i_fallback()
    test_part_image_geometry_contract_subverdict_blocks_semantic_accept()
    test_part_mesh_source_image_root_cause_routes_back_to_t2i()
    test_s2_evidence_builds_mesh_six_view_and_s1_style_layout_board()
    test_focused_visual_critics_use_single_mesh_board_and_layout_board()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
