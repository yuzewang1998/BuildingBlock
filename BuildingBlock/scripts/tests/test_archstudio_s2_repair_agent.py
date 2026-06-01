#!/usr/bin/env python
"""Offline regression checks for the ArchStudio-S2 reject/accept repair agent."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.report_bundle import refresh_s2_9999_report  # noqa: E402
from scene_synthesis.building_mesh.quality_agent import quality_findings_from_focused_visual_critics  # noqa: E402
from scene_synthesis.building_mesh.s2_repair_agent import (  # noqa: E402
    S2AgentConfig,
    S2Repairer,
    _append_or_replace_cli_option,
    _eligible_t2i_gpus,
    _t2i_command,
    apply_snap_resize_operation,
    apply_split_operation,
    build_prompt_revision_for_part,
    quality_action_to_s2_operation,
    repair_operations_from_findings,
    run_geometry_for_parts,
    run_t2i_for_parts,
    s2_findings_from_quality_findings,
    validate_s2_run,
    write_prompt_revisions_for_operation,
)




def write_dummy_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6360000002000150a2f5dc0000000049454e44ae426082"
        )
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
        ),
        encoding="utf-8",
    )


def write_minimal_run(run_dir: Path, *, reference_status: str = "succeeded") -> dict:
    part_a = "layout__0000__mass_a"
    part_b = "layout__0001__mass_b"
    contract = {
        "schema_version": "layout_mesh_contract.v1",
        "layout_id": "layout",
        "parts": [
            {
                "schema_version": "layout_mesh_contract.v1",
                "layout_id": "layout",
                "part_id": part_a,
                "target_class": "open_semantic_part",
                "source_actor_label": "single test architectural mass A",
                "part_description": "single test architectural mass A",
                "target_prompt": "3D mesh of single test architectural mass A",
                "part_prompt": "single isolated standalone architectural component asset A",
                "negative_prompt": "whole building",
                "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                "actor_location": [0, 0, 0],
                "actor_size": [1, 1, 1],
                "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"),
                "reference_image_path": str(run_dir / "parts" / part_a / "reference.png"),
                "reference_image": str(run_dir / "parts" / part_a / "reference.png"),
            },
            {
                "schema_version": "layout_mesh_contract.v1",
                "layout_id": "layout",
                "part_id": part_b,
                "target_class": "open_semantic_part",
                "source_actor_label": "single test architectural mass B",
                "part_description": "single test architectural mass B",
                "target_prompt": "3D mesh of single test architectural mass B",
                "part_prompt": "single isolated standalone architectural component asset B",
                "negative_prompt": "whole building",
                "bbox": {"center": [1, 0, 0], "size": [1, 1, 1]},
                "actor_location": [1, 0, 0],
                "actor_size": [1, 1, 1],
                "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"),
                "reference_image_path": str(run_dir / "parts" / part_b / "reference.png"),
                "reference_image": str(run_dir / "parts" / part_b / "reference.png"),
            },
        ],
    }
    (run_dir / "contract").mkdir(parents=True)
    (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    for part in contract["parts"]:
        image_path = Path(part["reference_image_path"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"not a real png, existence is enough for deterministic validation")
        (image_path.parent / "t2i_metadata.json").write_text(
            json.dumps({
                "part_id": part["part_id"],
                "status": reference_status,
                "provider": "qwen_image_local",
                "reference_image_path": str(image_path),
            }),
            encoding="utf-8",
        )
    obj_a = run_dir / "assemblies" / "parts" / f"{part_a}.obj"
    obj_b = run_dir / "assemblies" / "parts" / f"{part_b}.obj"
    write_box_obj(obj_a, [-0.5, -0.5, -0.5], [0.42, 0.5, 0.5])
    write_box_obj(obj_b, [1.12, -0.5, -0.5], [1.5, 0.5, 0.5])
    manifest = {
        "layout_id": "layout",
        "schema_version": "layout_mesh_contract.v1",
        "raw_assembly_path": str(run_dir / "assemblies" / "raw_hunyuan_assembly.obj"),
        "placeholder_assembly_path": str(run_dir / "assemblies" / "placeholder_filled_assembly.obj"),
        "parts": [
            {
                "layout_id": "layout",
                "schema_version": "layout_mesh_contract.v1",
                "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"),
                "part_id": part_a,
                "target_class": "open_semantic_part",
                "attempts": [],
                "raw_output_path": str(obj_a),
                "normalized_output_path": str(obj_a),
                "placeholder_output_path": str(obj_a),
            },
            {
                "layout_id": "layout",
                "schema_version": "layout_mesh_contract.v1",
                "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"),
                "part_id": part_b,
                "target_class": "open_semantic_part",
                "attempts": [],
                "raw_output_path": str(obj_b),
                "normalized_output_path": str(obj_b),
                "placeholder_output_path": str(obj_b),
            },
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "failures.json").write_text(json.dumps({"failures": []}), encoding="utf-8")
    return {"part_a": part_a, "part_b": part_b}




def test_focused_accept_with_minor_advisory_issue_does_not_block():
    focused = {
        "schema_version": "archstudio_s2.focused_visual_critics.v1",
        "part_index": [{"label": 1, "part_id": "part_a"}],
        "results": [{
            "case_id": "part_image_001",
            "scope": "part_image",
            "part_id": "part_a",
            "label": 1,
            "status": "completed",
            "answer": {
                "score": 0.82,
                "verdict": "accept",
                "summary": "acceptable with minor base detail",
                "issues": [{"issue_type": "background_or_base_artifact", "severity": 1, "evidence": "minor", "repair_hint": "optional"}],
            },
        }],
    }
    findings = quality_findings_from_focused_visual_critics(focused, valid_part_ids=["part_a"], score_threshold=0.72)
    assert findings["issues"] == []


def test_t2i_rerun_seed_changes_by_iteration():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_minimal_run(run_dir)
        config = S2AgentConfig(run_dir=run_dir, dry_run=True, t2i_seed=1234, t2i_rerun_seed_stride=1009, t2i_python=Path(sys.executable))
        first = run_t2i_for_parts(config, ["layout__0000__mass_a"], iteration=0)
        second = run_t2i_for_parts(config, ["layout__0000__mass_a"], iteration=3)
        assert first["t2i_seed"] == 1234
        assert second["t2i_seed"] == 4261
        assert first["command"] != second["command"]


def test_append_or_replace_cli_option_updates_seed_inside_bash_lc():
    command = 'bash -lc "CUDA_VISIBLE_DEVICES=6 python hunyuan.py --image $reference_image --seed 1234"'
    updated = _append_or_replace_cli_option(command, "--seed", 2243)
    assert '--seed 2243"' in updated
    assert updated.count("--seed") == 1

    command_without_seed = 'bash -lc "CUDA_VISIBLE_DEVICES=6 python hunyuan.py --image $reference_image"'
    appended = _append_or_replace_cli_option(command_without_seed, "--seed", 2243)
    assert appended.endswith('--seed 2243"')


def test_geometry_rerun_marks_byte_identical_mesh_as_unchanged():
    import scene_synthesis.building_mesh.s2_repair_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)
        part_id = ids["part_a"]
        previous_raw = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        previous_bytes = previous_raw.read_bytes()

        class FakeAttempt:
            def to_dict(self):
                return {"attempt_number": 1, "failure_type": None, "exit_code": 0}

        class FakeResult:
            def __init__(self, raw_output_path):
                self.layout_id = "layout"
                self.schema_version = "layout_mesh_contract.v1"
                self.contract_path = str(run_dir / "contract" / "layout_mesh_contract.json")
                self.part_id = part_id
                self.source_actor_label = None
                self.target_class = "open_semantic_part"
                self.attempts = [FakeAttempt()]
                self.raw_output_path = str(raw_output_path)
                self.lifecycle_states = []
                self.final_failure_type = None

            def to_dict(self):
                return {
                    "layout_id": self.layout_id,
                    "schema_version": self.schema_version,
                    "contract_path": self.contract_path,
                    "part_id": self.part_id,
                    "attempts": [attempt.to_dict() for attempt in self.attempts],
                    "raw_output_path": self.raw_output_path,
                    "lifecycle_states": [],
                }

        class FakeAdapter:
            def __init__(self, command_template, output_root, policy):
                self.command_template = command_template
                self.output_root = Path(output_root)

            def run_part(self, part):
                out_dir = self.output_root / part_id
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{part_id}.obj"
                out_path.write_bytes(previous_bytes)
                return FakeResult(out_path)

        old_adapter = agent.HunyuanAdapter
        try:
            agent.HunyuanAdapter = FakeAdapter
            config = S2AgentConfig(
                run_dir=run_dir,
                hunyuan_command='bash -lc "CUDA_VISIBLE_DEVICES=6 python hunyuan.py --image $reference_image --save-dir $output_dir --file-name $part_id --seed 1234"',
                hunyuan_gpus=["6"],
                hunyuan_seed=1234,
            )
            result = run_geometry_for_parts(config, [part_id], iteration=2)
        finally:
            agent.HunyuanAdapter = old_adapter

        assert result["status"] == "partial_or_failed"
        assert result["results"][0]["status"] == "unchanged_output"
        assert result["results"][0]["previous_raw_sha256"] == result["results"][0]["new_raw_sha256"]
        assert "--seed 4261" in result["results"][0]["command_template"]
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        entry = next(part for part in manifest["parts"] if part["part_id"] == part_id)
        assert "agent_geometry_rerun_unchanged" in entry.get("lifecycle_states", [])


def test_validation_rejects_gap_and_plans_snap():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)
        report = validate_s2_run(run_dir, acceptance_score=0.99, reject_severity=3)
        assert not report.accepted
        assert any(finding.issue_type == "gap_or_seam" for finding in report.findings)
        operations = repair_operations_from_findings(report.findings, enable_t2i=False, enable_geometry=False, enable_split=False)
        assert any(operation.operation_type == "deterministic_resize_snap" for operation in operations)
        result = apply_snap_resize_operation(run_dir, [ids["part_a"], ids["part_b"]], iteration=0)
        assert result["rebuild"]["complete_vertex_count"] > 0
        report_after = validate_s2_run(run_dir, acceptance_score=0.50, reject_severity=5, enable_split=False)
        assert report_after.metrics["summary"]["mesh_gap_count"] == 0


def test_split_operation_replaces_contract_part_with_surface_children():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)
        split = apply_split_operation(run_dir, [ids["part_a"]], iteration=0)
        children = split["replaced"][ids["part_a"]]
        assert len(children) >= 2
        contract = json.loads((run_dir / "contract" / "layout_mesh_contract.json").read_text(encoding="utf-8"))
        part_ids = {part["part_id"] for part in contract["parts"]}
        assert ids["part_a"] not in part_ids
        assert set(children).issubset(part_ids)
        assert all("agent_split_parent_part_id" in part for part in contract["parts"] if part["part_id"] in children)


def test_repairer_writes_reject_accept_trace_in_dry_run():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_minimal_run(run_dir, reference_status="failed")
        config = S2AgentConfig(
            run_dir=run_dir,
            max_iterations=1,
            dry_run=True,
            force_report=False,
            enable_geometry=False,
            enable_t2i=True,
            enable_snap=True,
            enable_split=False,
            t2i_python=Path(sys.executable),
        )
        result = S2Repairer(config).run()
        assert Path(result.trace_path).exists()
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        assert trace["status"] == "complete"
        assert trace["iterations"]
        assert trace["iterations"][0]["operations"]
        assert (run_dir / "agent_loop" / "index.html").exists()


def test_vlm_findings_are_converted_and_drive_operations():
    quality_findings = {
        "schema_version": "archstudio_s2_quality_findings.v1",
        "scene_score": 0.2,
        "global_summary": "mock VLM found semantic failure",
        "global_style_diagnosis": "mixed",
        "issues": [
            {
                "issue_id": "v001",
                "part_ids": ["part_a"],
                "labels": [1],
                "issue_type": "wrong_semantics",
                "severity": 5,
                "confidence": 0.91,
                "evidence": ["partboard_000"],
                "diagnosis": "image/mesh is a whole building, not a single part",
                "recommended_action": "rerun_t2i_geometry_texture",
                "repair_instruction": "tighten prompt to single isolated part and rerun geometry",
            }
        ],
    }
    converted = s2_findings_from_quality_findings(quality_findings, iteration=2)
    assert converted[0].source == "vlm_gpt55_quality_agent"
    assert converted[0].recommended_operation == "rerun_t2i_geometry"
    operations = repair_operations_from_findings(converted)
    assert operations[0].operation_type == "rerun_t2i_geometry"
    assert quality_action_to_s2_operation("rerun_t2i_geometry_texture", "wrong_semantics") == "rerun_t2i_geometry"


def test_repairer_uses_mock_vlm_evaluator_in_dry_run():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)
        mock_findings = {
            "schema_version": "archstudio_s2_quality_findings.v1",
            "scene_score": 0.25,
            "global_summary": "mock VLM",
            "global_style_diagnosis": "mock",
            "issues": [
                {
                    "issue_id": "v001",
                    "part_ids": [ids["part_a"]],
                    "labels": [1],
                    "issue_type": "wrong_semantics",
                    "severity": 5,
                    "confidence": 0.9,
                    "evidence": ["partboard_000"],
                    "diagnosis": "not a single part",
                    "recommended_action": "rerun_t2i_geometry_texture",
                    "repair_instruction": "rerun from a part-only reference image",
                }
            ],
        }
        config = S2AgentConfig(
            run_dir=run_dir,
            max_iterations=1,
            dry_run=True,
            force_report=False,
            enable_geometry=True,
            enable_t2i=True,
            enable_snap=True,
            enable_split=False,
            t2i_python=Path(sys.executable),
            evaluator_provider="mock",
            evaluator_mock_response=mock_findings,
        )
        result = S2Repairer(config).run()
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        assert trace["iterations"][0]["evaluator"]["status"] == "completed"
        assert trace["iterations"][0]["evaluator"]["issue_count"] == 1
        op_types = [op["operation_type"] for op in trace["iterations"][0]["operations"]]
        assert "rerun_t2i_geometry" in op_types
        assert trace["iterations"][0]["stage_records"]
        assert any(stage["stage"] == "prompt_repair" for stage in trace["iterations"][0]["stage_records"])
        t2i_stage = next(stage for stage in trace["iterations"][0]["stage_records"] if stage["stage"] == "t2i_generation")
        assert "before_artifacts" in t2i_stage["outputs"]
        assert "after_artifacts" in t2i_stage["outputs"]
        assert "prompt_diffs" in t2i_stage["outputs"]
        assert trace["score_trajectory"]
        assert "t2i" in trace["failure_taxonomy"]


def test_repairer_uses_focused_vlm_findings_in_dry_run(monkeypatch=None):
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)

        import scene_synthesis.building_mesh.s2_repair_agent as repair_agent

        original = repair_agent.run_focused_visual_critics

        def fake_focused(*_args, **_kwargs):
            return {
                "schema_version": "archstudio_s2.focused_visual_critics.v1",
                "model": "gpt-5.5",
                "results": [
                    {
                        "case_id": "part_image_001",
                        "scope": "part_image",
                        "part_id": ids["part_a"],
                        "label": 1,
                        "status": "completed",
                        "answer": {
                            "score": 0.22,
                            "verdict": "revise",
                            "summary": "Reference is a full building, not a single part.",
                            "issues": [{"issue_type": "whole_building", "severity": 5, "repair_hint": "single isolated part"}],
                        },
                        "request_path": str(run_dir / "fake_request.json"),
                        "answer_path": str(run_dir / "fake_answer.json"),
                        "input_images": [],
                    }
                ],
            }

        try:
            repair_agent.run_focused_visual_critics = fake_focused
            config = S2AgentConfig(
                run_dir=run_dir,
                max_iterations=1,
                dry_run=True,
                force_report=False,
                enable_geometry=True,
                enable_t2i=True,
                enable_snap=False,
                enable_split=False,
                t2i_python=Path(sys.executable),
                evaluator_provider="gpt55",
                evaluator_mode="focused",
                focused_critic_max_parts=1,
            )
            result = S2Repairer(config).run()
        finally:
            repair_agent.run_focused_visual_critics = original

        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        evaluator = trace["iterations"][0]["evaluator"]
        assert evaluator["mode"] == "focused"
        assert evaluator["status"] == "completed"
        assert evaluator["issue_count"] == 1
        assert trace["iterations"][0]["operations"]
        assert trace["iterations"][0]["operations"][0]["operation_type"] == "rerun_t2i_geometry"


def test_assemble_geometry_scope_prechecks_parts_before_scene_assembly():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)

        import scene_synthesis.building_mesh.s2_repair_agent as repair_agent

        original = repair_agent.run_focused_visual_critics
        calls = []

        def fake_focused(*_args, **kwargs):
            calls.append({"scopes": kwargs.get("scopes"), "part_ids": kwargs.get("part_ids")})
            assert kwargs.get("scopes") == ["part_image"]
            assert kwargs.get("part_ids") == [ids["part_a"]]
            return {
                "schema_version": "archstudio_s2.focused_visual_critics.v1",
                "model": "gpt-5.5",
                "scopes": ["part_image"],
                "part_index": [
                    {"label": 1, "part_id": ids["part_a"], "display_label": "mass A"},
                    {"label": 2, "part_id": ids["part_b"], "display_label": "mass B"},
                ],
                "results": [
                    {
                        "case_id": f"part_image_{len(calls):03d}",
                        "scope": "part_image",
                        "part_id": ids["part_a"],
                        "label": 1,
                        "status": "completed",
                        "answer": {
                            "score": 0.4,
                            "verdict": "revise",
                            "summary": "Part A image is a whole building, not an isolated component.",
                            "issues": [{"issue_type": "whole_building", "severity": 4, "repair_route": "rerun_t2i", "repair_hint": "rerun T2I for part A"}],
                        },
                        "request_path": str(run_dir / "fake_part_request.json"),
                        "answer_path": str(run_dir / "fake_part_answer.json"),
                        "input_images": [{"label": "part_reference_image", "source_path": str(run_dir / "fake.png"), "copied_path": str(run_dir / "fake.png")}],
                    }
                ],
            }

        try:
            repair_agent.run_focused_visual_critics = fake_focused
            config = S2AgentConfig(
                run_dir=run_dir,
                max_iterations=2,
                dry_run=True,
                force_report=False,
                enable_geometry=True,
                enable_t2i=True,
                enable_snap=True,
                enable_split=False,
                t2i_python=Path(sys.executable),
                evaluator_provider="gpt55",
                evaluator_mode="focused",
                agent_scope="assemble_geometry",
            )
            result = S2Repairer(config).run()
        finally:
            repair_agent.run_focused_visual_critics = original

        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        assert trace["policy"]["agent_scope"] == "assemble_geometry"
        assert trace["policy"]["sequential_part_gate"] is True
        assert calls[:2] == [
            {"scopes": ["part_image"], "part_ids": [ids["part_a"]]},
            {"scopes": ["part_image"], "part_ids": [ids["part_a"]]},
        ], calls
        assert [it["evaluator"]["current_part_id"] for it in trace["iterations"][:2]] == [ids["part_a"], ids["part_a"]]
        assert [it["evaluator"]["current_part_stage"] for it in trace["iterations"][:2]] == ["part_image", "part_image"]
        operation_types = [op["operation_type"] for iteration in trace["iterations"] for op in iteration["operations"]]
        assert "rerun_t2i_geometry" in operation_types
        assert "deterministic_resize_snap" not in operation_types
        assert trace["workflow_transitions"][0]["reason"] == "current_part_stage_findings_remain"
        assert trace["workflow_transitions"][0]["current_part_stage"] == "part_image"
        assert any(stage["inputs"].get("current_part_id") == ids["part_a"] for stage in trace["iterations"][0]["stage_records"] if isinstance(stage.get("inputs"), dict))


def test_assemble_geometry_scope_advances_part_image_to_mesh_then_next_part():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)

        import scene_synthesis.building_mesh.s2_repair_agent as repair_agent

        original = repair_agent.run_focused_visual_critics
        calls = []

        def accepted_result(scope, part_id, label):
            return {
                "schema_version": "archstudio_s2.focused_visual_critics.v1",
                "model": "gpt-5.5",
                "scopes": [scope],
                "part_index": [
                    {"label": 1, "part_id": ids["part_a"], "display_label": "mass A"},
                    {"label": 2, "part_id": ids["part_b"], "display_label": "mass B"},
                ],
                "results": [{
                    "case_id": f"{scope}_{label:03d}",
                    "scope": scope,
                    "part_id": part_id,
                    "label": label,
                    "status": "completed",
                    "answer": {"score": 0.95, "verdict": "accept", "summary": "ok", "issues": []},
                    "request_path": str(run_dir / f"{scope}_{label}.json"),
                    "answer_path": str(run_dir / f"{scope}_{label}_answer.json"),
                    "input_images": [],
                }],
            }

        def fake_focused(*_args, **kwargs):
            calls.append({"scopes": kwargs.get("scopes"), "part_ids": kwargs.get("part_ids")})
            scope = kwargs.get("scopes")[0]
            part_id = kwargs.get("part_ids")[0]
            label = 1 if part_id == ids["part_a"] else 2
            return accepted_result(scope, part_id, label)

        try:
            repair_agent.run_focused_visual_critics = fake_focused
            config = S2AgentConfig(
                run_dir=run_dir,
                max_iterations=3,
                dry_run=True,
                force_report=False,
                enable_geometry=True,
                enable_t2i=True,
                enable_snap=True,
                enable_split=False,
                t2i_python=Path(sys.executable),
                evaluator_provider="gpt55",
                evaluator_mode="focused",
                agent_scope="assemble_geometry",
            )
            result = S2Repairer(config).run()
        finally:
            repair_agent.run_focused_visual_critics = original

        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        assert calls[:4] == [
            {"scopes": ["part_image"], "part_ids": [ids["part_a"]]},
            {"scopes": ["part_mesh"], "part_ids": [ids["part_a"]]},
            {"scopes": ["part_image"], "part_ids": [ids["part_b"]]},
            {"scopes": ["part_mesh"], "part_ids": [ids["part_b"]]},
        ]
        assert trace["iterations"][0]["workflow_transition"]["reason"] == "current_part_image_accepted_advance_to_mesh"
        assert trace["iterations"][1]["workflow_transition"]["reason"] == "current_part_mesh_accepted_advance_to_next_part"
        assert trace["iterations"][2]["evaluator"]["current_part_id"] == ids["part_b"]
        assert trace["iterations"][2]["evaluator"]["current_part_stage"] == "part_image"


def test_sequential_max_iterations_resets_after_stage_acceptance():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)

        import scene_synthesis.building_mesh.s2_repair_agent as repair_agent

        original = repair_agent.run_focused_visual_critics
        calls = []

        def fake_focused(*_args, **kwargs):
            scopes = kwargs.get("scopes") or ["scene_assembly"]
            part_ids = kwargs.get("part_ids") or []
            scope = scopes[0]
            part_id = part_ids[0] if part_ids else None
            calls.append({"scopes": scopes, "part_ids": part_ids})
            label = 1 if part_id == ids["part_a"] else 2 if part_id == ids["part_b"] else None
            result = {
                "case_id": f"{scope}_{len(calls):03d}",
                "scope": scope,
                "part_id": part_id,
                "label": label,
                "status": "completed",
                "answer": {"score": 0.96, "verdict": "accept", "summary": "ok", "issues": []},
                "request_path": str(run_dir / f"request_{len(calls)}.json"),
                "answer_path": str(run_dir / f"answer_{len(calls)}.json"),
                "input_images": [],
            }
            Path(result["request_path"]).write_text(json.dumps({"prompt": {"question": "q"}}), encoding="utf-8")
            Path(result["answer_path"]).write_text(json.dumps({"answer": result["answer"]}), encoding="utf-8")
            return {
                "schema_version": "archstudio_s2.focused_visual_critics.v1",
                "model": "mock",
                "scopes": scopes,
                "part_index": [
                    {"label": 1, "part_id": ids["part_a"], "display_label": "mass A"},
                    {"label": 2, "part_id": ids["part_b"], "display_label": "mass B"},
                ],
                "results": [result],
            }

        try:
            repair_agent.run_focused_visual_critics = fake_focused
            config = S2AgentConfig(
                run_dir=run_dir,
                max_iterations=1,
                max_no_improvement_iterations=99,
                dry_run=True,
                force_report=False,
                enable_geometry=True,
                enable_t2i=True,
                enable_snap=False,
                enable_split=False,
                t2i_python=Path(sys.executable),
                evaluator_provider="gpt55",
                evaluator_mode="focused",
                agent_scope="assemble_geometry",
            )
            result = S2Repairer(config).run()
        finally:
            repair_agent.run_focused_visual_critics = original

        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        assert trace["policy"]["max_iterations_semantics"] == "per_part_stage_retry_budget"
        assert calls[:4] == [
            {"scopes": ["part_image"], "part_ids": [ids["part_a"]]},
            {"scopes": ["part_mesh"], "part_ids": [ids["part_a"]]},
            {"scopes": ["part_image"], "part_ids": [ids["part_b"]]},
            {"scopes": ["part_mesh"], "part_ids": [ids["part_b"]]},
        ]
        assert len(trace["iterations"]) > 2
        assert trace["stop_condition"] != "max_iterations_exhausted"


def test_prompt_revision_persists_vlm_repair_instruction():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        ids = write_minimal_run(run_dir)
        finding = s2_findings_from_quality_findings(
            {
                "schema_version": "archstudio_s2_quality_findings.v1",
                "scene_score": 0.2,
                "issues": [
                    {
                        "issue_id": "v001",
                        "part_ids": [ids["part_a"]],
                        "labels": [1],
                        "issue_type": "wrong_semantics",
                        "severity": 5,
                        "confidence": 0.9,
                        "evidence": ["partboard_000"],
                        "diagnosis": "reference is a whole building",
                        "recommended_action": "rerun_t2i_geometry_texture",
                        "repair_instruction": "make it a single isolated gallery bar component, not a whole courtyard",
                    }
                ],
            },
            iteration=0,
        )[0]
        operation = repair_operations_from_findings([finding])[0]
        contract = json.loads((run_dir / "contract" / "layout_mesh_contract.json").read_text(encoding="utf-8"))
        part = next(part for part in contract["parts"] if part["part_id"] == ids["part_a"])
        revision = build_prompt_revision_for_part(run_dir, part, iteration=0, operation=operation)
        assert "single isolated gallery bar component" in revision["prompt"]
        assert "whole building" in revision["negative_prompt"]
        result = write_prompt_revisions_for_operation(run_dir, operation, iteration=0)
        path = Path(result["revisions"][0]["prompt_revision_path"])
        assert path.exists()
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert persisted["revision_attempt"] == 1
        assert persisted["failure_stage"] == "t2i"


def test_repairer_auto_stops_on_no_improvement():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_minimal_run(run_dir, reference_status="failed")
        config = S2AgentConfig(
            run_dir=run_dir,
            max_iterations=3,
            max_no_improvement_iterations=1,
            dry_run=True,
            force_report=False,
            enable_geometry=True,
            enable_t2i=True,
            enable_snap=False,
            enable_split=False,
            t2i_python=Path(sys.executable),
        )
        result = S2Repairer(config).run()
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        assert trace["stop_condition"] in {"no_improvement", "all_operations_failed"}
        assert trace["score_trajectory"]


def test_9999_refresh_hook_updates_main_report_with_agent_io():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "summary.json").write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
        (report_dir / "index.html").write_text(
            "<html><body>old"
            "<!-- ARCHSTUDIO_S2_REPAIR_AGENT_START --><!-- ARCHSTUDIO_S2_REPAIR_AGENT_END -->"
            "<!-- ARCHSTUDIO_S2_QUALITY_AGENT_START --><!-- ARCHSTUDIO_S2_QUALITY_AGENT_END -->"
            "</body></html>",
            encoding="utf-8",
        )
        agent_dir = run_dir / "agent_loop"
        image_dir = agent_dir / "focused_visual_critics" / "scene_assembly" / "input_images"
        image_dir.mkdir(parents=True)
        image_path = image_dir / "01_true_colored_assembly_mesh_six_view_board.png"
        # A tiny but valid PNG keeps the report display helper on the normal path.
        try:
            from PIL import Image

            Image.new("RGB", (8, 8), (240, 240, 240)).save(image_path)
        except Exception:
            image_path.write_bytes(b"not-a-real-image")
        question_path = agent_dir / "question_prompt.txt"
        question_path.write_text("Only judge the six-view mesh board", encoding="utf-8")
        answer_path = agent_dir / "answer.json"
        answer_path.write_text(json.dumps({"findings": {"issues": []}}), encoding="utf-8")
        trace_path = agent_dir / "agent_trace.json"
        trace = {
            "schema_version": "archstudio_s2_repair_agent_loop.v1",
            "status": "complete",
            "accepted": False,
            "stop_condition": "max_iterations_exhausted",
            "final_scene_score": 0.33,
            "score_trajectory": [{"iteration": 0, "scene_score": 0.33, "delta": None, "no_improvement_count": 0}],
            "failure_taxonomy": {"assembly_validation": 1},
            "iterations": [
                {
                    "iteration": 0,
                    "accepted": False,
                    "scene_score": 0.33,
                    "num_findings": 0,
                    "evaluator": {
                        "provider": "mock",
                        "model": "gpt-5.5",
                        "mode": "focused",
                        "status": "completed",
                        "issue_count": 0,
                        "question_prompt_path": str(question_path),
                        "answer_path": str(answer_path),
                        "input_images": [
                            {
                                "index": 1,
                                "label": "true_colored_assembly_mesh_six_view_board",
                                "source_path": str(image_path),
                                "copied_path": str(image_path),
                            }
                        ],
                    },
                    "operations": [],
                    "stage_records": [
                        {
                            "stage": "assembly_validation",
                            "status": "completed",
                            "part_ids": [],
                            "inputs": {"vlm_input_image_count": 1},
                            "outputs": {"vlm_answer_path": str(answer_path)},
                        }
                    ],
                }
            ],
        }
        trace_path.write_text(json.dumps(trace), encoding="utf-8")

        refreshed = refresh_s2_9999_report(run_dir, full=False, trigger="test_experiment_done")
        html_text = (report_dir / "index.html").read_text(encoding="utf-8")
        assert Path(refreshed["refresh_hook_path"]).exists()
        assert "ArchStudio-S2 Reject/Accept Repair Agent" in html_text
        assert "Agent VLM inputs / outputs / repair timeline" in html_text
        assert "true_colored_assembly_mesh_six_view_board" in html_text
        assert "mesh 六视角合成图" in html_text
        assert "9999 refresh hook" in html_text
        assert "test_experiment_done" in html_text
        assert "Agent scope" in html_text
        assert "compact debug workflow status" in html_text
        assert "compact debug view; collapsed by default" in html_text


def test_9999_refresh_shows_all_focused_part_precheck_cases():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "summary.json").write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
        (report_dir / "index.html").write_text(
            "<html><head></head><body>old"
            "<!-- ARCHSTUDIO_S2_REPAIR_AGENT_START --><!-- ARCHSTUDIO_S2_REPAIR_AGENT_END -->"
            "<!-- ARCHSTUDIO_S2_QUALITY_AGENT_START --><!-- ARCHSTUDIO_S2_QUALITY_AGENT_END -->"
            "</body></html>",
            encoding="utf-8",
        )
        agent_dir = run_dir / "agent_loop"
        focused_dir = agent_dir / "focused_vlm_interactions" / "iteration_000"
        focused_dir.mkdir(parents=True)
        results = []
        for index in range(1, 35):
            case_id = f"part_image_{index:03d}"
            case_dir = focused_dir / case_id
            case_dir.mkdir()
            req = case_dir / "request.json"
            ans = case_dir / "answer.json"
            req.write_text(json.dumps({"prompt": {"question": "check one part", "do_not_judge": [], "part_context": {"layout_label": index, "part_description": f"part {index}", "bbox_size_xyz": [1, 1, 1], "text_description": f"part {index} text"}}}), encoding="utf-8")
            ans.write_text(json.dumps({"answer": {"score": 0.9, "verdict": "accept", "summary": case_id, "issues": []}}), encoding="utf-8")
            results.append({
                "case_id": case_id,
                "scope": "part_image",
                "part_id": f"part_{index:03d}",
                "status": "completed",
                "score": 0.9,
                "verdict": "accept",
                "summary": case_id,
                "request_path": str(req),
                "answer_path": str(ans),
                "input_images": [],
            })
        focused_summary = focused_dir / "focused_visual_critics_summary.json"
        focused_summary.write_text(json.dumps({"results": results, "scopes": ["part_image"], "provider": "mock", "model": "mock"}), encoding="utf-8")
        answer_path = agent_dir / "answer.json"
        answer_path.write_text(json.dumps({"findings": {"issues": []}}), encoding="utf-8")
        trace_path = agent_dir / "agent_trace.json"
        trace_path.write_text(json.dumps({
            "schema_version": "archstudio_s2_repair_agent_loop.v1",
            "status": "complete",
            "accepted": False,
            "stop_condition": "max_iterations_exhausted",
            "final_scene_score": 0.5,
            "iterations": [{
                "iteration": 0,
                "accepted": False,
                "scene_score": 0.5,
                "evaluator": {
                    "provider": "mock",
                    "model": "mock",
                    "mode": "focused",
                    "status": "completed",
                    "agent_scope": "assemble_geometry",
                    "agent_phase": "per_part_input_and_mesh_precheck",
                    "focused_summary_path": str(focused_summary),
                    "answer_path": str(answer_path),
                    "issue_count": 0,
                },
                "operations": [],
                "stage_records": [],
            }],
        }), encoding="utf-8")

        refresh_s2_9999_report(run_dir, full=False, trigger="test_all_part_cases")
        html_text = (report_dir / "index.html").read_text(encoding="utf-8")
        assert "part_image_001" in html_text
        assert "part_image_034" in html_text
        assert "本轮 focused case 总数" in html_text
        assert "layout #" in html_text
        assert "bbox size XYZ" in html_text
        assert "part description" in html_text
        assert "verdict:" in html_text



def test_focused_part_repair_flow_shows_prompt_revision_and_rerun_records():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "summary.json").write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
        (report_dir / "index.html").write_text("<html><head></head><body><!-- ARCHSTUDIO_S2_REPAIR_AGENT_START --><!-- ARCHSTUDIO_S2_REPAIR_AGENT_END --><!-- ARCHSTUDIO_S2_QUALITY_AGENT_START --><!-- ARCHSTUDIO_S2_QUALITY_AGENT_END --></body></html>", encoding="utf-8")
        part_id = "layout__000__mass_a"
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "parts": [{
                "part_id": part_id,
                "target_class": "open_semantic_part",
                "part_description": "single test architectural mass A",
                "part_prompt": "single isolated standalone architectural component asset A",
                "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                "reference_image_path": str(run_dir / "parts" / part_id / "reference.png"),
            }],
        }
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        write_dummy_png(run_dir / "parts" / part_id / "reference.png")
        write_dummy_png(run_dir / "parts" / part_id / "reference_old.png")
        (run_dir / "parts" / part_id / "prompt_old.txt").write_text("old prompt", encoding="utf-8")
        (run_dir / "parts" / part_id / "prompt.txt").write_text("regenerated prompt with core details only", encoding="utf-8")
        prompt_revision = {
            "schema_version": "archstudio_s2_agent_prompt_revision.v1",
            "part_id": part_id,
            "prompt": "regenerated prompt with core details only",
            "negative_prompt": "whole building",
            "repair_instruction": "make the image a clean isolated part",
            "revision_attempt": 2,
        }
        pr_path = run_dir / "parts" / part_id / "agent_prompt_revision.json"
        pr_path.parent.mkdir(parents=True, exist_ok=True)
        pr_path.write_text(json.dumps(prompt_revision), encoding="utf-8")
        trace = {
            "schema_version": "archstudio_s2_repair_agent_loop.v1",
            "status": "complete",
            "accepted": False,
            "stop_condition": "max_iterations_exhausted",
            "iterations": [{
                "iteration": 0,
                "scene_score": 0.2,
                "evaluator": {
                    "provider": "mock",
                    "model": "mock",
                    "mode": "focused",
                    "status": "completed",
                    "agent_scope": "assemble_geometry",
                    "agent_phase": "per_part_input_and_mesh_precheck",
                    "focused_summary_path": str(run_dir / "agent_loop" / "focused_vlm_interactions" / "iteration_000" / "focused_visual_critics_summary.json"),
                    "answer_path": str(run_dir / "agent_loop" / "focused_vlm_interactions" / "iteration_000" / "answer.json"),
                },
                "stage_records": [
                    {
                        "stage": "prompt_repair",
                        "status": "succeeded",
                        "part_ids": [part_id],
                        "inputs": {},
                        "outputs": {"revisions": [{"part_id": part_id, "prompt_revision_path": str(pr_path), "revision_attempt": 2, "status": "succeeded"}]},
                    },
                    {
                        "stage": "t2i_generation",
                        "status": "succeeded",
                        "part_ids": [part_id],
                        "inputs": {},
                        "outputs": {
                            "status": "succeeded",
                            "command": ["python", "generate_missing_reference_images.py", "--part-id", part_id],
                            "before_artifacts": {part_id: {"paths": {"reference_image": {"path": str(run_dir / "parts" / part_id / "reference_old.png"), "exists": True}}, "metadata": {"prompt_text": "old prompt"}}},
                            "after_artifacts": {part_id: {"paths": {"reference_image": {"path": str(run_dir / "parts" / part_id / "reference.png"), "exists": True}}, "metadata": {"prompt_text": "regenerated prompt with core details only"}}},
                            "prompt_diffs": {part_id: {"changed": True, "added_words_preview": ["regenerated"]}},
                        },
                    },
                    {
                        "stage": "geometry_generation",
                        "status": "succeeded",
                        "part_ids": [part_id],
                        "inputs": {},
                        "outputs": {"status": "succeeded", "results": [{"part_id": part_id, "status": "succeeded", "command_template": "hunyuan ..."}]},
                    },
                ],
            }],
        }
        (run_dir / "agent_loop").mkdir(parents=True, exist_ok=True)
        (run_dir / "agent_loop" / "agent_trace.json").write_text(json.dumps(trace), encoding="utf-8")
        focused_dir = run_dir / "agent_loop" / "focused_vlm_interactions" / "iteration_000"
        focused_dir.mkdir(parents=True, exist_ok=True)
        req = focused_dir / "part_image_001.json"
        ans = focused_dir / "answer.json"
        req.write_text(json.dumps({"prompt": {"question": "q", "do_not_judge": [], "part_context": {"layout_label": 1, "part_description": "single test architectural mass A", "bbox_size_xyz": [1, 1, 1], "text_description": "single isolated standalone architectural component asset A"}}}), encoding="utf-8")
        ans.write_text(json.dumps({"answer": {"score": 0.2, "verdict": "revise", "summary": "needs repair", "issues": [{"issue_type": "bad_aspect", "severity": 4, "evidence": "e", "repair_hint": "h"}]}}), encoding="utf-8")
        (focused_dir / "focused_visual_critics_summary.json").write_text(json.dumps({"part_index": [{"label": 1, "part_id": part_id, "display_label": "single test architectural mass A"}], "results": [{"case_id": "part_image_001", "scope": "part_image", "part_id": None, "label": 1, "status": "completed", "score": 0.2, "verdict": "revise", "summary": "needs repair", "request_path": str(req), "answer_path": str(ans), "input_images": [{"label": "part_reference_image", "source_path": str(run_dir / "parts" / part_id / "reference.png"), "copied_path": str(run_dir / "parts" / part_id / "reference.png")}]}], "scopes": ["part_image"], "provider": "mock", "model": "mock"}), encoding="utf-8")
        refresh_s2_9999_report(run_dir, full=False, trigger="repair_flow_test")
        html_text = (report_dir / "index.html").read_text(encoding="utf-8")
        assert "layout #" in html_text
        assert "bbox size XYZ" in html_text
        assert "verdict: revise" in html_text
        assert "Agent action · prompt_repair" in html_text
        assert "Agent action · t2i_generation" in html_text
        assert html_text.index("单次 VLM call") < html_text.index("Agent action · prompt_repair")
        assert "旧 T2I reference image" in html_text
        assert "新生成 T2I reference image" in html_text
        assert "旧 prompt" in html_text
        assert "新 prompt" in html_text
        assert "same global numeric label as layout_s1_style_multiview_context" in html_text
        assert "generate_missing_reference_images.py" in html_text
        assert "hunyuan" in html_text


def test_t2i_rerun_localizes_reference_and_reads_agent_revision():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_dir = tmp_path / "agent_run"
        source_dir = tmp_path / "source_run"
        part_id = "layout__000__mass_a"
        source_reference = source_dir / "parts" / part_id / "reference.png"
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "parts": [{
                "part_id": part_id,
                "target_class": "open_semantic_part",
                "part_description": "single test architectural mass A",
                "part_prompt": "old source prompt that should not be used",
                "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                "reference_image_path": str(source_reference),
                "reference_image": str(source_reference),
            }],
        }
        (run_dir / "contract").mkdir(parents=True)
        contract_path = run_dir / "contract" / "layout_mesh_contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        revision_path = run_dir / "parts" / part_id / "agent_prompt_revision.json"
        revision_path.parent.mkdir(parents=True)
        revision_path.write_text(json.dumps({
            "schema_version": "archstudio_s2_agent_prompt_revision.v1",
            "part_id": part_id,
            "prompt": "NEW AGENT REPAIRED PROMPT with isolated gallery wing",
            "negative_prompt": "whole building",
            "repair_instruction": "make it isolated",
            "revision_attempt": 3,
        }), encoding="utf-8")

        config = S2AgentConfig(run_dir=run_dir, provider="procedural_reference", t2i_python=Path(sys.executable))
        command = _t2i_command(config, [part_id])
        assert "--output-run-dir" in command
        assert str(run_dir) in command
        completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr + completed.stdout
        local_reference = run_dir / "parts" / part_id / "reference.png"
        assert local_reference.exists()
        assert not source_reference.exists()
        metadata = json.loads((run_dir / "parts" / part_id / "t2i_metadata.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "succeeded"
        assert metadata["prompt"] == "NEW AGENT REPAIRED PROMPT with isolated gallery wing"
        assert metadata["agent_prompt_revision"]["revision_attempt"] == 3



def test_t2i_gpu_filter_skips_busy_gpus_and_dispatches_only_eligible():
    import scene_synthesis.building_mesh.s2_repair_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "agent_run"
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps({"parts": []}), encoding="utf-8")
        calls = []
        old_query = agent._query_gpu_memory_snapshot
        old_run = agent.subprocess.run
        old_env = agent.os.environ.get("S2_T2I_MIN_FREE_GPU_MEM_MIB")

        class Completed:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_query():
            return [
                {"index": "0", "memory_used_mib": 47000, "memory_free_mib": 1000},
                {"index": "1", "memory_used_mib": 46000, "memory_free_mib": 2000},
                {"index": "7", "memory_used_mib": 100, "memory_free_mib": 48000},
            ]

        def fake_run(command, **kwargs):
            calls.append({"command": command, "env": kwargs.get("env") or {}})
            return Completed()

        try:
            agent._query_gpu_memory_snapshot = fake_query
            agent.subprocess.run = fake_run
            agent.os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = "30000"
            config = S2AgentConfig(
                run_dir=run_dir,
                provider="qwen_image_local",
                t2i_python=Path(sys.executable),
                t2i_gpus=["0", "1", "7"],
            )
            result = agent.run_t2i_for_parts(config, ["part_a", "part_b"])
        finally:
            agent._query_gpu_memory_snapshot = old_query
            agent.subprocess.run = old_run
            if old_env is None:
                agent.os.environ.pop("S2_T2I_MIN_FREE_GPU_MEM_MIB", None)
            else:
                agent.os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = old_env

        assert result["status"] == "succeeded"
        assert result["eligible_gpus"] == ["7"]
        assert [item["gpu"] for item in result["subprocesses"]] == ["7"]
        assert calls and {call["env"].get("CUDA_VISIBLE_DEVICES") for call in calls} == {"7"}
        assert all(call["env"].get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True" for call in calls)
        assert {item["gpu"] for item in result["skipped_gpus"]} == {"0", "1"}


def test_t2i_gpu_filter_reports_no_eligible_gpu_without_subprocess():
    import scene_synthesis.building_mesh.s2_repair_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "agent_run"
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps({"parts": []}), encoding="utf-8")
        calls = []
        old_query = agent._query_gpu_memory_snapshot
        old_run = agent.subprocess.run
        old_env = agent.os.environ.get("S2_T2I_MIN_FREE_GPU_MEM_MIB")

        def fake_query():
            return [
                {"index": "0", "memory_used_mib": 47000, "memory_free_mib": 1000},
                {"index": "1", "memory_used_mib": 46000, "memory_free_mib": 2000},
            ]

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("T2I subprocess should not launch when no GPU is eligible")

        try:
            agent._query_gpu_memory_snapshot = fake_query
            agent.subprocess.run = fake_run
            agent.os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = "30000"
            config = S2AgentConfig(
                run_dir=run_dir,
                provider="qwen_image_local",
                t2i_python=Path(sys.executable),
                t2i_gpus=["0", "1"],
            )
            result = agent.run_t2i_for_parts(config, ["part_a"])
        finally:
            agent._query_gpu_memory_snapshot = old_query
            agent.subprocess.run = old_run
            if old_env is None:
                agent.os.environ.pop("S2_T2I_MIN_FREE_GPU_MEM_MIB", None)
            else:
                agent.os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = old_env

        assert result["status"] == "failed_no_eligible_gpu"
        assert result["subprocesses"] == []
        assert result["eligible_gpus"] == []
        assert "No requested T2I GPU has enough free memory" in result["stderr_tail"]
        assert calls == []


def test_t2i_oom_retries_on_another_eligible_gpu():
    import scene_synthesis.building_mesh.s2_repair_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "agent_run"
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps({"parts": []}), encoding="utf-8")
        calls = []
        old_query = agent._query_gpu_memory_snapshot
        old_run = agent.subprocess.run
        old_env = agent.os.environ.get("S2_T2I_MIN_FREE_GPU_MEM_MIB")

        class Completed:
            def __init__(self, returncode, stderr=""):
                self.returncode = returncode
                self.stdout = ""
                self.stderr = stderr

        def fake_query():
            return [
                {"index": "6", "memory_used_mib": 10, "memory_free_mib": 48000},
                {"index": "7", "memory_used_mib": 10, "memory_free_mib": 48000},
            ]

        def fake_run(command, **kwargs):
            gpu = (kwargs.get("env") or {}).get("CUDA_VISIBLE_DEVICES")
            calls.append(gpu)
            if gpu == "6":
                return Completed(2, "OutOfMemoryError: CUDA out of memory")
            return Completed(0, "")

        try:
            agent._query_gpu_memory_snapshot = fake_query
            agent.subprocess.run = fake_run
            agent.os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = "30000"
            config = S2AgentConfig(
                run_dir=run_dir,
                provider="qwen_image_local",
                t2i_python=Path(sys.executable),
                t2i_gpus=["6", "7"],
            )
            result = agent.run_t2i_for_parts(config, ["part_a"])
        finally:
            agent._query_gpu_memory_snapshot = old_query
            agent.subprocess.run = old_run
            if old_env is None:
                agent.os.environ.pop("S2_T2I_MIN_FREE_GPU_MEM_MIB", None)
            else:
                agent.os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = old_env

        assert result["status"] == "succeeded"
        assert calls == ["6", "7"]
        proc = result["subprocesses"][0]
        assert proc["gpu"] == "7"
        assert proc["retried_after_cuda_oom"] is True
        assert [attempt["gpu"] for attempt in proc["attempts"]] == ["6", "7"]
        assert all("attempts" not in attempt for attempt in proc["attempts"])
        json.dumps(proc)


def test_t2i_wrapper_persists_gpu_and_failure_evidence():
    import scene_synthesis.building_mesh.s2_repair_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "agent_run"
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps({"parts": []}), encoding="utf-8")
        part_dir = run_dir / "parts" / "part_a"
        part_dir.mkdir(parents=True)
        (part_dir / "t2i_metadata.json").write_text(
            json.dumps({"part_id": "part_a", "provider": "qwen_image_local", "status": "failed"}),
            encoding="utf-8",
        )
        old_query = agent._query_gpu_memory_snapshot
        old_run = agent.subprocess.run
        old_env = agent.os.environ.get("S2_T2I_MIN_FREE_GPU_MEM_MIB")

        class Completed:
            returncode = 2
            stdout = "stdout evidence"
            stderr = "stderr evidence"

        try:
            agent._query_gpu_memory_snapshot = lambda: [{"index": "7", "memory_used_mib": 10, "memory_free_mib": 48000}]
            agent.subprocess.run = lambda *args, **kwargs: Completed()
            agent.os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = "30000"
            config = S2AgentConfig(
                run_dir=run_dir,
                provider="qwen_image_local",
                t2i_python=Path(sys.executable),
                t2i_gpus=["7"],
            )
            result = agent.run_t2i_for_parts(config, ["part_a"])
        finally:
            agent._query_gpu_memory_snapshot = old_query
            agent.subprocess.run = old_run
            if old_env is None:
                agent.os.environ.pop("S2_T2I_MIN_FREE_GPU_MEM_MIB", None)
            else:
                agent.os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = old_env

        assert result["status"] == "failed"
        metadata = json.loads((part_dir / "t2i_metadata.json").read_text(encoding="utf-8"))
        assert metadata["selected_gpu"] == "7"
        assert metadata["wrapper_status"] == "failed"
        assert metadata["wrapper_returncode"] == 2
        assert metadata["wrapper_attempts"][0]["gpu"] == "7"
        assert "stderr evidence" in metadata["wrapper_stderr_tail"]




def test_part_artifact_snapshot_freezes_before_after_files():
    import scene_synthesis.building_mesh.s2_repair_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "agent_run"
        part_id = "layout__000__mass_a"
        part_dir = run_dir / "parts" / part_id
        part_dir.mkdir(parents=True)
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps({"parts": [{"part_id": part_id}]}), encoding="utf-8")
        (run_dir / "manifest.json").write_text(json.dumps({"parts": [{"part_id": part_id}]}), encoding="utf-8")
        (run_dir / "failures.json").write_text(json.dumps({"failures": []}), encoding="utf-8")
        reference = part_dir / "reference.png"
        prompt = part_dir / "prompt.txt"
        reference.write_bytes(b"old-reference")
        prompt.write_text("old prompt", encoding="utf-8")

        before = agent._part_artifact_snapshot(run_dir, part_id, freeze_dir=run_dir / "agent_loop" / "snap" / "before", snapshot_label="before")
        reference.write_bytes(b"new-reference")
        prompt.write_text("new prompt", encoding="utf-8")
        after = agent._part_artifact_snapshot(run_dir, part_id, freeze_dir=run_dir / "agent_loop" / "snap" / "after", snapshot_label="after")

        before_path = Path(before["paths"]["reference_image"]["path"])
        after_path = Path(after["paths"]["reference_image"]["path"])
        assert before_path != reference
        assert after_path != reference
        assert before_path.read_bytes() == b"old-reference"
        assert after_path.read_bytes() == b"new-reference"
        assert before["paths"]["reference_image"]["live_path"] == str(reference)
        assert before["metadata"]["prompt_text"] == "old prompt"
        assert after["metadata"]["prompt_text"] == "new prompt"

def test_geometry_rerun_uses_local_repaired_reference_image():
    import scene_synthesis.building_mesh.s2_repair_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_dir = tmp_path / "agent_run"
        source_reference = tmp_path / "source_run" / "parts" / "part_a" / "reference.png"
        local_reference = run_dir / "parts" / "part_a" / "reference.png"
        write_dummy_png(source_reference)
        write_dummy_png(local_reference)
        contract = {
            "schema_version": "layout_mesh_contract.v1",
            "layout_id": "layout",
            "parts": [{
                "schema_version": "layout_mesh_contract_part.v1",
                "layout_id": "layout",
                "part_id": "part_a",
                "target_prompt": "test mass",
                "target_class": "open_semantic_part",
                "bbox": {"center": [0, 0, 0], "size": [1, 1, 1]},
                "contract_path": str(run_dir / "contract" / "layout_mesh_contract.json"),
                "reference_image_path": str(source_reference),
                "reference_image": str(source_reference),
            }],
        }
        manifest = {"parts": [{"part_id": "part_a"}]}
        (run_dir / "contract").mkdir(parents=True)
        (run_dir / "contract" / "layout_mesh_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "failures.json").write_text(json.dumps({"failures": []}), encoding="utf-8")

        captured = {}
        old_adapter = agent.HunyuanAdapter
        old_rebuild = agent.rebuild_run_assemblies

        class DummyResult:
            raw_output_path = None
            final_failure_type = "missing_output_mesh"
            layout_id = "layout"
            schema_version = "layout_mesh_contract.v1"
            attempts = []

            def to_dict(self):
                return {"part_id": "part_a", "attempts": [], "raw_output_path": None}

        class DummyAdapter:
            def __init__(self, *args, **kwargs):
                pass

            def run_part(self, part):
                captured["reference_image_path"] = part.get("reference_image_path")
                return DummyResult()

        try:
            agent.HunyuanAdapter = DummyAdapter
            agent.rebuild_run_assemblies = lambda run_dir_arg: {"status": "skipped"}
            result = agent.run_geometry_for_parts(
                S2AgentConfig(run_dir=run_dir, hunyuan_command="python fake.py --image $reference_image"),
                ["part_a"],
            )
        finally:
            agent.HunyuanAdapter = old_adapter
            agent.rebuild_run_assemblies = old_rebuild

        assert captured["reference_image_path"] == str(local_reference)
        assert result["results"][0]["reference_image_path"] == str(local_reference)

def main() -> int:
    test_focused_accept_with_minor_advisory_issue_does_not_block()
    test_t2i_rerun_seed_changes_by_iteration()
    test_append_or_replace_cli_option_updates_seed_inside_bash_lc()
    test_geometry_rerun_marks_byte_identical_mesh_as_unchanged()
    test_validation_rejects_gap_and_plans_snap()
    test_split_operation_replaces_contract_part_with_surface_children()
    test_repairer_writes_reject_accept_trace_in_dry_run()
    test_vlm_findings_are_converted_and_drive_operations()
    test_repairer_uses_mock_vlm_evaluator_in_dry_run()
    test_repairer_uses_focused_vlm_findings_in_dry_run()
    test_assemble_geometry_scope_prechecks_parts_before_scene_assembly()
    test_assemble_geometry_scope_advances_part_image_to_mesh_then_next_part()
    test_sequential_max_iterations_resets_after_stage_acceptance()
    test_prompt_revision_persists_vlm_repair_instruction()
    test_repairer_auto_stops_on_no_improvement()
    test_9999_refresh_hook_updates_main_report_with_agent_io()
    test_9999_refresh_shows_all_focused_part_precheck_cases()
    test_focused_part_repair_flow_shows_prompt_revision_and_rerun_records()
    test_t2i_rerun_localizes_reference_and_reads_agent_revision()
    test_t2i_gpu_filter_skips_busy_gpus_and_dispatches_only_eligible()
    test_t2i_gpu_filter_reports_no_eligible_gpu_without_subprocess()
    test_t2i_oom_retries_on_another_eligible_gpu()
    test_t2i_wrapper_persists_gpu_and_failure_evidence()
    test_part_artifact_snapshot_freezes_before_after_files()
    test_geometry_rerun_uses_local_repaired_reference_image()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
