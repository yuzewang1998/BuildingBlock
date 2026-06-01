"""Trace helpers for the ArchStudio-S2 part-level generation agent.

The current S2 agent is intentionally lightweight: it records the observable
decision chain for every layout part rather than hiding the pipeline behind a
single final mesh.  Later evaluator/retry agents can consume the same trace.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


TRACE_SCHEMA_VERSION = "archstudio_s2_agent_trace.v1"


def _load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rel_or_abs(path: str | Path | None, base: Path) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return str(candidate)


def build_agent_trace(run_dir: str | Path, texture_dir: str | Path | None = None) -> dict[str, Any]:
    """Collect per-part T2I/I23D/texture statuses into one inspectable trace."""
    run_dir = Path(run_dir).resolve()
    contract = _load_json_if_exists(run_dir / "contract" / "layout_mesh_contract.json") or {"parts": []}
    manifest = _load_json_if_exists(run_dir / "manifest.json") or {"parts": []}
    failures = _load_json_if_exists(run_dir / "failures.json") or {"failures": []}

    manifest_by_id = {
        str(item.get("part_id")): item
        for item in manifest.get("parts", [])
        if isinstance(item, Mapping) and item.get("part_id")
    }
    failure_by_part: dict[str, list[dict[str, Any]]] = {}
    for item in failures.get("failures", []) or []:
        if not isinstance(item, Mapping):
            continue
        failure_by_part.setdefault(str(item.get("part_id", "")), []).append(dict(item))

    texture_manifest = None
    texture_by_id: dict[str, dict[str, Any]] = {}
    if texture_dir:
        texture_dir = Path(texture_dir).resolve()
        texture_manifest = _load_json_if_exists(texture_dir / "texture_manifest.json")
        if isinstance(texture_manifest, Mapping):
            texture_by_id = {
                str(item.get("part_id")): dict(item)
                for item in texture_manifest.get("parts", []) or []
                if isinstance(item, Mapping) and item.get("part_id")
            }

    part_records = []
    for label, part in enumerate(contract.get("parts", []) or [], start=1):
        part_id = str(part.get("part_id", ""))
        reference_image = Path(part.get("reference_image_path") or part.get("reference_image") or "")
        t2i_metadata = _load_json_if_exists(reference_image.parent / "t2i_metadata.json") if str(reference_image) else None
        geometry = manifest_by_id.get(part_id, {})
        texture = texture_by_id.get(part_id, {})
        attempts = geometry.get("attempts") or []
        final_attempt = attempts[-1] if attempts else {}
        record = {
            "label": label,
            "part_id": part_id,
            "part_description": part.get("part_description") or part.get("open_vocab_label") or part.get("source_actor_label"),
            "part_description_core": part.get("part_description_core"),
            "part_visual_subject": part.get("part_visual_subject"),
            "part_context_tail": part.get("part_context_tail"),
            "semantic_role": part.get("semantic_role", ""),
            "bbox": part.get("bbox", {}),
            "source_bbox_size": part.get("source_bbox_size"),
            "bbox_repair_warning": part.get("bbox_repair_warning"),
            "t2i": {
                "status": (t2i_metadata or {}).get("status", "missing_metadata"),
                "provider": (t2i_metadata or {}).get("provider", part.get("t2i_provider", "")),
                "prompt_policy_version": (t2i_metadata or {}).get("prompt_trace", {}).get("prompt_policy_version") or part.get("prompt_policy_version", ""),
                "prompt": (t2i_metadata or {}).get("prompt") or part.get("part_prompt", ""),
                "negative_prompt": (t2i_metadata or {}).get("negative_prompt") or part.get("negative_prompt", ""),
                "reference_image_path": str(reference_image) if str(reference_image) else "",
                "ratio_metadata": (t2i_metadata or {}).get("ratio_metadata"),
                "prompt_trace": (t2i_metadata or {}).get("prompt_trace", {}),
            },
            "geometry": {
                "status": "succeeded" if geometry.get("raw_output_path") else "failed_or_placeholder",
                "raw_output_path": geometry.get("raw_output_path", ""),
                "normalized_output_path": geometry.get("normalized_output_path", ""),
                "placeholder_output_path": geometry.get("placeholder_output_path", ""),
                "lifecycle_states": geometry.get("lifecycle_states", []),
                "attempt_count": len(attempts),
                "final_failure_type": final_attempt.get("failure_type"),
                "failures": failure_by_part.get(part_id, []),
            },
            "texture": {
                "status": texture.get("status", "not_run" if not texture_manifest else "missing_record"),
                "output_obj": texture.get("output_obj", ""),
                "output_glb": texture.get("output_glb", ""),
                "surface_preserved": texture.get("surface_preserved", texture.get("geometry_preserved")),
                "note": texture.get("note", texture.get("reason", "")),
            },
        }
        part_records.append(record)

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "texture_dir": str(Path(texture_dir).resolve()) if texture_dir else "",
        "layout_id": contract.get("layout_id", manifest.get("layout_id", "")),
        "agent_policy": {
            "name": "ArchStudio-S2 part agent",
            "version": "v2_reject_accept_repair_loop_available",
            "current_capabilities": [
                "part-focused T2I prompt construction",
                "per-part T2I metadata capture",
                "per-part Hunyuan-Omni attempt/status capture",
                "optional texture-stage status capture",
                "S1-style validate -> reject/accept -> repair loop via scripts/run_archstudio_s2_agent_loop.py",
                "deterministic mesh snap/resize for bbox-contact gaps",
                "selected-part T2I and Hunyuan geometry reruns",
                "generic bbox-surface child-part decomposition for box-like open-vocabulary parts",
            ],
            "next_capabilities": [
                "multimodal GPT visual evaluator as an acceptance critic",
                "multi-candidate selection and prompt mutation policies",
                "texture-stage reject/accept repair loop",
            ],
        },
        "summary": {
            "num_parts": len(part_records),
            "num_t2i_succeeded": sum(1 for item in part_records if item["t2i"]["status"] == "succeeded"),
            "num_geometry_succeeded": sum(1 for item in part_records if item["geometry"]["status"] == "succeeded"),
            "num_texture_succeeded": sum(1 for item in part_records if str(item["texture"]["status"]).startswith("succeeded")),
        },
        "parts": part_records,
    }


def write_agent_trace(run_dir: str | Path, texture_dir: str | Path | None = None) -> str:
    run_dir = Path(run_dir).resolve()
    trace = build_agent_trace(run_dir, texture_dir=texture_dir)
    output_path = run_dir / "agent_trace" / "s2_agent_trace.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output_path)


def copy_trace_to_texture_dir(run_dir: str | Path, texture_dir: str | Path) -> str | None:
    """Copy the geometry-side trace into a texture run for texture reports."""
    source = Path(run_dir).resolve() / "agent_trace" / "s2_agent_trace.json"
    if not source.exists():
        return None
    target = Path(texture_dir).resolve() / "agent_trace" / "s2_agent_trace.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return str(target)
