# ArchStudio-S2 Assemble Geometry Agent Plan

## Current S2 pipeline boundary

Stage 2 currently consumes Stage 1 part-level layout records with detailed text descriptions and keeps the generation pipeline ordered as:

1. **T2I / part reference image**: create one part-level reference image per layout part.
2. **I23D / I3D geometry**: call Hunyuan-Omni3D with the part image plus layout bbox constraints to generate an OBJ/GLB mesh.
3. **Assemble**: normalize each generated mesh into its layout bbox and concatenate per-part geometry into a scene assembly.
4. **Texture**: optional downstream Hunyuan3D-Paint texture stage.

The first agent milestone intentionally does **not** replace that pipeline.  It adds an auditable reject/accept and repair-or-route layer around the **Assemble geometry** output first.

## Code-state review

| Stage | Current entry points | Relevant code | Role in agent milestone |
| --- | --- | --- | --- |
| T2I | `scripts/generate_part_reference_images.py`; `scripts/run_archstudio_s2_agent_loop.py` repair reruns | `scene_synthesis/building_mesh/reference_images.py:QwenImageLocalProvider`, `provider_from_name`, `write_t2i_metadata` | Upstream route target only in assemble-only mode; do not bulk-change prompts during assemble milestone. |
| I23D / I3D | `scripts/generate_building_mesh_hunyuan.py`; `scripts/hunyuan_bbox_single.py` | `scene_synthesis/building_mesh/hunyuan_adapter.py:HunyuanAdapter`, `build_hunyuan_command`, `discover_output_meshes` | Upstream route target for part-level geometry failures; assemble-only mode records planned rerun handoff instead of launching GPU generation. |
| Assemble | main S2 generation and rebuild scripts | `scene_synthesis/building_mesh/assembly.py:build_assemblies`, `normalize_mesh_to_bbox`, `cleanup_mesh_to_bbox`, `write_placeholder_mesh` | First concrete repair surface: deterministic bbox snap/resize and reassembly. |
| Metrics / VLM quality | `scripts/evaluate_archstudio_s2_quality.py`; `scripts/run_archstudio_s2_focused_critics.py` | `scene_synthesis/building_mesh/quality_agent.py:collect_quality_metrics`, `run_focused_visual_critics`, `quality_findings_from_focused_visual_critics`, `build_repair_plan` | Produces scene-assembly issues with numeric labels, part ids, severity, evidence, repair route, and repair hint. |
| Agent loop | `scripts/run_archstudio_s2_agent_loop.py` | `scene_synthesis/building_mesh/s2_repair_agent.py:S2Repairer`, `S2AgentConfig`, `validate_s2_run`, `repair_operations_from_findings`, `apply_snap_resize_operation`, `rebuild_run_assemblies` | Owns iteration, trace, operation execution, and 9999 refresh. |
| 9999 report | normal report generation plus post-agent refresh | `scene_synthesis/building_mesh/report_bundle.py:build_report`, `refresh_s2_9999_report`, `render_s2_repair_agent_loop_section`, `render_quality_agent_section` | Must show each agent step, input images, JSON outputs, repairs, and stop condition. |

## Implemented assemble-geometry scope

The current CLI accepts:

```bash
python scripts/run_archstudio_s2_agent_loop.py \
  --run-dir <existing_s2_run> \
  --agent-scope assemble_geometry \
  --evaluator-provider <mock|gpt55|openai|aimirror> \
  --fast-report-refresh
```

The `assemble_geometry` scope constrains the loop as follows:

- Focused critic scopes default to `scene_assembly` only.
- VLM input is a true surface mesh six-view board, not point-cloud scatter.
- The scene VLM prompt requires concrete `labels`, optional `part_ids`, `issue_type`, `severity`, `evidence`, `repair_route`, and `repair_hint`.
- Labels are mapped back to stable contract `part_id`s through `part_index`.
- Deterministic geometry issues (`gap_or_seam`, `floating`, `penetration`, `overlap`, `scale_mismatch`) route to bbox snap/resize repair.
- Upstream semantic or low-quality issues route to `rerun_i3d` or `rerun_t2i_i3d`; in assemble-only mode these become explicit handoff records instead of silently launching the upstream stages.
- Each trace write triggers the 9999 refresh hook so report pages expose latest inputs/outputs without a full report rebuild.

## Assemble Geometry Agent loop contract

Per iteration:

1. Collect deterministic quality metrics over the assembled run.
2. Render/copy focused scene-assembly VLM inputs:
   - colored assembled mesh true-surface six-view board;
   - part index / layout numeric context in JSON request;
   - S1-style layout board with plain global numeric labels where layout context is needed.
3. Ask the focused scene critic for JSON:
   - `score`, `verdict`, `summary`;
   - issue list with `issue_type`, `labels`, `part_ids`, `severity`, `evidence`, `repair_route`, `repair_hint`.
4. Convert VLM result into normalized quality findings.
5. Plan part-specific operations:
   - deterministic snap/resize for assembly contact failures;
   - I3D rerun handoff for bad geometry/orientation/detail;
   - T2I+I3D handoff for wrong semantics/reference mismatch;
   - manual review for ambiguous cases.
6. Apply enabled assemble-local operations and rebuild assembly.
7. Persist `agent_loop/agent_trace.json`, focused request/answer files, operation records, and 9999 report sections.
8. Stop on accept, max iterations, no improvement, or no enabled safe operation.

## 9999 visualization gates

A run is reviewable only if `report/index.html` exposes:

- Agent scope and stop condition.
- Per-iteration score trajectory and issue count.
- VLM input images with clickable full-size paths.
- VLM request JSON and answer JSON links.
- Operation timeline with part ids, operation type, status, and repair hints.
- Final assembly assets after repair.
- `report/s2_9999_refresh.json` hook metadata.

The current mini-run evidence is:

- Run directory: `/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2_agent_loop/zz_assemble_geometry_agent_minirun_20260523_233252`
- 9999 URL: `http://192.168.192.123:9999/projects/buildingblock/runs/b3Blbl9zZW1hbnRpY19zMl9hZ2VudF9sb29wL3p6X2Fzc2VtYmxlX2dlb21ldHJ5X2FnZW50X21pbmlydW5fMjAyNjA1MjNfMjMzMjUy/`
- Trace: `agent_loop/agent_trace.json` reports `status=complete`, `stop_condition=accepted`, `agent_scope=assemble_geometry`.
- Loop behavior: iteration 0 rejected a gap/seam and applied deterministic snap; iteration 1 accepted.

## Current risks / next implementation cuts

1. **Dense real Hunyuan runs are still expensive.** Full report refresh and snap/reassembly on large OBJ assets can take minutes. Assemble-only mode should continue to prefer fast report refresh and reuse already-rendered VLM inputs where possible.
2. **True full-building run evidence is partial.** The mini-run proves loop mechanics and 9999 visibility. A larger real Hunyuan run should be used as the next visual acceptance artifact once performance is bounded.
3. **Upstream repair is intentionally routed, not executed.** This is correct for the assemble-first milestone, but the later I3D/T2I agent stories must turn these route records into bounded part-level reruns.
4. **Operation safety must stay part-local.** Global re-generation is outside the assemble milestone; repairs should preserve the existing pipeline artifacts and edit only selected part meshes/manifest entries.
5. **VLM provider failure must remain auditable.** If GPT/VLM fails, trace the failure and either stop or fall back to deterministic-only mode according to `--evaluator-required`.

## Recommended next step

Before implementing broader T2I/I3D agents, run `--agent-scope assemble_geometry` on one representative real Hunyuan S2 run with:

- max one iteration first;
- `--fast-report-refresh` enabled;
- true mesh six-view board persisted;
- 9999 report link shared;
- no upstream reruns enabled unless explicitly testing route execution.

That makes the user review the real agent behavior first, then we can tighten prompts/repair routes before expanding to per-part T2I and I3D loops.
