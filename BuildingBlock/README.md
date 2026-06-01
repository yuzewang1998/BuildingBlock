# BuildingBlock

[**[SIGGRAPH 2025]BuildingBlock: A Hybrid Approach for Structured Building Generation**](https://arxiv.org/pdf/2505.04051)

This directory contains the core BuildingBlock code used for training and generation.

## Installation

Run the initialization script from the repository root:

```bash
bash initialization.sh
```

## Dataset Preparation

Extract the open-source dataset package, then run:

```bash
python 1-json_rotate_augment.py
python 2-normUeJson.py
python 3-json2boxnp.py
cp dataset_stats.txt ./BoxCenterSizeLabelNp
```

## Running

Training and generation entry points live under `scripts/`.



## Stage2 / Stage2 with Agentic

This repository now contains an experimental **Stage2 with Agentic** track.  It is
Stage2 code: it starts from the Stage1 part-level layout with detailed part text,
then runs/inspects the Stage2 pipeline:

```text
Stage1 layout + part descriptions
  -> T2I part reference images
  -> I23D / Hunyuan3D-Omni part meshes
  -> Assemble part meshes into the full building
  -> Texture generation
```

The agentic layer does not replace the Stage2 pipeline.  It adds focused VLM
review, repair routing, iteration traces, and 9999 visualization around Stage2.
For the full restart notes, current progress, known issues, and future plan, see:
[`docs/stage2_with_agentic_readme.md`](docs/stage2_with_agentic_readme.md).

### Stage2 code map

| Path | Purpose |
| --- | --- |
| `scene_synthesis/building_mesh/` | Core Stage2 layout-to-mesh, mesh assembly, visualization, reporting, and agent logic. |
| `scene_synthesis/building_mesh/layout_io.py` | Loads/parses Stage1 part-level layout records and geometry contracts. |
| `scene_synthesis/building_mesh/prompting.py` | Builds Stage2 prompts from part descriptions, bbox/layout context, and repair hints. |
| `scene_synthesis/building_mesh/reference_images.py` | T2I reference-image providers, including local Qwen-Image support and metadata writing. |
| `scene_synthesis/building_mesh/assembly.py` | Normalizes generated part meshes to layout bboxes and assembles the full building mesh. |
| `scene_synthesis/building_mesh/visualization.py` | S1-style layout multiview rendering, true mesh six-view rendering, and report assets. |
| `scene_synthesis/building_mesh/quality_agent.py` | Focused VLM critic inputs/outputs for part-image, part-mesh, and assembled-scene checks. |
| `scene_synthesis/building_mesh/s2_repair_agent.py` | Agent repair planning/execution: T2I prompt repair, geometry reruns, assemble-local fixes, iteration state. |
| `scene_synthesis/building_mesh/s2_agent_trace.py` | Trace schema/helpers for recording Agent steps, VLM calls, actions, and artifacts. |
| `scene_synthesis/building_mesh/report_bundle.py` | 9999 report generation/refresh, including Focused Visual Critic QA and Agent action cards. |
| `scripts/run_open_semantic_s2_batch.py` | Runs the open-semantic Stage2 batch pipeline. |
| `scripts/generate_part_reference_images.py` | Generates Stage2 part-level T2I reference images. |
| `scripts/hunyuan_bbox_single.py` | Runs single-part Hunyuan bbox-conditioned I23D generation. |
| `scripts/run_archstudio_s2_agent_loop.py` | Main Stage2 Agent loop entry point for real runs. |
| `scripts/run_archstudio_s2_focused_critics.py` | Runs focused VLM critics separately for debugging/evaluation. |
| `scripts/evaluate_archstudio_s2_quality.py` | Quality evaluation helper for Stage2 outputs. |
| `scripts/repair_archstudio_s2_parts.py` | Repair helper for selected Stage2 parts. |
| `scripts/rebuild_archstudio_s2_assemblies_from_manifest.py` | Rebuilds assembled meshes from manifests after part edits/repairs. |
| `scripts/texture_archstudio_s2_hunyuan_paint.py` | Texture-generation stage helper. |
| `scripts/tests/test_archstudio_s2_quality_agent.py` | Regression tests for focused critic / quality-agent behavior. |
| `scripts/tests/test_archstudio_s2_repair_agent.py` | Regression tests for Stage2 repair-agent behavior. |

## Layout to Mesh V1

After generating a BuildingBlock layout JSON, prepare a Hunyuan3D-Omni mesh run:

```bash
python scripts/generate_building_mesh_hunyuan.py path/to/layout.json \
  --run-dir outputs/hunyuan_mesh_run \
  --prepare-only
```

To run Hunyuan per part, pass a command template for the official Hunyuan
inference entry point. The template supports `$contract_path`, `$output_dir`,
`$part_id`, `$target_prompt`, and `$target_class`.

```bash
python scripts/generate_building_mesh_hunyuan.py path/to/layout.json \
  --run-dir outputs/hunyuan_mesh_run \
  --hunyuan-command 'python /path/to/Hunyuan3D-Omni/inference.py --control_type bbox ...'
```

The V1 pipeline writes a versioned layout-to-mesh contract, per-part logs, a
raw Hunyuan assembly, a placeholder-filled assembly, and manifest/failure
reports.
