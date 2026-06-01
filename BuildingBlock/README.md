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


## Stage2 with Agentic

The current experimental Stage2 agent layer is documented in
[`docs/stage2_with_agentic_readme.md`](docs/stage2_with_agentic_readme.md).
It records the S1-layout-to-S2 pipeline boundary, focused VLM critic loops,
9999 visualization/debug workflow, known blockers, and restart checklist.

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
