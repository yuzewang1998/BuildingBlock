# BuildingBlock
[**[SIGGRAPH 2025]BuildingBlock: A Hybrid Approach for Structured Building Generation**](https://arxiv.org/pdf/2505.04051) <br>

![pipeline_cropped_00](https://github.com/user-attachments/assets/1fa83ce6-4152-4277-811f-79849562cabc)

![1749536947556](https://github.com/user-attachments/assets/2fbe3d2c-89cf-4583-9e95-04eb447886a1)

## Installation

### 1. Download Docker

Ensure Docker is properly installed on your system. Visit [Docker's official website](https://www.docker.com/get-started) for installation instructions specific to your operating system.

```bash
docker pull dreaminghuang/building_block:0.1
```
### 2. Run Docker Container

Mount your local directory to the container and start an interactive session:

```bash
docker run -it --gpus all -v your_path_of_building_block:/building_block -w /building_block building_block:0.1 /bin/bash
```

Replace `your_path_of_building_block` with the absolute path to your local directory where you want to store the project files.

### 3. Initialize Environment

Run the initialization script to set up the required dependencies:

```bash
bash initialization.sh
```

## Dataset Preparation

### 1. Dataset Download and Extraction
Download the dataset from  
👉 https://huggingface.co/datasets/dreaming-huang/buildingblock/blob/main/building_block_data_opensource_only_layout_and_cond.zip <br>
Extract the `BoxCenterSizeLabel_all` directory from the building_block_data_opensource_only_layout_and_cond.zip:
```bash
unzip building_block_data_opensource_only_layout_and_cond.zip
```

### 2. Process Dataset

Run the following Python scripts in sequence to preprocess the data:

```bash
python 1-json_rotate_augment.py
```
This script performs data augmentation through rotation of the original JSON files.

```bash
python 2-normUeJson.py
```
This script normalizes the Unreal Engine JSON format data.

```bash
python 3-json2boxnp.py
```
This script converts the JSON data to NumPy arrays in box representation format.

```bash
cp dataset_stats.txt ./BoxCenterSizeLabelNp
```

## Running the Model

Navigate to the scripts directory:

```bash
cd scripts
```

Execute the commands provided in `command.sh` to train and/or evaluate the model:

```bash
# View the available commands
cat command.sh

# Execute specific commands as needed
# For example:
python train_diffusion_building_DDP.py ../config/text/diffusion_building_DIT.yaml uncond  --experiment_tag uncond --n_processes 0 --with_swanlab_logger
```


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
[`BuildingBlock/docs/stage2_with_agentic_readme.md`](BuildingBlock/docs/stage2_with_agentic_readme.md)
when reading from the repository root, or
[`docs/stage2_with_agentic_readme.md`](docs/stage2_with_agentic_readme.md)
inside the `BuildingBlock/` package directory.

### Stage2 code map

| Path | Purpose |
| --- | --- |
| `BuildingBlock/scene_synthesis/building_mesh/` | Core Stage2 layout-to-mesh, mesh assembly, visualization, reporting, and agent logic. |
| `BuildingBlock/scene_synthesis/building_mesh/layout_io.py` | Loads/parses Stage1 part-level layout records and geometry contracts. |
| `BuildingBlock/scene_synthesis/building_mesh/prompting.py` | Builds Stage2 prompts from part descriptions, bbox/layout context, and repair hints. |
| `BuildingBlock/scene_synthesis/building_mesh/reference_images.py` | T2I reference-image providers, including local Qwen-Image support and metadata writing. |
| `BuildingBlock/scene_synthesis/building_mesh/assembly.py` | Normalizes generated part meshes to layout bboxes and assembles the full building mesh. |
| `BuildingBlock/scene_synthesis/building_mesh/visualization.py` | S1-style layout multiview rendering, true mesh six-view rendering, and report assets. |
| `BuildingBlock/scene_synthesis/building_mesh/quality_agent.py` | Focused VLM critic inputs/outputs for part-image, part-mesh, and assembled-scene checks. |
| `BuildingBlock/scene_synthesis/building_mesh/s2_repair_agent.py` | Agent repair planning/execution: T2I prompt repair, geometry reruns, assemble-local fixes, iteration state. |
| `BuildingBlock/scene_synthesis/building_mesh/s2_agent_trace.py` | Trace schema/helpers for recording Agent steps, VLM calls, actions, and artifacts. |
| `BuildingBlock/scene_synthesis/building_mesh/report_bundle.py` | 9999 report generation/refresh, including Focused Visual Critic QA and Agent action cards. |
| `BuildingBlock/scripts/run_open_semantic_s2_batch.py` | Runs the open-semantic Stage2 batch pipeline. |
| `BuildingBlock/scripts/generate_part_reference_images.py` | Generates Stage2 part-level T2I reference images. |
| `BuildingBlock/scripts/hunyuan_bbox_single.py` | Runs single-part Hunyuan bbox-conditioned I23D generation. |
| `BuildingBlock/scripts/run_archstudio_s2_agent_loop.py` | Main Stage2 Agent loop entry point for real runs. |
| `BuildingBlock/scripts/run_archstudio_s2_focused_critics.py` | Runs focused VLM critics separately for debugging/evaluation. |
| `BuildingBlock/scripts/evaluate_archstudio_s2_quality.py` | Quality evaluation helper for Stage2 outputs. |
| `BuildingBlock/scripts/repair_archstudio_s2_parts.py` | Repair helper for selected Stage2 parts. |
| `BuildingBlock/scripts/rebuild_archstudio_s2_assemblies_from_manifest.py` | Rebuilds assembled meshes from manifests after part edits/repairs. |
| `BuildingBlock/scripts/texture_archstudio_s2_hunyuan_paint.py` | Texture-generation stage helper. |
| `BuildingBlock/scripts/tests/test_archstudio_s2_quality_agent.py` | Regression tests for focused critic / quality-agent behavior. |
| `BuildingBlock/scripts/tests/test_archstudio_s2_repair_agent.py` | Regression tests for Stage2 repair-agent behavior. |

## Project Structure

- `1-json_rotate_augment.py`: Data augmentation script
- `2-normUeJson.py`: JSON normalization script
- `3-json2boxnp.py`: JSON to NumPy conversion script
- `scripts/`: Contains model training and evaluation scripts
- `configs/`: Configuration files for different model settings
- `BoxCenterSizeLabel_all/`: Directory containing the dataset

## Notes

- Make sure your GPU drivers are properly configured for Docker GPU passthrough
- The dataset processing may take significant time depending on the size of the dataset
- Check the log files for any errors during processing

For more detailed information about the model architecture and training parameters, please refer to the documentation in the respective script files.
