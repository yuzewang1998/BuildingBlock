# ArchStudio-S2 local Qwen-Image T2I

ArchStudio-S2 now supports local Qwen-Image for part reference generation.
This avoids SiliconFlow/API balance issues while keeping Hunyuan-Omni geometry
and Hunyuan3D-Paint texture stages unchanged.

## Environment

- Conda env: `/home/wangyz/anaconda3/envs/archstudio_qwen_image`
- Python: `/home/wangyz/anaconda3/envs/archstudio_qwen_image/bin/python`
- Model: `Qwen/Qwen-Image`
- HF cache: `/mnt/data/wangyz/huggingface/hub/models--Qwen--Qwen-Image`
- HF mirror endpoint: `https://hf-mirror.com`

The smoke test output is:

```text
/mnt/data/wangyz/BuildingBlock/outputs/qwen_image_local_smoke/reference.png
```

A pipeline-level one-part smoke test output is:

```text
/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2/v9p2d_01_courtyard_museum_local_qwen_onepart
```

The metadata should show:

```json
{
  "provider": "qwen_image_local",
  "status": "succeeded",
  "used_fallback": false
}
```

## Run one S2 scene with local Qwen-Image

```bash
python scripts/run_open_semantic_s2_batch.py \
  --case 01 \
  --run-suffix _local_qwen_v1 \
  --provider qwen_image_local \
  --t2i-python /home/wangyz/anaconda3/envs/archstudio_qwen_image/bin/python \
  --t2i-gpu 0 \
  --hf-home /mnt/data/wangyz/huggingface \
  --hf-endpoint https://hf-mirror.com \
  --gpu 1 \
  --steps 24 \
  --guidance-scale 4.0 \
  --force-t2i \
  --force-geometry \
  --force-report
```

## Run all 10 open-semantic scenes

```bash
python scripts/archstudio_s2_run_all_partfocus.py \
  --run-suffix _local_qwen_v1 \
  --provider qwen_image_local \
  --t2i-python /home/wangyz/anaconda3/envs/archstudio_qwen_image/bin/python \
  --t2i-gpu-list 0,1,2,3 \
  --gpu-list 4,5,6,0 \
  --max-parallel 4 \
  --steps 24 \
  --guidance-scale 4.0 \
  --force-t2i \
  --force-geometry \
  --force-report
```

## Notes

- `--provider` now defaults to `qwen_image_local` in the open-semantic S2 scripts.
- `--fallback-provider` now defaults to empty, so local T2I failures are visible instead of silently producing procedural placeholders.
- Add `--fallback-provider procedural_reference` only when continuity is more important than visual fidelity.
- Local Qwen-Image uses the dedicated T2I Python env only for reference image generation; geometry and texture environments are unchanged.
