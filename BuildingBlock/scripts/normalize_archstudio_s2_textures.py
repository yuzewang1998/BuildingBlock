#!/usr/bin/env python3
"""Create a texture-normalized ArchStudio-S2 texture run.

This is an additive post-pass for Hunyuan3D-Paint texture runs.  It does not
change geometry, UVs, layout, or part assembly membership.  It copies a source
texture run, rewrites diffuse maps with a generic architectural style
normalization policy, then rebuilds per-part GLBs, the combined OBJ/GLB, and the
HTML report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.texture_archstudio_s2_hunyuan_paint import (  # noqa: E402
    assemble_obj,
    export_glb,
    export_status_check_glb,
    mesh_bbox,
    write_texture_report,
)

DEFAULT_SOURCE_TEXTURE_RUN = Path(
    "/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2_texture/"
    "v9p2d_01_courtyard_museum_v9repair05_hy3dpaint"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2_texture/"
    "v9p2d_01_courtyard_museum_v9repair06_texture_normalized"
)

ARCH_STONE = np.asarray([0.66, 0.61, 0.53], dtype=np.float32)
ARCH_STONE_LIGHT = np.asarray([0.73, 0.69, 0.61], dtype=np.float32)
PAVING_STONE = np.asarray([0.62, 0.59, 0.52], dtype=np.float32)
SHADOW_GLASS = np.asarray([0.055, 0.068, 0.075], dtype=np.float32)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replace_path_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new) if value.startswith(old) else value
    if isinstance(value, list):
        return [replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_path_prefix(item, old, new) for key, item in value.items()}
    return value


def clamp01(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0, 1.0)


def seeded_noise(shape: tuple[int, int], seed_text: str) -> np.ndarray:
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed_text)) % (2**32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
    return noise


def normalized_gray_detail(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    gray_img = Image.fromarray(np.uint8(np.clip(gray * 255.0, 0, 255)), mode="L")
    low = np.asarray(gray_img.filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.float32) / 255.0
    # The Hunyuan atlas often contains object silhouettes or green/noisy panels.
    # Use only softened luminance as weak material variation; suppress strong atlas semantics.
    low_norm = (low - float(np.percentile(low, 5))) / max(float(np.percentile(low, 95) - np.percentile(low, 5)), 1e-6)
    low_norm = clamp01(low_norm)
    detail = gray - low
    detail = np.clip(detail, -0.12, 0.12)
    return low_norm, detail


def part_semantic_bucket(contract_part: dict[str, Any] | None, entry: dict[str, Any]) -> str:
    part = contract_part or {}
    semantic_role = str(part.get("semantic_role", "")).lower()
    legacy_hint = str(part.get("legacy_compatibility_hint", "")).lower()
    label_desc = " ".join(
        str(part.get(key, ""))
        for key in ("open_vocab_label", "part_description", "generation_prompt")
    ).lower()
    part_id = str(entry.get("part_id", "")).lower()
    text = " ".join([semantic_role, legacy_hint, label_desc, part_id])

    # Prefer S1's open semantic role / compatibility hint over incidental words in
    # the prompt.  This stays open-vocabulary while avoiding brittle matches like
    # "portal behind a column row" -> column or "parapet U footprint" -> paving.
    if "opening" in semantic_role or legacy_hint in {"opening", "window", "door", "portal"}:
        return "opening_shadow"
    if any(token in semantic_role for token in ("roofline", "parapet")) or legacy_hint in {"roof", "parapet", "rail"}:
        return "light_stone_detail"
    if semantic_role in {"structure"} or legacy_hint in {"column", "pillar", "support"}:
        return "light_stone_detail"
    if any(token in semantic_role for token in ("landscape", "circulation", "ground", "paving")) or legacy_hint in {"floor", "ground", "paving", "circulation", "stair"}:
        return "paving_circulation"

    if any(token in text for token in ("window", "portal", "slot", "aperture", "glass", "shadow")):
        return "opening_shadow"
    if any(token in text for token in ("column", "pillar", "support", "parapet", "rail", "roofline", "cap beam", "coping")):
        return "light_stone_detail"
    if any(token in text for token in ("paving", "ground", "floor", "courtyard", "plaza", "walkway", "stair", "step", "circulation")):
        return "paving_circulation"
    return "neutral_architecture"


def neutral_arch_texture(rgb: np.ndarray, base: np.ndarray, seed_text: str, *, strength: float) -> np.ndarray:
    low, detail = normalized_gray_detail(rgb)
    h, w = low.shape
    grain = seeded_noise((h, w), seed_text)
    grain_img = Image.fromarray(np.uint8(np.clip((grain - grain.min()) / max(grain.max() - grain.min(), 1e-6) * 255, 0, 255)), mode="L")
    grain_soft = np.asarray(grain_img.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0 - 0.5
    luminance = 0.78 + (low - 0.5) * strength + detail * 0.45 + grain_soft * 0.035
    out = base[None, None, :] * luminance[..., None]
    # Remove green cast generically by pulling chroma toward a warm neutral palette.
    out[..., 1] = np.minimum(out[..., 1], (out[..., 0] + out[..., 2]) * 0.54 + 0.08)
    return clamp01(out)


def opening_texture(rgb: np.ndarray, seed_text: str) -> np.ndarray:
    low, detail = normalized_gray_detail(rgb)
    h, w = low.shape
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    vertical_reflection = 0.05 * (np.sin(x * np.pi * 6.0 + (sum(map(ord, seed_text)) % 17)) + 1.0)
    top_shadow = 0.10 * (1.0 - y)
    luminance = 0.42 + low * 0.18 + detail * 0.25 + vertical_reflection + top_shadow
    out = SHADOW_GLASS[None, None, :] * luminance[..., None]
    # Subtle cool reflection, still visually reads as a dark void/glass opening.
    out[..., 2] += 0.025 * low
    out[..., 1] += 0.012 * low
    return clamp01(out)


def normalize_diffuse(path: Path, bucket: str, seed_text: str) -> dict[str, Any]:
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    before_mean = rgb.reshape(-1, 3).mean(axis=0)
    before_green_excess = float(np.maximum(rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]), 0.0).mean())

    if bucket == "opening_shadow":
        out = opening_texture(rgb, seed_text)
    elif bucket == "paving_circulation":
        out = neutral_arch_texture(rgb, PAVING_STONE, seed_text, strength=0.30)
    elif bucket == "light_stone_detail":
        out = neutral_arch_texture(rgb, ARCH_STONE_LIGHT, seed_text, strength=0.24)
    else:
        out = neutral_arch_texture(rgb, ARCH_STONE, seed_text, strength=0.26)

    after_mean = out.reshape(-1, 3).mean(axis=0)
    after_green_excess = float(np.maximum(out[..., 1] - np.maximum(out[..., 0], out[..., 2]), 0.0).mean())
    Image.fromarray(np.uint8(np.round(out * 255.0))).save(path, quality=94)
    return {
        "path": str(path),
        "bucket": bucket,
        "before_mean_rgb": [float(x) for x in before_mean],
        "after_mean_rgb": [float(x) for x in after_mean],
        "before_green_excess": before_green_excess,
        "after_green_excess": after_green_excess,
    }


def load_contract_by_part_id(source_run: Path) -> dict[str, dict[str, Any]]:
    candidates = [
        source_run / "contract" / "layout_mesh_contract.json",
        source_run / "layout_mesh_contract.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = load_json(candidate)
        return {str(part.get("part_id")): part for part in data.get("parts", []) if part.get("part_id")}
    return {}


def refresh_manifest_paths(manifest: dict[str, Any], old_dir: Path, new_dir: Path, baseline: Path | None) -> dict[str, Any]:
    refreshed = replace_path_prefix(manifest, str(old_dir), str(new_dir))
    refreshed["baseline_texture_run"] = str(baseline.resolve()) if baseline else str(old_dir.resolve())
    refreshed["texture_model"] = "Hunyuan3D-Paint-v2-1 + v9repair06 generic architectural texture normalization"
    refreshed["texture_normalization"] = {
        "version": "v9repair06",
        "policy": "geometry_preserving_diffuse_postprocess",
        "geometry_changed": False,
        "uv_changed": False,
        "style_goals": [
            "suppress green/noisy atlas cast",
            "unify stone/concrete/paving into warm neutral architectural palette",
            "make opening/void/glass-like parts read as dark shadow/glass",
        ],
        "semantic_basis": "open-vocabulary semantic_role/part_description buckets; no legacy class-specific geometry changes",
    }
    return refreshed


def rebuild_outputs(output_dir: Path, texture_manifest: dict[str, Any]) -> None:
    entries = texture_manifest.get("parts", [])
    for entry in entries:
        output_obj = Path(entry.get("output_obj", ""))
        if not output_obj.exists():
            continue
        output_glb = output_obj.with_suffix(".glb")
        glb_result = export_glb(output_obj, output_glb, viewer_frame=True)
        entry["output_glb"] = str(output_glb) if glb_result and not str(glb_result).startswith("GLB export failed") else ""
        entry["glb_warning"] = glb_result if str(glb_result).startswith("GLB export failed") else ""
        # Bbox should be unchanged; record it again as validation evidence.
        try:
            entry["output_bbox_after_texture_normalization"] = mesh_bbox(output_obj)
        except Exception as exc:  # noqa: BLE001
            entry["texture_normalization_bbox_warning"] = repr(exc)

    assembly_obj = output_dir / "assemblies" / "textured_or_fallback_assembly.obj"
    assemble_obj(entries, assembly_obj)
    assembly_glb_result = export_glb(assembly_obj, assembly_obj.with_suffix(".glb"), viewer_frame=True)
    status_check_glb_result = export_status_check_glb(entries, output_dir / "assemblies" / "textured_or_fallback_status_check.glb")
    texture_manifest["assembly_obj"] = str(assembly_obj)
    texture_manifest["assembly_glb"] = (
        str(assembly_obj.with_suffix(".glb"))
        if assembly_glb_result and not str(assembly_glb_result).startswith("GLB export failed")
        else ""
    )
    texture_manifest["assembly_glb_warning"] = (
        assembly_glb_result if str(assembly_glb_result).startswith("GLB export failed") else ""
    )
    texture_manifest["assembly_status_check_glb"] = (
        str(output_dir / "assemblies" / "textured_or_fallback_status_check.glb")
        if status_check_glb_result and not str(status_check_glb_result).startswith("Status-check GLB export failed")
        else ""
    )
    texture_manifest["assembly_status_check_glb_warning"] = (
        status_check_glb_result if str(status_check_glb_result).startswith("Status-check GLB export failed") else ""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-texture-run", type=Path, default=DEFAULT_SOURCE_TEXTURE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-texture-run", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_texture_run = args.source_texture_run.resolve()
    output_dir = args.output_dir.resolve()
    if not (source_texture_run / "texture_manifest.json").exists():
        raise FileNotFoundError(source_texture_run / "texture_manifest.json")
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"output exists: {output_dir}; pass --force to replace it")
        shutil.rmtree(output_dir)
    shutil.copytree(source_texture_run, output_dir)

    texture_manifest = load_json(output_dir / "texture_manifest.json")
    texture_manifest = refresh_manifest_paths(texture_manifest, source_texture_run, output_dir, args.baseline_texture_run)
    source_run = Path(texture_manifest["source_run"])
    contract_by_part = load_contract_by_part_id(source_run)

    normalization_records: list[dict[str, Any]] = []
    for entry in texture_manifest.get("parts", []):
        part_id = str(entry.get("part_id", ""))
        output_obj = Path(entry.get("output_obj", ""))
        diffuse = output_obj.with_suffix(".jpg")
        if not diffuse.exists():
            entry["texture_normalization_status"] = "missing_diffuse"
            continue
        bucket = part_semantic_bucket(contract_by_part.get(part_id), entry)
        record = normalize_diffuse(diffuse, bucket, part_id)
        entry["texture_normalization_status"] = "applied"
        entry["texture_normalization_bucket"] = bucket
        entry["texture_normalized_diffuse"] = str(diffuse)
        normalization_records.append({"part_id": part_id, **record})

    texture_manifest["num_texture_normalized"] = len(normalization_records)
    texture_manifest["texture_normalization_records"] = normalization_records
    texture_manifest["num_success"] = sum(1 for item in texture_manifest.get("parts", []) if str(item.get("status", "")).startswith("succeeded"))
    texture_manifest["num_failures"] = len([item for item in texture_manifest.get("parts", []) if str(item.get("status", "")).startswith("failed")])
    texture_manifest["num_basic_material"] = sum(
        1 for item in texture_manifest.get("parts", []) if str(item.get("status", "")).startswith("succeeded_basic_material")
    )
    texture_manifest["num_wall_skipped"] = sum(1 for item in texture_manifest.get("parts", []) if item.get("status") == "skipped_wall_untextured")

    rebuild_outputs(output_dir, texture_manifest)
    failures = [item for item in texture_manifest.get("parts", []) if str(item.get("status", "")).startswith("failed")]
    write_json(output_dir / "texture_manifest.json", texture_manifest)
    write_json(output_dir / "texture_failures.json", failures)
    write_json(output_dir / "texture_normalization_summary.json", {
        "source_texture_run": str(source_texture_run),
        "output_dir": str(output_dir),
        "num_texture_normalized": len(normalization_records),
        "records": normalization_records,
    })
    write_texture_report(output_dir, texture_manifest, failures)
    print(json.dumps({
        "output_dir": str(output_dir),
        "num_parts": texture_manifest.get("num_parts"),
        "num_success": texture_manifest.get("num_success"),
        "num_failures": texture_manifest.get("num_failures"),
        "num_texture_normalized": len(normalization_records),
        "assembly_glb": texture_manifest.get("assembly_glb"),
        "report": str(output_dir / "report" / "index.html"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
