#!/usr/bin/env python3
"""Additive ArchStudio-S2 texture stage using Hunyuan3D-Paint-v2-1.

This script consumes an existing ArchStudio-S2 geometry run. It never calls or
changes the layout/T2I/Hunyuan-Omni geometry pipeline; it only reads normalized
part meshes plus their reference images, writes textured part OBJ assets, and
emits a separate texture manifest/assembly.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


DEFAULT_PAINT_REPO = Path("/home/wangyz/project/0working/Hunyuan3D-2.1")
DEFAULT_SOURCE_RUN = Path(
    "/mnt/data/wangyz/BuildingBlock/outputs/part_level_t2i/"
    "real_hunyuan_v8_house0067_prompt_antibase"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/data/wangyz/BuildingBlock/outputs/part_level_texture/"
    "archstudio_s2_v8_hy3dpaint"
)

LAYOUT_TO_VIEWER_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mesh_bbox(path: Path) -> dict[str, Any]:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        return {
            "vertex_count": 0,
            "face_count": int(len(mesh.faces)),
            "min": [],
            "max": [],
            "extent": [],
        }
    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(mesh.faces)),
        "min": min_xyz.tolist(),
        "max": max_xyz.tolist(),
        "extent": (max_xyz - min_xyz).tolist(),
    }


def load_source_contract(source_run: Path) -> dict[str, Any]:
    """Load the layout contract beside a geometry run when available."""
    candidates = [
        source_run / "contract" / "layout_mesh_contract.json",
        source_run / "input" / "layout_mesh_contract.original.json",
        source_run / "layout_mesh_contract.json",
    ]
    manifest_path = source_run / "manifest.json"
    try:
        manifest = load_json(manifest_path)
        for item in manifest.get("parts", []):
            contract_path = item.get("contract_path")
            if contract_path:
                candidates.append(Path(contract_path))
    except Exception:
        pass
    for candidate in candidates:
        try:
            if candidate and Path(candidate).exists():
                payload = load_json(Path(candidate))
                if isinstance(payload, dict) and payload.get("parts"):
                    return payload
        except Exception:
            continue
    return {"parts": []}


def load_source_agent_trace(source_run: Path) -> dict[str, Any] | None:
    trace_path = source_run / "agent_trace" / "s2_agent_trace.json"
    if not trace_path.exists():
        return None
    try:
        return load_json(trace_path)
    except Exception:
        return None


def contract_part_lookup(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(part.get("part_id")): part
        for part in contract.get("parts", [])
        if part.get("part_id")
    }


def valid_reference_image_path(path: Path | str | None) -> Path | None:
    """Return a usable image file path, never ``Path("") == "."``.

    Geometry manifests from older runs may omit ``reference_image_path``.  Using
    ``Path(value or "")`` turns missing values into the current directory, and
    a directory passes ``exists()`` even though Hunyuan Paint expects an image
    file.  Keep this validation generic and only accept concrete files.
    """
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_file():
        return candidate
    return None


def resolve_part_reference_image(
    manifest_part: dict[str, Any],
    contract_part: dict[str, Any] | None,
) -> Path | None:
    for value in (
        manifest_part.get("reference_image_path"),
        manifest_part.get("reference_image"),
    ):
        resolved = valid_reference_image_path(value)
        if resolved is not None:
            return resolved

    if contract_part:
        for value in (
            contract_part.get("reference_image_path"),
            contract_part.get("reference_image"),
        ):
            resolved = valid_reference_image_path(value)
            if resolved is not None:
                return resolved
    return None


def infer_scene_center(contract: dict[str, Any]) -> np.ndarray:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for part in contract.get("parts", []):
        bbox = part.get("bbox") or {}
        center = bbox.get("center")
        size = bbox.get("size")
        if not center or not size or len(center) != 3 or len(size) != 3:
            continue
        c = np.asarray([float(value) for value in center], dtype=np.float64)
        s = np.asarray([float(value) for value in size], dtype=np.float64)
        mins.append(c - s * 0.5)
        maxs.append(c + s * 0.5)
    if not mins:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    return (np.vstack(mins).min(axis=0) + np.vstack(maxs).max(axis=0)) * 0.5


def bbox_from_contract_part(part: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    bbox = part.get("bbox") or {}
    center = bbox.get("center")
    size = bbox.get("size")
    if not center or not size or len(center) != 3 or len(size) != 3:
        return None
    return (
        np.asarray([float(value) for value in center], dtype=np.float64),
        np.asarray([float(value) for value in size], dtype=np.float64),
    )


def infer_part_frame_axes(
    contract_part: dict[str, Any] | None,
    input_bbox: dict[str, Any],
    scene_center: np.ndarray,
) -> dict[str, Any] | None:
    """Infer a generic outward/up/local frame from bbox geometry.

    The 2D reference image has a strong semantic frame: image +V is
    architectural up and the visible face is outward-facing.  This function
    maps that image frame to layout axes without class/name special cases:
    vertical is always layout +Z, the facade normal is the thinnest horizontal
    bbox axis with sign pointing away from scene center, and the image
    horizontal axis is the remaining horizontal axis.
    """
    if contract_part:
        parsed = bbox_from_contract_part(contract_part)
    else:
        parsed = None
    if parsed is not None:
        center, size = parsed
    else:
        mn = np.asarray(input_bbox.get("min", []), dtype=np.float64)
        mx = np.asarray(input_bbox.get("max", []), dtype=np.float64)
        if mn.shape != (3,) or mx.shape != (3,):
            return None
        center = (mn + mx) * 0.5
        size = mx - mn
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        return None

    horizontal_axes = [0, 1]
    normal_axis = min(horizontal_axes, key=lambda axis: float(size[axis]))
    u_axis = 1 - normal_axis
    z_axis = 2
    sign = 1.0 if float(center[normal_axis] - scene_center[normal_axis]) >= 0.0 else -1.0
    outward = np.zeros(3, dtype=np.float64)
    outward[normal_axis] = sign
    u = np.zeros(3, dtype=np.float64)
    u[u_axis] = 1.0
    v = np.zeros(3, dtype=np.float64)
    v[z_axis] = 1.0
    # Keep a right-handed local basis where U x V = N.  This can flip the image
    # horizontal axis for one side of the building, but it preserves image up
    # and outward-facing normals without any class/name rules.
    if float(np.dot(np.cross(u, v), outward)) < 0:
        u = -u
    return {
        "u_axis": u,
        "v_axis": v,
        "n_axis": outward,
        "normal_axis_index": int(normal_axis),
        "horizontal_axis_index": int(u_axis),
        "outward_sign": int(sign),
        "scene_center": scene_center.tolist(),
        "part_center": center.tolist(),
        "part_size": size.tolist(),
        "policy": "bbox_thinnest_horizontal_axis_outward_from_scene_center_z_up",
    }


def transform_matrix_from_basis(columns: list[np.ndarray], origin: np.ndarray | None = None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack(columns)
    if origin is not None:
        matrix[:3, 3] = origin
    return matrix


def layout_to_paint_canonical_transform(frame: dict[str, Any]) -> np.ndarray:
    """Return layout->paint input transform for the inferred part-local frame.

    Hunyuan Paint internally converts its input coordinates with
    ``internal = [-x, z, -y]``.  To make Paint's internal frame see
    image-horizontal as +X, outward normal as +Y, and image-up as +Z, the file
    we give Paint must use: input-x = -U, input-y = -V, input-z = N.
    """
    u = np.asarray(frame["u_axis"], dtype=np.float64)
    v = np.asarray(frame["v_axis"], dtype=np.float64)
    n = np.asarray(frame["n_axis"], dtype=np.float64)
    canonical_from_layout = np.column_stack([-u, -v, n]).T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = canonical_from_layout
    return matrix


def transform_obj_geometry(source_obj: Path, target_obj: Path, transform: np.ndarray) -> None:
    """Apply a vertex-only OBJ transform while preserving faces/UV/material lines."""
    target_obj.parent.mkdir(parents=True, exist_ok=True)
    source_lines = source_obj.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten: list[str] = []
    for raw in source_lines:
        if raw.startswith("v "):
            parts = raw.strip().split()
            if len(parts) >= 4:
                xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3]), 1.0], dtype=np.float64)
                mapped = transform @ xyz
                suffix = " ".join(parts[4:])
                line = f"v {mapped[0]:.9f} {mapped[1]:.9f} {mapped[2]:.9f}"
                if suffix:
                    line += " " + suffix
                rewritten.append(line)
                continue
        rewritten.append(raw)
    target_obj.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def prepare_paint_frame_input(
    input_obj: Path,
    output_obj: Path,
    frame: dict[str, Any] | None,
) -> dict[str, Any]:
    if frame is None:
        shutil.copy2(input_obj, output_obj)
        return {
            "paint_frame_alignment": "disabled_no_contract_frame",
            "paint_frame_policy": "identity",
        }
    transform = layout_to_paint_canonical_transform(frame)
    transform_obj_geometry(input_obj, output_obj, transform)
    return {
        "paint_frame_alignment": "layout_to_paint_canonical",
        "paint_frame_policy": frame["policy"],
        "paint_frame_normal_axis_index": frame["normal_axis_index"],
        "paint_frame_horizontal_axis_index": frame["horizontal_axis_index"],
        "paint_frame_outward_sign": frame["outward_sign"],
        "paint_frame_layout_to_input": transform.tolist(),
    }


def restore_paint_frame_output(
    painted_obj: Path,
    output_obj: Path,
    frame_info: dict[str, Any],
) -> None:
    transform_payload = frame_info.get("paint_frame_layout_to_input")
    if not transform_payload:
        if painted_obj.resolve() != output_obj.resolve():
            copy_textured_obj_asset(painted_obj, output_obj)
        return
    transform = np.asarray(transform_payload, dtype=np.float64)
    inverse = np.linalg.inv(transform)
    copy_info = {}
    if painted_obj.resolve() != output_obj.resolve():
        copy_info = copy_textured_obj_asset(painted_obj, output_obj)
    elif not output_obj.exists():
        raise FileNotFoundError(f"Missing painted OBJ to restore: {painted_obj}")
    transform_obj_geometry(output_obj, output_obj, inverse)
    frame_info.update(copy_info)


def copy_with_suffix_stem(source_path: Path, target_path: Path) -> None:
    if source_path.exists() and source_path.resolve() != target_path.resolve():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def canonical_paint_asset_path(output_obj: Path) -> Path:
    return output_obj.with_name(output_obj.stem + "__paint_canonical.obj")


def finalize_canonical_paint_output(canonical_obj: Path, output_obj: Path, frame_info: dict[str, Any]) -> None:
    """Restore canonical-frame Hunyuan Paint output to layout frame and final names."""
    restore_paint_frame_output(canonical_obj, output_obj, frame_info)
    canonical_stem = canonical_obj.stem
    output_stem = output_obj.stem
    for suffix in [".mtl", ".jpg", "_metallic.jpg", "_roughness.jpg", ".png", "_metallic.png", "_roughness.png"]:
        source_asset = canonical_obj.with_name(f"{canonical_stem}{suffix}")
        target_asset = output_obj.with_name(f"{output_stem}{suffix}")
        copy_with_suffix_stem(source_asset, target_asset)
    if output_obj.with_suffix(".mtl").exists():
        mtl_text = output_obj.with_suffix(".mtl").read_text(encoding="utf-8", errors="ignore")
        output_obj.with_suffix(".mtl").write_text(mtl_text.replace(canonical_stem, output_stem), encoding="utf-8")
    if output_obj.exists():
        obj_text = output_obj.read_text(encoding="utf-8", errors="ignore")
        output_obj.write_text(obj_text.replace(canonical_stem, output_stem), encoding="utf-8")


def to_viewer_frame(loaded):
    if isinstance(loaded, trimesh.Scene):
        scene = trimesh.Scene()
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph.get(node_name)
            mesh = loaded.geometry[geom_name].copy()
            mesh.apply_transform(transform)
            mesh.apply_transform(LAYOUT_TO_VIEWER_TRANSFORM)
            scene.add_geometry(mesh, geom_name=str(geom_name), node_name=str(node_name))
        return scene
    mesh = loaded.copy()
    mesh.apply_transform(LAYOUT_TO_VIEWER_TRANSFORM)
    return mesh


def export_glb(source_obj: Path, target_glb: Path, *, viewer_frame: bool = False) -> str | None:
    """Export an OBJ/MTL textured mesh to GLB with trimesh when possible."""
    try:
        target_glb.parent.mkdir(parents=True, exist_ok=True)
        loaded = trimesh.load(str(source_obj), process=False)
        if viewer_frame:
            loaded = to_viewer_frame(loaded)
        loaded.export(str(target_glb))
        if target_glb.exists() and target_glb.stat().st_size > 0:
            return str(target_glb)
    except Exception as exc:  # noqa: BLE001 - surfaced in manifest as warning.
        return f"GLB export failed: {exc!r}"
    return None


def max_abs_delta(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return float("inf")
    return float(max(abs(float(a) - float(b)) for a, b in zip(left, right)))


def geometry_delta(input_bbox: dict[str, Any], output_bbox: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_max_abs_delta": max_abs_delta(input_bbox.get("min", []), output_bbox.get("min", [])),
        "max_max_abs_delta": max_abs_delta(input_bbox.get("max", []), output_bbox.get("max", [])),
        "extent_max_abs_delta": max_abs_delta(input_bbox.get("extent", []), output_bbox.get("extent", [])),
        "face_count_delta": int(output_bbox.get("face_count", 0)) - int(input_bbox.get("face_count", 0)),
        "vertex_count_delta": int(output_bbox.get("vertex_count", 0)) - int(input_bbox.get("vertex_count", 0)),
    }


def bbox_preserved(delta: dict[str, Any], tolerance: float) -> bool:
    return (
        delta["min_max_abs_delta"] <= tolerance
        and delta["max_max_abs_delta"] <= tolerance
        and delta["extent_max_abs_delta"] <= tolerance
    )


def geometry_preserved(delta: dict[str, Any], tolerance: float, *, allow_topology_change: bool = False) -> bool:
    if not bbox_preserved(delta, tolerance):
        return False
    if allow_topology_change:
        return True
    return delta["face_count_delta"] == 0 and delta["vertex_count_delta"] == 0


def surface_preserved(delta: dict[str, Any], tolerance: float, *, allow_topology_change: bool = False) -> bool:
    """Return whether the painted asset preserves the spatial surface envelope.

    Hunyuan Paint often duplicates vertices during UV unwrap while keeping the
    face count and bbox unchanged.  That is not a geometric failure for our
    current research stage; it is a texture-coordinate representation detail.
    """
    if not bbox_preserved(delta, tolerance):
        return False
    if allow_topology_change:
        return True
    return delta["face_count_delta"] == 0


def is_success_status(status: str) -> bool:
    return status.startswith("succeeded")


def remeshed_paint_input_required(face_count: int, args: argparse.Namespace) -> bool:
    """Return whether this part should use Hunyuan Paint's remesh pre-pass.

    This is not a basic-material fallback.  The part still goes through
    Hunyuan3D-Paint; remeshing is only a model-side compatibility step for very
    dense Hunyuan-Omni meshes that otherwise exceed practical Paint rasterizer
    limits.
    """
    return (
        bool(args.force_hunyuan_paint_dense)
        and args.hunyuan_remesh_face_threshold is not None
        and face_count > args.hunyuan_remesh_face_threshold
    )


def average_reference_color(reference_image: Path, default: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        from PIL import Image

        image = Image.open(reference_image).convert("RGBA").resize((256, 256))
        rgba = np.asarray(image).astype(np.float32)
        alpha = rgba[..., 3] > 20
        rgb = rgba[..., :3][alpha] if alpha.any() else rgba[..., :3].reshape(-1, 3)
        mean = np.clip(rgb.mean(axis=0) / 255.0, 0.05, 0.95)
        return tuple(float(x) for x in mean)
    except Exception:
        return default


def write_solid_texture(texture_path: Path, color: tuple[float, float, float]) -> None:
    try:
        from PIL import Image

        rgb = tuple(int(np.clip(c, 0.0, 1.0) * 255) for c in color)
        image = Image.new("RGB", (1024, 1024), rgb)
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(texture_path, quality=95)
    except Exception:
        texture_path.write_bytes(b"")


def write_tiled_wall_texture(texture_path: Path, color: tuple[float, float, float]) -> None:
    """Create a repeatable, generic wall texture for placeholder wall geometry."""
    try:
        from PIL import Image, ImageDraw, ImageFilter

        rng = np.random.default_rng(42)
        size = 1024
        base = np.array([int(np.clip(c, 0.0, 1.0) * 255) for c in color], dtype=np.int16)
        noise = rng.normal(0, 5, (size, size, 1)).astype(np.int16)
        pixels = np.clip(base.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        draw = ImageDraw.Draw(image, "RGBA")

        mortar = tuple(int(max(c * 255 * 0.72, 30)) for c in color) + (150,)
        highlight = tuple(int(min(c * 255 * 1.12, 255)) for c in color) + (80,)
        brick_h = 86
        brick_w = 176
        mortar_w = 5
        for y in range(0, size + brick_h, brick_h):
            draw.rectangle([0, y - mortar_w // 2, size, y + mortar_w // 2], fill=mortar)
            offset = 0 if (y // brick_h) % 2 == 0 else brick_w // 2
            for x in range(-offset, size + brick_w, brick_w):
                draw.rectangle([x, y, x + mortar_w, y + brick_h], fill=mortar)
                draw.line([x + 10, y + 10, x + brick_w - 18, y + 10], fill=highlight, width=2)

        # A little blur keeps the texture architectural rather than cartoonish.
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(texture_path, quality=96)
    except Exception:
        write_solid_texture(texture_path, color)


def parse_obj_vertices_faces(path: Path) -> tuple[list[list[float]], list[list[int]], list[str]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    comments: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as src:
        for raw in src:
            if raw.startswith("#"):
                comments.append(raw.rstrip())
                continue
            if raw.startswith("v "):
                vertices.append([float(x) for x in raw.split()[1:4]])
                continue
            if raw.startswith("f "):
                face: list[int] = []
                for token in raw.split()[1:]:
                    vertex_token = token.split("/")[0]
                    if vertex_token:
                        face.append(int(vertex_token))
                if len(face) >= 3:
                    faces.append(face)
    return vertices, faces, comments


def box_vertices_faces_from_bbox(
    vertices: list[list[float]],
    min_thickness: float = 0.0,
) -> tuple[list[list[float]], list[list[int]], bool]:
    points = np.asarray(vertices, dtype=np.float64)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) * 0.5
    half = (mx - mn) * 0.5
    adjusted = False
    if min_thickness > 0:
        min_half = float(min_thickness) * 0.5
        for axis in range(3):
            if half[axis] < min_half:
                half[axis] = min_half
                adjusted = True
    x0, y0, z0 = (center - half).tolist()
    x1, y1, z1 = (center + half).tolist()
    box_vertices = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]
    box_faces = [
        [1, 2, 3, 4],
        [5, 8, 7, 6],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 8, 4],
        [4, 8, 5, 1],
    ]
    return box_vertices, box_faces, adjusted


def dominant_axis_uv_regions(raw_paint_output_obj: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return UV atlas regions used by Hunyuan Paint for each dominant face axis.

    The raw Hunyuan Paint wall output is a subdivided same-bbox box with a real
    atlas.  If we remap its generated texture onto a stable bbox with generic
    repeated UVs, the wall often looks like a flat color because each large bbox
    face samples broad low-frequency atlas areas.  Reusing the atlas region for
    the corresponding face orientation preserves the model-generated variation
    while keeping the final geometry simple and robust.
    """
    if not raw_paint_output_obj.exists():
        return {}
    loaded = trimesh.load(str(raw_paint_output_obj), force="mesh", process=False)
    uv = getattr(loaded.visual, "uv", None)
    if uv is None or len(loaded.faces) == 0:
        return {}
    normals = np.asarray(loaded.face_normals)
    faces = np.asarray(loaded.faces)
    regions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for axis in range(3):
        face_indices = np.where(np.argmax(np.abs(normals), axis=1) == axis)[0]
        if len(face_indices) == 0:
            continue
        face_uv = np.asarray(uv)[faces[face_indices].reshape(-1)]
        uv_min = face_uv.min(axis=0)
        uv_max = face_uv.max(axis=0)
        if np.all(np.isfinite(uv_min)) and np.all(np.isfinite(uv_max)):
            regions[axis] = (uv_min, uv_max)
    return regions


def copy_textured_obj_asset(source_obj: Path, target_obj: Path) -> dict[str, Any]:
    """Copy a Hunyuan Paint OBJ/MTL/texture bundle under the target OBJ stem."""
    source_obj = Path(source_obj)
    target_obj = Path(target_obj)
    target_obj.parent.mkdir(parents=True, exist_ok=True)
    source_stem = source_obj.stem
    target_stem = target_obj.stem

    source_text = source_obj.read_text(encoding="utf-8", errors="ignore")
    target_text = source_text.replace(source_stem, target_stem)
    target_obj.write_text(target_text, encoding="utf-8")

    copied_maps: list[str] = []
    source_mtl = source_obj.with_suffix(".mtl")
    if source_mtl.exists():
        target_mtl = target_obj.with_suffix(".mtl")
        mtl_text = source_mtl.read_text(encoding="utf-8", errors="ignore")
        target_mtl.write_text(mtl_text.replace(source_stem, target_stem), encoding="utf-8")
        for suffix in [".jpg", "_metallic.jpg", "_roughness.jpg", ".png", "_metallic.png", "_roughness.png"]:
            source_texture = source_obj.with_name(f"{source_stem}{suffix}")
            if source_texture.exists():
                target_texture = target_obj.with_name(f"{target_stem}{suffix}")
                shutil.copy2(source_texture, target_texture)
                copied_maps.append(str(target_texture))
    return {"copied_hunyuan_maps": copied_maps}


def orient_obj_faces_outward(obj_path: Path) -> dict[str, Any]:
    """Flip OBJ face winding so all face normals point away from the mesh bbox center.

    Hunyuan Paint preserves the wall input topology, including its face winding.
    The subdivided wall paint input can contain opposite box sides with the same
    winding, so half the wall faces point inward and model-viewer may only show
    those faces from inside.  This repair keeps vertex/UV/material data intact
    and only reverses face token order when a face normal points toward the
    bbox center.
    """
    obj_path = Path(obj_path)
    lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    vertices: list[np.ndarray] = []
    for raw in lines:
        if raw.startswith("v "):
            parts = raw.split()
            if len(parts) >= 4:
                vertices.append(np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64))
    if not vertices:
        return {"outward_faces_flipped": 0, "outward_faces_checked": 0}
    points = np.asarray(vertices, dtype=np.float64)
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5

    def vertex_index(token: str) -> int | None:
        value = token.split("/")[0]
        if not value:
            return None
        index = int(value)
        if index < 0:
            index = len(vertices) + index + 1
        return index - 1

    rewritten: list[str] = []
    flipped = 0
    checked = 0
    for raw in lines:
        if not raw.startswith("f "):
            rewritten.append(raw)
            continue
        tokens = raw.split()[1:]
        indices = [vertex_index(token) for token in tokens]
        if len(tokens) < 3 or any(index is None or index < 0 or index >= len(vertices) for index in indices):
            rewritten.append(raw)
            continue
        face_points = [vertices[int(index)] for index in indices]
        normal = None
        for second in range(1, len(face_points) - 1):
            candidate = np.cross(face_points[second] - face_points[0], face_points[second + 1] - face_points[0])
            if np.linalg.norm(candidate) > 1e-12:
                normal = candidate
                break
        if normal is None:
            rewritten.append(raw)
            continue
        face_center = np.mean(np.asarray(face_points, dtype=np.float64), axis=0)
        checked += 1
        if float(np.dot(normal, face_center - center)) < 0:
            tokens = list(reversed(tokens))
            flipped += 1
        rewritten.append("f " + " ".join(tokens))
    obj_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return {"outward_faces_flipped": flipped, "outward_faces_checked": checked}


def obj_index_to_zero(raw_index: str, count: int) -> int | None:
    if not raw_index:
        return None
    index = int(raw_index)
    if index < 0:
        index = count + index + 1
    index -= 1
    if index < 0 or index >= count:
        return None
    return index


def stable_axis_sign(axis: np.ndarray) -> np.ndarray:
    """Give a tangent axis a stable global sign for cross-part UV consistency."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return axis
    axis = axis / norm
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    return axis


def face_world_uv_axes(face_points: list[np.ndarray], horizontal_normal_threshold: float) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Return generic world-space UV axes for a face without using part classes.

    The convention is intentionally architectural but class-agnostic:
    - Near-horizontal faces use layout XY as the texture frame.
    - Other faces use world Z as texture V, so vertical material patterns keep
      the same up direction across independent parts.
    """
    if len(face_points) < 3:
        return None
    normal = None
    for second in range(1, len(face_points) - 1):
        candidate = np.cross(face_points[second] - face_points[0], face_points[second + 1] - face_points[0])
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-12:
            normal = candidate / norm
            break
    if normal is None:
        return None

    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, world_up))) >= horizontal_normal_threshold:
        return (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "horizontal_xy",
        )

    u_axis = np.cross(world_up, normal)
    if float(np.linalg.norm(u_axis)) <= 1e-12:
        u_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    u_axis = stable_axis_sign(u_axis)
    return (u_axis, world_up, "vertical_world_z")


def should_apply_world_axis_uv(output_bbox: dict[str, Any], args: argparse.Namespace) -> bool:
    mode = getattr(args, "world_axis_uv_normalization", "off")
    if mode == "off":
        return False
    if mode == "all":
        return True
    extents = [float(value) for value in output_bbox.get("extent", []) if float(value) > 1e-8]
    if not extents:
        return False
    thin_ratio = min(extents) / max(extents)
    return thin_ratio <= float(args.world_axis_uv_planar_thin_ratio)


def normalize_obj_uv_world_axes(obj_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Rewrite OBJ texture coordinates using a generic layout/world axis frame.

    This is a post-Hunyuan-Paint UV normalization pass.  It does not look at
    semantic labels (wall/roof/window/etc.) and it does not edit geometry or
    generated texture pixels.  It only replaces per-face UVs so directional
    material patterns share a consistent world-up orientation across parts.
    """
    obj_path = Path(obj_path)
    lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    vertices: list[np.ndarray] = []
    for raw in lines:
        if raw.startswith("v "):
            parts = raw.split()
            if len(parts) >= 4:
                vertices.append(np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64))
    if not vertices:
        return {
            "world_axis_uv_applied": False,
            "world_axis_uv_reason": "no vertices",
            "world_axis_uv_faces": 0,
        }

    points = np.asarray(vertices, dtype=np.float64)
    origin = points.min(axis=0)
    tile_size = max(float(args.world_axis_uv_tile_size), 1e-8)

    def parse_token(token: str) -> tuple[int | None, str, str]:
        chunks = token.split("/")
        vertex_index = obj_index_to_zero(chunks[0] if chunks else "", len(vertices))
        texture_chunk = chunks[1] if len(chunks) > 1 else ""
        normal_chunk = chunks[2] if len(chunks) > 2 else ""
        return vertex_index, texture_chunk, normal_chunk

    rewritten_body: list[str] = []
    generated_vt: list[str] = []
    faces_rewritten = 0
    faces_skipped = 0
    vertical_faces = 0
    horizontal_faces = 0
    vt_index = 1
    first_face_line = None

    for raw in lines:
        if raw.startswith("vt "):
            continue
        if not raw.startswith("f "):
            rewritten_body.append(raw)
            continue
        if first_face_line is None:
            first_face_line = len(rewritten_body)
        tokens = raw.split()[1:]
        parsed = [parse_token(token) for token in tokens]
        vertex_indices = [item[0] for item in parsed]
        if len(tokens) < 3 or any(index is None for index in vertex_indices):
            rewritten_body.append(raw)
            faces_skipped += 1
            continue

        face_points = [vertices[int(index)] for index in vertex_indices]
        axes = face_world_uv_axes(face_points, float(args.world_axis_uv_horizontal_normal_threshold))
        if axes is None:
            rewritten_body.append(raw)
            faces_skipped += 1
            continue
        u_axis, v_axis, frame_name = axes
        if frame_name == "horizontal_xy":
            horizontal_faces += 1
        else:
            vertical_faces += 1

        rewritten_tokens: list[str] = []
        for token, (vertex_index, _texture_chunk, normal_chunk) in zip(tokens, parsed):
            position = vertices[int(vertex_index)]
            relative = position - origin
            u = float(np.dot(relative, u_axis) / tile_size)
            v = float(np.dot(relative, v_axis) / tile_size)
            generated_vt.append(f"vt {u:.6f} {v:.6f}")
            vertex_raw = token.split("/")[0]
            if normal_chunk:
                rewritten_tokens.append(f"{vertex_raw}/{vt_index}/{normal_chunk}")
            else:
                rewritten_tokens.append(f"{vertex_raw}/{vt_index}")
            vt_index += 1
        rewritten_body.append("f " + " ".join(rewritten_tokens))
        faces_rewritten += 1

    if first_face_line is None or faces_rewritten == 0:
        return {
            "world_axis_uv_applied": False,
            "world_axis_uv_reason": "no rewriteable faces",
            "world_axis_uv_faces": faces_rewritten,
            "world_axis_uv_faces_skipped": faces_skipped,
        }

    rewritten = (
        rewritten_body[:first_face_line]
        + generated_vt
        + rewritten_body[first_face_line:]
    )
    obj_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return {
        "world_axis_uv_applied": True,
        "world_axis_uv_mode": args.world_axis_uv_normalization,
        "world_axis_uv_tile_size": tile_size,
        "world_axis_uv_faces": faces_rewritten,
        "world_axis_uv_faces_skipped": faces_skipped,
        "world_axis_uv_vertical_faces": vertical_faces,
        "world_axis_uv_horizontal_faces": horizontal_faces,
        "world_axis_uv_generated_vt": len(generated_vt),
        "world_axis_uv_policy": "generic_geometry_world_axis_projection_no_class_rules",
    }


def export_subdivided_box_like_mesh(input_obj: Path, output_obj: Path, divisions: int = 24) -> None:
    """Create a same-bounds subdivided box for wall painting.

    Hunyuan Paint can return a black atlas for the ultra-low-poly wall
    placeholder. This keeps the wall shape and bbox unchanged while giving the
    renderer/projector enough surface samples to paint onto.
    """
    vertices, _faces, comments = parse_obj_vertices_faces(input_obj)
    points = np.asarray(vertices, dtype=np.float64)
    if points.size == 0:
        shutil.copy2(input_obj, output_obj)
        return
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    x0, y0, z0 = mn.tolist()
    x1, y1, z1 = mx.tolist()
    d = max(int(divisions), 1)

    out_vertices: list[tuple[float, float, float]] = []
    out_faces: list[tuple[int, int, int]] = []

    def add_face(origin, u_vec, v_vec):
        start = len(out_vertices) + 1
        for j in range(d + 1):
            for i in range(d + 1):
                u = i / d
                v = j / d
                p = np.asarray(origin) + u * np.asarray(u_vec) + v * np.asarray(v_vec)
                out_vertices.append((float(p[0]), float(p[1]), float(p[2])))
        for j in range(d):
            for i in range(d):
                a = start + j * (d + 1) + i
                b = a + 1
                c = a + (d + 1)
                e = c + 1
                out_faces.append((a, b, e))
                out_faces.append((a, e, c))

    # The orientation is not critical for texture painting; use duplicated
    # per-face vertices to avoid UV/normal sharing issues at box edges.
    add_face((x0, y0, z0), (x1 - x0, 0, 0), (0, y1 - y0, 0))  # bottom/front-like
    add_face((x0, y0, z1), (x1 - x0, 0, 0), (0, y1 - y0, 0))  # top/back-like
    add_face((x0, y0, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0))
    add_face((x0, y1, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0))
    add_face((x0, y0, z0), (0, y1 - y0, 0), (0, 0, z1 - z0))
    add_face((x1, y0, z0), (0, y1 - y0, 0), (0, 0, z1 - z0))

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_obj.open("w", encoding="utf-8") as handle:
        handle.write("# subdivided wall paint input generated from placeholder\n")
        for comment in comments[:3]:
            handle.write(comment + "\n")
        for v in out_vertices:
            handle.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for f in out_faces:
            handle.write(f"f {f[0]} {f[1]} {f[2]}\n")


def export_tiled_wall_material_obj(
    input_obj: Path,
    output_obj: Path,
    reference_image: Path,
    material_name: str,
    default_color: tuple[float, float, float] = (0.74, 0.69, 0.60),
    tile_size: float = 0.075,
) -> None:
    """Preserve wall placeholder geometry and add tiled UV texture coordinates."""
    vertices, faces, comments = parse_obj_vertices_faces(input_obj)
    if not vertices or not faces:
        export_basic_material_obj(input_obj, output_obj, reference_image, material_name, default_color)
        return

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    safe_material = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in material_name)
    texture_path = output_obj.with_suffix(".jpg")
    color = average_reference_color(reference_image, default_color)
    write_tiled_wall_texture(texture_path, color)

    mtl_path = output_obj.with_suffix(".mtl")
    mtl_path.write_text(
        "\n".join(
            [
                f"newmtl {safe_material}",
                f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}",
                f"Ka {max(color[0] * 0.22, 0.02):.6f} {max(color[1] * 0.22, 0.02):.6f} {max(color[2] * 0.22, 0.02):.6f}",
                "Ks 0.02 0.02 0.02",
                "Ns 12.0",
                "d 1.0",
                "illum 2",
                f"map_Kd {texture_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    points = np.asarray(vertices, dtype=np.float64)
    mins = points.min(axis=0)
    uv_lines: list[str] = []
    face_lines: list[str] = []
    vt_index = 1
    for face in faces:
        face_points = np.asarray([vertices[i - 1] for i in face], dtype=np.float64)
        spans = np.ptp(face_points, axis=0)
        axes = list(np.argsort(spans)[-2:])
        if len(axes) < 2 or spans[axes[0]] == 0 and spans[axes[1]] == 0:
            axes = [0, 2]
        axes = sorted(axes)
        uv_indices: list[int] = []
        for vertex_index in face:
            vertex = np.asarray(vertices[vertex_index - 1], dtype=np.float64)
            u = (vertex[axes[0]] - mins[axes[0]]) / max(tile_size, 1e-6)
            v = (vertex[axes[1]] - mins[axes[1]]) / max(tile_size, 1e-6)
            uv_lines.append(f"vt {u:.6f} {v:.6f}")
            uv_indices.append(vt_index)
            vt_index += 1
        face_lines.append(
            "f " + " ".join(f"{vertex_index}/{uv_index}" for vertex_index, uv_index in zip(face, uv_indices))
        )

    with output_obj.open("w", encoding="utf-8") as dst:
        dst.write(f"mtllib {mtl_path.name}\n")
        for comment in comments:
            dst.write(comment + "\n")
        for vertex in vertices:
            dst.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for line in uv_lines:
            dst.write(line + "\n")
        dst.write(f"usemtl {safe_material}\n")
        for line in face_lines:
            dst.write(line + "\n")


def _box_face_outward_axis(face: list[int], vertices: list[list[float]], center: np.ndarray) -> int:
    points = np.asarray([vertices[i - 1] for i in face], dtype=np.float64)
    direction = points.mean(axis=0) - center
    if not np.any(np.isfinite(direction)):
        return 1
    return int(np.argmax(np.abs(direction)))


def detect_wall_facade_axis(vertices: list[list[float]]) -> int:
    """Pick the layout axis normal for the externally visible wall face.

    The current civic-gate layouts place the principal facades on the shallow
    wall thickness axis.  This rule remains geometric rather than class-name
    specific: for a bbox wall, the exterior elevation face is the face whose
    normal follows the smallest bbox extent.  The generated facade texture is
    projected once onto those two outside faces; other box sides get muted UVs.
    """
    points = np.asarray(vertices, dtype=np.float64)
    extents = points.max(axis=0) - points.min(axis=0)
    return int(np.argmin(extents))


def export_box_with_generated_texture_obj(
    input_obj: Path,
    output_obj: Path,
    generated_texture: Path,
    material_name: str,
    tile_size: float = 0.075,
    min_thickness: float = 0.0,
    uv_regions: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    facade_projection: bool = False,
) -> dict[str, Any]:
    """Use a generated image map on a stable bbox wall box.

    For wall placeholders, ``facade_projection`` maps the generated image once
    onto the detected exterior elevation faces instead of repeating a tiny
    material tile.  This keeps the bbox geometry stable while allowing rich,
    non-mechanical facade texture on the visible wall face.
    """
    vertices, faces, comments = parse_obj_vertices_faces(input_obj)
    if not vertices or not faces:
        shutil.copy2(input_obj, output_obj)
        return {"display_bbox_adjusted": False, "min_display_thickness": min_thickness}
    if not generated_texture.exists():
        raise FileNotFoundError(f"Missing generated Hunyuan Paint texture: {generated_texture}")
    vertices, faces, adjusted = box_vertices_faces_from_bbox(vertices, min_thickness=min_thickness)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    safe_material = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in material_name)
    target_texture = output_obj.with_suffix(".jpg")
    if generated_texture.resolve() != target_texture.resolve():
        shutil.copy2(generated_texture, target_texture)

    mtl_path = output_obj.with_suffix(".mtl")
    mtl_path.write_text(
        "\n".join(
            [
                f"newmtl {safe_material}",
                "Kd 0.85 0.85 0.85",
                "Ka 0.15 0.15 0.15",
                "Ks 0.02 0.02 0.02",
                "Ns 16.0",
                "d 1.0",
                "illum 2",
                f"map_Kd {target_texture.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    points = np.asarray(vertices, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    facade_axis = detect_wall_facade_axis(vertices) if facade_projection else None
    facade_face_count = 0
    muted_side_face_count = 0
    uv_lines: list[str] = []
    face_lines: list[str] = []
    vt_index = 1
    for face in faces:
        if len(face) < 3:
            continue
        face_axis = _box_face_outward_axis(face, vertices, center)
        use_facade_uv = facade_axis is not None and face_axis == facade_axis
        if use_facade_uv:
            facade_face_count += 1
        else:
            muted_side_face_count += 1
        triangles = [
            [face[0], face[index], face[index + 1]]
            for index in range(1, len(face) - 1)
        ]
        for triangle in triangles:
            face_points = np.asarray([vertices[i - 1] for i in triangle], dtype=np.float64)
            spans = np.ptp(face_points, axis=0)
            normal_axis = int(np.argmin(spans))
            axes = sorted(list(np.argsort(spans)[-2:]))
            uv_indices: list[int] = []
            for vertex_index in triangle:
                vertex = np.asarray(vertices[vertex_index - 1], dtype=np.float64)
                if use_facade_uv:
                    span_u = max(maxs[axes[0]] - mins[axes[0]], 1e-9)
                    span_v = max(maxs[axes[1]] - mins[axes[1]], 1e-9)
                    u = float((vertex[axes[0]] - mins[axes[0]]) / span_u)
                    v = float((vertex[axes[1]] - mins[axes[1]]) / span_v)
                elif uv_regions and normal_axis in uv_regions:
                    uv_min, uv_max = uv_regions[normal_axis]
                    span_u = max(maxs[axes[0]] - mins[axes[0]], 1e-9)
                    span_v = max(maxs[axes[1]] - mins[axes[1]], 1e-9)
                    local_u = (vertex[axes[0]] - mins[axes[0]]) / span_u
                    local_v = (vertex[axes[1]] - mins[axes[1]]) / span_v
                    u = float(uv_min[0] + local_u * (uv_max[0] - uv_min[0]))
                    v = float(uv_min[1] + local_v * (uv_max[1] - uv_min[1]))
                else:
                    # Side/top/bottom faces use a small, stable crop of the same
                    # generated image, avoiding obvious repetitive tiling while
                    # keeping visual continuity with the facade map.
                    span_u = max(maxs[axes[0]] - mins[axes[0]], 1e-9)
                    span_v = max(maxs[axes[1]] - mins[axes[1]], 1e-9)
                    local_u = (vertex[axes[0]] - mins[axes[0]]) / span_u
                    local_v = (vertex[axes[1]] - mins[axes[1]]) / span_v
                    u = float(0.04 + 0.18 * local_u)
                    v = float(0.04 + 0.18 * local_v)
                uv_lines.append(f"vt {u:.6f} {v:.6f}")
                uv_indices.append(vt_index)
                vt_index += 1
            face_lines.append(
                "f " + " ".join(f"{vertex_index}/{uv_index}" for vertex_index, uv_index in zip(triangle, uv_indices))
            )
            face_lines.append(
                "f "
                + " ".join(
                    f"{vertex_index}/{uv_index}"
                    for vertex_index, uv_index in reversed(list(zip(triangle, uv_indices)))
                )
            )

    with output_obj.open("w", encoding="utf-8") as dst:
        dst.write(f"mtllib {mtl_path.name}\n")
        for comment in comments:
            dst.write(comment + "\n")
        dst.write("# stable wall bbox mesh using generated facade texture projection\n")
        for vertex in vertices:
            dst.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for line in uv_lines:
            dst.write(line + "\n")
        dst.write(f"usemtl {safe_material}\n")
        for line in face_lines:
            dst.write(line + "\n")
    return {
        "display_bbox_adjusted": adjusted,
        "min_display_thickness": min_thickness,
        "facade_projection": bool(facade_projection),
        "facade_axis": int(facade_axis) if facade_axis is not None else None,
        "facade_box_faces": facade_face_count,
        "muted_side_box_faces": muted_side_face_count,
    }


def export_basic_material_obj(
    input_obj: Path,
    output_obj: Path,
    reference_image: Path,
    material_name: str,
    default_color: tuple[float, float, float] = (0.72, 0.68, 0.60),
) -> None:
    """Preserve geometry and add a simple generated material.

    This is the robust texture fallback for wall placeholders and meshes that
    are too dense for Hunyuan3D-Paint direct inference. It is intentionally not
    marked as a failure: the geometry is preserved and a basic diffuse material
    is emitted so the assembly stays complete and browsable.
    """
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    safe_material = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in material_name)
    texture_path = output_obj.with_suffix(".jpg")
    color = average_reference_color(reference_image, default_color)
    write_solid_texture(texture_path, color)

    mtl_path = output_obj.with_suffix(".mtl")
    mtl_path.write_text(
        "\n".join(
            [
                f"newmtl {safe_material}",
                f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}",
                f"Ka {max(color[0] * 0.25, 0.02):.6f} {max(color[1] * 0.25, 0.02):.6f} {max(color[2] * 0.25, 0.02):.6f}",
                "Ks 0.05 0.05 0.05",
                "Ns 20.0",
                "d 1.0",
                "illum 2",
                f"map_Kd {texture_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    inserted_material = False
    with input_obj.open("r", encoding="utf-8", errors="ignore") as src, output_obj.open("w", encoding="utf-8") as dst:
        dst.write(f"mtllib {mtl_path.name}\n")
        for raw in src:
            if raw.startswith("mtllib "):
                continue
            if raw.startswith("usemtl "):
                if not inserted_material:
                    dst.write(f"usemtl {safe_material}\n")
                    inserted_material = True
                continue
            if raw.startswith("f ") and not inserted_material:
                dst.write(f"usemtl {safe_material}\n")
                inserted_material = True
            dst.write(raw)


def stable_part_color(part_id: str, status: str, target_class: str) -> list[int]:
    if target_class == "wall":
        return [68, 170, 255, 210]
    if "basic_material" in status:
        return [255, 164, 56, 255]
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(part_id)) % 360
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(seed / 360.0, 0.62, 0.94)
    return [int(r * 255), int(g * 255), int(b * 255), 255]


def export_status_check_glb(entries: list[dict[str, Any]], target_glb: Path) -> str | None:
    """Export a visual QA GLB with all parts color-coded by status.

    The textured GLB is the deliverable. This status-check GLB is only for the
    report: it makes wall/fallback/basic parts visually obvious and provides a
    second browser surface when texture materials or browser caches obscure the
    assembled content.
    """
    try:
        scene = trimesh.Scene()
        for entry in entries:
            mesh_path = Path(entry.get("output_obj") or entry.get("fallback_obj") or "")
            if not mesh_path.exists():
                continue
            loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            else:
                mesh = loaded
            color = stable_part_color(
                str(entry.get("part_id", mesh_path.stem)),
                str(entry.get("status", "")),
                str(entry.get("target_class", "")),
            )
            mesh = mesh.copy()
            mesh.apply_transform(LAYOUT_TO_VIEWER_TRANSFORM)
            mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile(color, (len(mesh.vertices), 1)))
            scene.add_geometry(mesh, geom_name=str(entry.get("part_id", mesh_path.stem)))
        target_glb.parent.mkdir(parents=True, exist_ok=True)
        scene.export(str(target_glb))
        if target_glb.exists() and target_glb.stat().st_size > 0:
            return str(target_glb)
    except Exception as exc:  # noqa: BLE001 - surfaced in manifest as warning.
        return f"Status-check GLB export failed: {exc!r}"
    return None


def prepare_paint_imports(paint_repo: Path) -> None:
    hy3dpaint = paint_repo / "hy3dpaint"
    sys.path.insert(0, str(hy3dpaint))
    os.chdir(str(paint_repo))
    try:
        from utils.torchvision_fix import apply_fix

        apply_fix()
    except Exception as exc:
        print(f"torchvision_fix skipped: {exc}")


def build_paint_pipeline(paint_repo: Path, max_num_view: int, resolution: int):
    prepare_paint_imports(paint_repo)
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    hy3dpaint = paint_repo / "hy3dpaint"
    config = Hunyuan3DPaintConfig(max_num_view=max_num_view, resolution=resolution)
    config.multiview_cfg_path = str(hy3dpaint / "cfgs" / "hunyuan-paint-pbr.yaml")
    config.realesrgan_ckpt_path = str(hy3dpaint / "ckpt" / "RealESRGAN_x4plus.pth")
    return Hunyuan3DPaintPipeline(config)


def selected_parts(parts: list[dict[str, Any]], part_ids: set[str], limit: int | None) -> list[dict[str, Any]]:
    chosen = []
    for part in parts:
        if part_ids and part.get("part_id") not in part_ids:
            continue
        chosen.append(part)
        if limit is not None and len(chosen) >= limit:
            break
    return chosen


def rewrite_face_token(token: str, offsets: dict[str, int]) -> str:
    chunks = token.split("/")
    if chunks[0]:
        chunks[0] = str(int(chunks[0]) + offsets["v"])
    if len(chunks) > 1 and chunks[1]:
        chunks[1] = str(int(chunks[1]) + offsets["vt"])
    if len(chunks) > 2 and chunks[2]:
        chunks[2] = str(int(chunks[2]) + offsets["vn"])
    return "/".join(chunks)


def parse_mtl_for_assembly(source_mtl: Path, material_prefix: str, assembly_dir: Path) -> tuple[list[str], dict[str, str]]:
    if not source_mtl.exists():
        return [], {}
    rewritten_lines: list[str] = []
    material_map: dict[str, str] = {}
    for raw in source_mtl.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            rewritten_lines.append(raw)
            continue
        if line.startswith("newmtl "):
            old_name = line.split(maxsplit=1)[1].strip()
            new_name = f"{material_prefix}_{old_name}"
            material_map[old_name] = new_name
            rewritten_lines.append(f"newmtl {new_name}")
            continue
        parts = line.split(maxsplit=1)
        if parts and parts[0].startswith("map_") and len(parts) == 2:
            texture_name = parts[1].split()[-1]
            source_texture = source_mtl.parent / texture_name
            if source_texture.exists():
                texture_key = f"{material_prefix}/{source_texture.name}".encode("utf-8", errors="ignore")
                texture_hash = hashlib.sha1(texture_key).hexdigest()[:12]
                target_texture = assembly_dir / f"tex_{texture_hash}{source_texture.suffix}"
                # Always refresh copied texture assets.  Wall texture iteration
                # can rewrite the same final diffuse filename after quality
                # fallback; keeping an older assembly-local copy makes the GLB
                # appear flat even when the part OBJ/MTL is correct.
                shutil.copy2(source_texture, target_texture)
                rewritten_lines.append(raw.replace(texture_name, target_texture.name))
                continue
        rewritten_lines.append(raw)
    return rewritten_lines, material_map


def append_obj_to_assembly(
    source_obj: Path,
    out_obj,
    out_mtl_lines: list[str],
    assembly_dir: Path,
    part_name: str,
    offsets: dict[str, int],
    textured: bool,
) -> None:
    material_prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in part_name)
    material_map: dict[str, str] = {}
    default_material = f"{material_prefix}_untextured"
    if textured:
        source_mtl = source_obj.with_suffix(".mtl")
        mtl_lines, material_map = parse_mtl_for_assembly(source_mtl, material_prefix, assembly_dir)
        out_mtl_lines.extend(mtl_lines)
        out_mtl_lines.append("")
    else:
        out_mtl_lines.extend(
            [
                f"newmtl {default_material}",
                "Kd 0.75 0.75 0.75",
                "Ka 0.15 0.15 0.15",
                "Ks 0.0 0.0 0.0",
                "d 1.0",
                "illum 2",
                "",
            ]
        )

    out_obj.write(f"o {part_name}\n")
    if not textured:
        out_obj.write(f"usemtl {default_material}\n")

    local_counts = {"v": 0, "vt": 0, "vn": 0}
    for raw in source_obj.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("mtllib "):
            continue
        if raw.startswith("o ") or raw.startswith("g "):
            continue
        if raw.startswith("usemtl "):
            old_material = raw.split(maxsplit=1)[1].strip()
            out_obj.write(f"usemtl {material_map.get(old_material, old_material)}\n")
            continue
        if raw.startswith("v "):
            local_counts["v"] += 1
            out_obj.write(raw + "\n")
            continue
        if raw.startswith("vt "):
            local_counts["vt"] += 1
            out_obj.write(raw + "\n")
            continue
        if raw.startswith("vn "):
            local_counts["vn"] += 1
            out_obj.write(raw + "\n")
            continue
        if raw.startswith("f "):
            face = raw.split()[1:]
            out_obj.write("f " + " ".join(rewrite_face_token(token, offsets) for token in face) + "\n")
            continue
        out_obj.write(raw + "\n")

    offsets["v"] += local_counts["v"]
    offsets["vt"] += local_counts["vt"]
    offsets["vn"] += local_counts["vn"]


def assemble_obj(entries: list[dict[str, Any]], assembly_obj: Path) -> None:
    assembly_obj.parent.mkdir(parents=True, exist_ok=True)
    assembly_mtl = assembly_obj.with_suffix(".mtl")
    mtl_lines: list[str] = []
    offsets = {"v": 0, "vt": 0, "vn": 0}
    with assembly_obj.open("w", encoding="utf-8") as out_obj:
        out_obj.write(f"mtllib {assembly_mtl.name}\n")
        for entry in entries:
            mesh_path = Path(entry.get("output_obj") or entry.get("fallback_obj") or "")
            if not mesh_path.exists():
                continue
            append_obj_to_assembly(
                mesh_path,
                out_obj,
                mtl_lines,
                assembly_obj.parent,
                entry.get("part_id", mesh_path.stem),
                offsets,
                textured=is_success_status(str(entry.get("status", ""))),
            )
    assembly_mtl.write_text("\n".join(mtl_lines), encoding="utf-8")


def raw_asset_url(path: str | Path) -> str:
    if not path:
        return ""
    resolved = Path(path).resolve()
    encoded = base64.urlsafe_b64encode(str(resolved).encode("utf-8")).decode("ascii")
    url = f"/projects/buildingblock/raw/{encoded.rstrip('=')}"
    if resolved.exists():
        stat = resolved.stat()
        url += f"?v={stat.st_mtime_ns}-{stat.st_size}"
    return url


def scene_asset_stats(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "geometry_count": 0, "node_count": 0, "names": []}
    try:
        loaded = trimesh.load(str(p), process=False)
        if isinstance(loaded, trimesh.Scene):
            return {
                "exists": True,
                "geometry_count": len(loaded.geometry),
                "node_count": len(loaded.graph.nodes_geometry),
                "names": sorted(str(name) for name in loaded.geometry.keys()),
                "bounds": loaded.bounds.tolist() if loaded.bounds is not None else [],
            }
        return {
            "exists": True,
            "geometry_count": 1,
            "node_count": 1,
            "names": [p.stem],
            "bounds": loaded.bounds.tolist() if loaded.bounds is not None else [],
        }
    except Exception as exc:  # noqa: BLE001 - report should survive validation failures.
        return {"exists": True, "geometry_count": 0, "node_count": 0, "names": [], "error": repr(exc)}


def existing_asset(root: Path, name: str) -> Path | None:
    path = root / name
    return path if path.exists() else None


def image_figure(path: Path | None, caption: str, extra_class: str = "") -> str:
    if not path:
        return ""
    return (
        f"<figure class='{html.escape(extra_class)}'>"
        f"<img src='{html.escape(raw_asset_url(path))}' alt='{html.escape(caption)}' />"
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def model_figure(path: Path | None, caption: str, note: str = "") -> str:
    if not path:
        return ""
    return f"""
      <figure class="model-card">
        <model-viewer src="{html.escape(raw_asset_url(path))}" camera-controls auto-rotate
          interaction-prompt="none" exposure="1.08" shadow-intensity="0.75"></model-viewer>
        <figcaption><b>{html.escape(caption)}</b>{f"<br><span>{html.escape(note)}</span>" if note else ""}</figcaption>
      </figure>
    """


def section_if_any(title: str, body: str, description: str = "") -> str:
    if not body.strip():
        return ""
    return f"""
      <section class="summary">
        <h2>{html.escape(title)}</h2>
        {f"<p>{html.escape(description)}</p>" if description else ""}
        {body}
      </section>
    """


def write_overall_sections(texture_manifest: dict[str, Any]) -> str:
    """Reuse the source geometry report assets so texture reports keep V7/V8 context."""
    source_run = Path(texture_manifest["source_run"])
    source_assets = source_run / "report" / "assets"

    layout_model = existing_asset(source_assets, "layout_3d_boxes.glb")
    complete_model = existing_asset(source_assets, "assembly_complete_per_part_color_yup.glb")
    # At the current ArchStudio-S2 stage, "raw" should still include wall
    # placeholders. Later optimization/remeshing stages may split raw and
    # complete assemblies again, but for now the wall-inclusive complete asset
    # is the correct geometry context for both browser slots.
    raw_model = complete_model or existing_asset(source_assets, "assembly_raw_per_part_color_yup.glb")
    textured_model_raw = texture_manifest.get("assembly_glb", "")
    textured_model = Path(textured_model_raw) if textured_model_raw else None
    if textured_model and not textured_model.exists():
        textured_model = None
    status_model_raw = texture_manifest.get("assembly_status_check_glb", "")
    status_model = Path(status_model_raw) if status_model_raw else None
    if status_model and not status_model.exists():
        status_model = None

    source_report = source_run / "report" / "index.html"
    strict_generated = texture_manifest.get("num_basic_material", 0) == 0 and texture_manifest.get("num_failures", 0) == 0
    source_report_link = (
        f"<p><a href='{html.escape(raw_asset_url(source_report))}'>Open source geometry report</a></p>"
        if source_report.exists()
        else ""
    )
    agent_trace = texture_manifest.get("source_agent_trace") or {}
    trace_summary = agent_trace.get("summary", {}) if isinstance(agent_trace, dict) else {}
    trace_policy = agent_trace.get("agent_policy", {}) if isinstance(agent_trace, dict) else {}
    trace_rows = []
    if isinstance(agent_trace, dict):
        for part in agent_trace.get("parts", [])[:80]:
            t2i = part.get("t2i", {}) or {}
            geometry = part.get("geometry", {}) or {}
            texture_record = next(
                (
                    item for item in texture_manifest.get("parts", [])
                    if item.get("part_id") == part.get("part_id")
                ),
                {},
            )
            trace_rows.append(
                "<tr>"
                f"<td>{html.escape(str(part.get('label', '')))}</td>"
                f"<td>{html.escape(str(part.get('part_description_core') or part.get('part_description') or ''))}</td>"
                f"<td>{html.escape(str(part.get('semantic_role', '')))}</td>"
                f"<td>{html.escape(str(t2i.get('status', '')))}</td>"
                f"<td>{html.escape(str(geometry.get('status', '')))}</td>"
                f"<td>{html.escape(str(texture_record.get('status', 'not_run')))}</td>"
                "</tr>"
            )
    trace_section = section_if_any(
        "ArchStudio-S2 Agent Trace",
        (
            "<p class='small'>当前 agent 是可审计的单次流程 agent：逐 part 记录 T2I → Hunyuan-Omni geometry → texture 的状态、prompt core/context 拆分和失败原因。后续 evaluator/retry agent 会在这个 trace 上继续扩展。</p>"
            f"<p><b>Policy:</b> {html.escape(str(trace_policy.get('version', 'n/a')))} · "
            f"Parts={html.escape(str(trace_summary.get('num_parts', texture_manifest.get('num_parts', ''))))} · "
            f"T2I ok={html.escape(str(trace_summary.get('num_t2i_succeeded', 'n/a')))} · "
            f"Geometry ok={html.escape(str(trace_summary.get('num_geometry_succeeded', 'n/a')))} · "
            f"Texture ok={html.escape(str(texture_manifest.get('num_success', 'n/a')))}</p>"
            f"<p><a href='{html.escape(raw_asset_url(texture_manifest.get('source_agent_trace_path', '')))}'>source s2_agent_trace.json</a></p>"
            "<details open><summary>Per-part pipeline status</summary>"
            "<table><thead><tr><th>#</th><th>core subject</th><th>role</th><th>T2I</th><th>geometry</th><th>texture</th></tr></thead>"
            f"<tbody>{''.join(trace_rows) if trace_rows else '<tr><td colspan=\"6\">No source trace yet.</td></tr>'}</tbody></table></details>"
        ),
        "和 Stage1 类似，这里不是只给最终结果，而是把每个 part 的决策/状态展开。",
    )
    baseline_section = ""
    baseline_run_raw = texture_manifest.get("baseline_texture_run", "")
    if baseline_run_raw:
        baseline_run = Path(baseline_run_raw)
        baseline_manifest_path = baseline_run / "texture_manifest.json"
        baseline_report_path = baseline_run / "report" / "index.html"
        baseline_glb = None
        baseline_summary = {}
        if baseline_manifest_path.exists():
            try:
                baseline_manifest = load_json(baseline_manifest_path)
                baseline_summary = {
                    "num_parts": baseline_manifest.get("num_parts"),
                    "num_success": baseline_manifest.get("num_success"),
                    "num_failures": baseline_manifest.get("num_failures"),
                    "num_basic_material": baseline_manifest.get("num_basic_material"),
                    "assembly_glb": baseline_manifest.get("assembly_glb", ""),
                }
                if baseline_summary.get("assembly_glb"):
                    baseline_glb = Path(str(baseline_summary["assembly_glb"]))
            except Exception as exc:  # noqa: BLE001 - report should still render.
                baseline_summary = {"error": repr(exc)}
        baseline_gallery = "".join(
            [
                model_figure(baseline_glb if baseline_glb and baseline_glb.exists() else None, "Baseline textured assembly", "上一版 strict 结果，用于视觉对比。"),
                model_figure(textured_model, "Final pre-agent textured assembly", "本次最终非 agent 优化版。"),
            ]
        )
        baseline_section = section_if_any(
            "Before / final comparison",
            (
                f"<div class='model-grid'>{baseline_gallery}</div>"
                f"<p><b>Baseline:</b> {html.escape(str(baseline_run))}</p>"
                f"<pre>{html.escape(json.dumps(baseline_summary, ensure_ascii=False, indent=2))}</pre>"
                + (
                    f"<p><a href='{html.escape(raw_asset_url(baseline_report_path))}'>Open baseline report</a></p>"
                    if baseline_report_path.exists()
                    else ""
                )
            ),
            "用于确认本次 prompt / cleanup / strict texture 的整体变化，而不是只看单个 part。",
        )

    model_gallery = "".join(
        [
            model_figure(layout_model, "Large 3D layout boxes", "原始 layout 的整体 3D bbox，可交互浏览。"),
            model_figure(raw_model, "Raw geometry assembly", "当前阶段 raw 也包含 wall placeholder；后续 optimize 后再与 complete 区分。"),
            model_figure(complete_model, "Complete geometry assembly", "wall placeholder + generated part 的完整几何拼装。"),
            model_figure(textured_model, "Strict generated textured assembly" if strict_generated else "Textured / fallback assembly", "所有 part 均走 Hunyuan3D-Paint 生成贴图，无 basic material fallback。" if strict_generated else "成功部件贴图，失败/跳过部件回退原几何。"),
            model_figure(status_model, "Strict texture status-check assembly" if strict_generated else "Textured / fallback status-check assembly", "状态着色检查版：用于确认每个 part 都进入 assembly。" if strict_generated else "状态着色检查版：wall=蓝色，basic/fallback=橙色。"),
        ]
    )
    textured_stats = scene_asset_stats(textured_model) if textured_model else {"geometry_count": 0, "names": []}
    status_stats = scene_asset_stats(status_model) if status_model else {"geometry_count": 0, "names": []}
    fallback_rows = []
    uv_rows = []
    for item in texture_manifest.get("parts", []):
        status = str(item.get("status", ""))
        if "basic_material" in status or status != "succeeded":
            fallback_rows.append(
                "<tr>"
                f"<td>{html.escape(item.get('target_class', ''))}</td>"
                f"<td>{html.escape(item.get('part_id', ''))}</td>"
                f"<td>{html.escape(status)}</td>"
                f"<td>{html.escape(str(item.get('geometry_preserved', 'n/a')))}</td>"
                f"<td>{html.escape(item.get('note') or item.get('reason') or '')}</td>"
                "</tr>"
            )
        if item.get("world_axis_uv_applied"):
            uv_rows.append(
                "<tr>"
                f"<td>{html.escape(item.get('target_class', ''))}</td>"
                f"<td>{html.escape(item.get('part_id', ''))}</td>"
                f"<td>{html.escape(str(item.get('world_axis_uv_faces', '')))}</td>"
                f"<td>{html.escape(str(item.get('world_axis_uv_vertical_faces', '')))}</td>"
                f"<td>{html.escape(str(item.get('world_axis_uv_horizontal_faces', '')))}</td>"
                f"<td>{html.escape(str(item.get('world_axis_uv_tile_size', '')))}</td>"
                "</tr>"
            )
    validation_body = f"""
      <div class="validation">
        <p><b>World-axis UV normalization:</b>
          mode={html.escape(str(texture_manifest.get('world_axis_uv_normalization', 'off')))} ·
          applied_parts={html.escape(str(texture_manifest.get('num_world_axis_uv_applied', 0)))} ·
          tile_size={html.escape(str(texture_manifest.get('world_axis_uv_tile_size', '')))}
        </p>
        <p><b>Textured GLB validation:</b>
          geometry_count={html.escape(str(textured_stats.get('geometry_count', 0)))} ·
          node_count={html.escape(str(textured_stats.get('node_count', 0)))} ·
          expected_parts={html.escape(str(texture_manifest.get('num_parts', '')))}
        </p>
        <p><b>Status-check GLB validation:</b>
          geometry_count={html.escape(str(status_stats.get('geometry_count', 0)))} ·
          node_count={html.escape(str(status_stats.get('node_count', 0)))} ·
          expected_parts={html.escape(str(texture_manifest.get('num_parts', '')))}
        </p>
        <details open>
          <summary>{"Warnings / non-succeeded entries" if strict_generated else "Fallback/basic/wall entries included in assembly"}</summary>
          <table>
            <thead><tr><th>class</th><th>part_id</th><th>status</th><th>geometry preserved</th><th>note/reason</th></tr></thead>
            <tbody>{''.join(fallback_rows) if fallback_rows else '<tr><td colspan="5">No fallback/basic entries; strict generated texture path succeeded for all parts.</td></tr>'}</tbody>
          </table>
        </details>
        <details>
          <summary>Geometry names inside textured GLB</summary>
          <pre>{html.escape(json.dumps(textured_stats.get('names', []), ensure_ascii=False, indent=2))}</pre>
        </details>
        <details open>
          <summary>Generic world-axis UV normalized parts</summary>
          <table>
            <thead><tr><th>class</th><th>part_id</th><th>faces</th><th>vertical faces</th><th>horizontal faces</th><th>tile size</th></tr></thead>
            <tbody>{''.join(uv_rows) if uv_rows else '<tr><td colspan="6">World-axis UV normalization disabled or no planar parts selected.</td></tr>'}</tbody>
          </table>
        </details>
      </div>
    """

    layout_gallery = "".join(
        image_figure(existing_asset(source_assets, name), caption)
        for name, caption in [
            ("layout_overview.png", "Layout overview"),
            ("layout_text_callouts_xz.png", "Layout + part text callouts"),
            ("layout_3d_boxes.png", "Layout 3D bbox render"),
            ("layout_3d_view_iso.png", "Layout 3D bbox · ISO"),
            ("layout_3d_view_top.png", "Layout 3D bbox · top"),
            ("layout_3d_view_front.png", "Layout 3D bbox · front"),
            ("layout_3d_view_side.png", "Layout 3D bbox · side"),
            ("layout_xz.png", "Layout XZ elevation"),
            ("layout_yz.png", "Layout YZ elevation"),
        ]
    )

    assembly_gallery = "".join(
        image_figure(existing_asset(source_assets, name), caption)
        for name, caption in [
            ("assembly_complete_view_iso.png", "Raw assembly with wall · ISO"),
            ("assembly_complete_view_top.png", "Raw assembly with wall · top"),
            ("assembly_complete_view_front.png", "Raw assembly with wall · front"),
            ("assembly_complete_view_side.png", "Raw assembly with wall · side"),
            ("assembly_complete_view_iso.png", "Complete assembly · ISO"),
            ("assembly_complete_view_top.png", "Complete assembly · top"),
            ("assembly_complete_view_front.png", "Complete assembly · front"),
            ("assembly_complete_view_side.png", "Complete assembly · side"),
        ]
    )

    return "\n".join(
        [
            section_if_any(
                "Pre-agent final policy",
                "<p><span class='badge'>global civic stone prompts</span><span class='badge'>generic geometry cleanup</span><span class='badge'>strict Hunyuan Paint</span><span class='badge'>documented limits</span></p>",
                "本版仍是 non-agent 单次流程优化，不做新模型、不训练、不做 class-specific hacks、不做重型多轮优化。",
            ),
            trace_section,
            baseline_section,
            section_if_any(
                "Overall 3D browsers",
                f"<div class='model-grid'>{model_gallery}</div>{validation_body}{source_report_link}",
                "Texture 是 downstream additive stage：这里保留 V7/V8 的整体 layout 和 assembly 浏览，并新增 textured assembly。",
            ),
            section_if_any(
                "Layout visualizations inherited from V8",
                f"<div class='image-grid compact'>{layout_gallery}</div>",
            ),
            section_if_any(
                "Assembly multi-view renders inherited from V8",
                f"<div class='image-grid compact'>{assembly_gallery}</div>",
            ),
        ]
    )


def write_texture_report(output_dir: Path, texture_manifest: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_dir": str(output_dir),
        "layout_id": Path(texture_manifest["source_run"]).name,
        "num_parts": texture_manifest["num_parts"],
        "num_success": texture_manifest["num_success"],
        "num_failures": texture_manifest["num_failures"],
        "num_wall_skipped": texture_manifest["num_wall_skipped"],
        "num_basic_material": texture_manifest.get("num_basic_material", 0),
        "texture_model": texture_manifest["texture_model"],
        "assembly_obj": texture_manifest["assembly_obj"],
        "assembly_glb": texture_manifest.get("assembly_glb", ""),
        "assembly_status_check_glb": texture_manifest.get("assembly_status_check_glb", ""),
        "report_type": "archstudio_s2_texture",
    }
    write_json(report_dir / "summary.json", summary)

    overall_sections = write_overall_sections(texture_manifest)

    cards = []
    for item in texture_manifest["parts"]:
        output_obj = item.get("output_obj", "")
        diffuse = str(Path(output_obj).with_suffix(".jpg")) if output_obj else ""
        texture_img = (
            f"<img src='{html.escape(raw_asset_url(diffuse))}' alt='texture' />"
            if diffuse and Path(diffuse).exists()
            else "<div class='missing'>No generated texture</div>"
        )
        reference = item.get("reference_image", "")
        reference_img = (
            f"<img src='{html.escape(raw_asset_url(reference))}' alt='reference' />"
            if reference and Path(reference).exists()
            else "<div class='missing'>No reference</div>"
        )
        glb = item.get("output_glb", "")
        mesh_viewer = (
            f"<model-viewer src='{html.escape(raw_asset_url(glb))}' camera-controls auto-rotate "
            "interaction-prompt='none' exposure='1.1' shadow-intensity='0.75'></model-viewer>"
            if glb and Path(glb).exists()
            else "<div class='missing dark'>No textured 3D preview</div>"
        )
        delta = item.get("geometry_delta", {})
        links = []
        for label, path in [
            ("textured GLB", item.get("output_glb", "")),
            ("textured OBJ", output_obj),
            ("input OBJ", item.get("input_obj", "")),
            ("fallback OBJ", item.get("fallback_obj", "")),
        ]:
            if path and Path(path).exists():
                links.append(f"<a href='{html.escape(raw_asset_url(path))}'>{html.escape(label)}</a>")
        cards.append(
            f"""
            <article class="card {html.escape(item['status'])}">
              <div class="meta">
                <div>
                  <div class="label">{html.escape(item.get('target_class', ''))}</div>
                  <h2>{html.escape(item['part_id'])}</h2>
                </div>
                <strong>{html.escape(item['status'])}</strong>
              </div>
              <div class="assets">
                <figure>{reference_img}<figcaption>reference image</figcaption></figure>
                <figure>{texture_img}<figcaption>generated diffuse texture</figcaption></figure>
                <figure>{mesh_viewer}<figcaption>textured mesh preview</figcaption></figure>
              </div>
              <p class="small">surface_preserved={html.escape(str(item.get('surface_preserved', item.get('geometry_preserved', 'n/a'))))}
                · bbox_preserved={html.escape(str(item.get('bbox_preserved', 'n/a')))}
                · extent_delta={html.escape(str(delta.get('extent_max_abs_delta', '')))}
                · face_delta={html.escape(str(delta.get('face_count_delta', '')))}</p>
              <div class="links">{' '.join(links)}</div>
              <details><summary>record</summary><pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre></details>
            </article>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>ArchStudio-S2 Texture Report</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    body {{ margin: 24px; background: #f7f3eb; color: #171411; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    h1 {{ margin-bottom: 4px; }}
    .summary, .card {{ background: rgba(255,255,255,.86); border: 1px solid #ddd4c8; border-radius: 16px; padding: 16px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 16px; }}
    .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .image-grid.compact {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .meta {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; }}
    .label {{ color: #735f4f; text-transform: uppercase; letter-spacing: .12em; font-size: 12px; }}
    .assets {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    img, model-viewer, .missing {{ width: 100%; height: 260px; object-fit: contain; background: #fff; border: 1px solid #e3ddd4; border-radius: 12px; }}
    .model-grid model-viewer {{ height: 440px; }}
    .image-grid img {{ height: 220px; }}
    .missing {{ display: grid; place-items: center; color: #735f4f; }}
    model-viewer, .missing.dark {{ background: #151922; color: #d9dde8; }}
    figcaption, .small {{ color: #735f4f; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e6ded2; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: #735f4f; text-transform: uppercase; letter-spacing: .06em; }}
    .validation, .limits {{ margin-top: 14px; padding: 12px; background: #fbf8f2; border: 1px dashed #d4c7b8; border-radius: 12px; }}
    .badge {{ display:inline-block; padding: 4px 8px; border-radius: 999px; background:#e8f4df; color:#2d5a26; font-size:12px; font-weight:700; margin-right:6px; }}
    pre {{ overflow-x: auto; font-size: 12px; }}
    a {{ display: inline-block; margin-right: 10px; color: #8a3f1d; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>ArchStudio-S2 Texture Report</h1>
  <p><b>Project:</b> ArchStudio: A Multi-Agent Framework for Layout-Guided 3D Architectural Asset Generation</p>
  <p><b>Stage:</b> Stage 2 / ArchStudio-S2 additive texture stage</p>
  <section class="summary">
    <p><b>Source run:</b> {html.escape(texture_manifest['source_run'])}</p>
    <p><b>Texture model:</b> {html.escape(texture_manifest['texture_model'])}</p>
    <p><b>Parts:</b> {texture_manifest['num_parts']} · <b>Success:</b> {texture_manifest['num_success']} · <b>Failures:</b> {texture_manifest['num_failures']} · <b>Basic material fallback:</b> {texture_manifest.get('num_basic_material', 0)} · <b>Wall skipped:</b> {texture_manifest['num_wall_skipped']}</p>
    <p><span class="badge">strict generated texture</span><span class="badge">no basic material fallback</span><span class="badge">Hunyuan3D-Paint all parts</span></p>
    <div class="limits"><b>Known remaining limits:</b> geometry/texture realism is still bounded by Hunyuan-Omni and Hunyuan3D-Paint single-pass generation. This report records pipeline artifacts and leaves semantic multi-pass repair to the next agent stage.</div>
    <p><a href="{html.escape(raw_asset_url(texture_manifest['assembly_obj']))}">Download textured assembly OBJ</a>
       {"<a href='" + html.escape(raw_asset_url(texture_manifest.get('assembly_glb', ''))) + "'>Download textured assembly GLB</a>" if texture_manifest.get('assembly_glb') else ""}
       {"<a href='" + html.escape(raw_asset_url(texture_manifest.get('assembly_status_check_glb', ''))) + "'>Download status-check assembly GLB</a>" if texture_manifest.get('assembly_status_check_glb') else ""}
       <a href="{html.escape(raw_asset_url(output_dir / 'texture_manifest.json'))}">texture_manifest.json</a>
       <a href="{html.escape(raw_asset_url(output_dir / 'texture_failures.json'))}">texture_failures.json</a></p>
  </section>
  {overall_sections}
  <h2>Part Board</h2>
  <section class="grid">
    {''.join(cards)}
  </section>
</body>
</html>
"""
    (report_dir / "index.html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = source_run / "manifest.json"
    manifest = load_json(manifest_path)
    source_contract = load_source_contract(source_run)
    contract_by_part_id = contract_part_lookup(source_contract)
    scene_center = infer_scene_center(source_contract)
    parts = selected_parts(manifest.get("parts", []), set(args.part_id or []), args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = None
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, part in enumerate(parts):
        part_id = part["part_id"]
        target_class = str(part.get("target_class", "")).lower()
        part_out_dir = output_dir / "parts" / part_id
        part_out_dir.mkdir(parents=True, exist_ok=True)
        input_mesh = Path(part.get("normalized_output_path") or part.get("placeholder_output_path") or "")
        contract_part = contract_by_part_id.get(part_id)
        reference_image = resolve_part_reference_image(part, contract_part)
        output_obj = part_out_dir / f"{part_id}__hy3dpaint.obj"

        entry = {
            "part_id": part_id,
            "target_class": target_class,
            "input_obj": str(input_mesh),
            "reference_image": str(reference_image) if reference_image else "",
            "output_obj": str(output_obj),
            "status": "pending",
        }

        if not input_mesh.exists() or reference_image is None:
            entry.update(
                {
                    "status": "failed_untextured",
                    "fallback_obj": str(input_mesh) if input_mesh.exists() else "",
                    "reason": "missing input mesh or reference image",
                }
            )
            failures.append(entry)
            entries.append(entry)
            continue

        input_bbox = mesh_bbox(input_mesh)
        paint_input_mesh = input_mesh
        paint_frame_info: dict[str, Any] = {}
        if args.paint_frame_alignment:
            inferred_frame = infer_part_frame_axes(contract_part, input_bbox, scene_center)
            frame_input_mesh = part_out_dir / f"{part_id}__paint_frame_input.obj"
            paint_frame_info = prepare_paint_frame_input(input_mesh, frame_input_mesh, inferred_frame)
            paint_input_mesh = frame_input_mesh
            entry.update(paint_frame_info)
        if target_class == "wall" and args.wall_subdivide_for_paint:
            wall_subdivide_source = paint_input_mesh
            paint_input_mesh = part_out_dir / f"{part_id}__paint_input_subdivided.obj"
            export_subdivided_box_like_mesh(wall_subdivide_source, paint_input_mesh, args.wall_subdivision)
            entry["paint_input_obj"] = str(paint_input_mesh)
            entry["paint_input_note"] = (
                "wall sent to Hunyuan Paint as a same-bbox subdivided box in the current paint frame; "
                "original placeholder bbox is preserved after texture-stage restoration/assembly"
            )
        use_hunyuan_remesh = remeshed_paint_input_required(input_bbox["face_count"], args)
        if use_hunyuan_remesh:
            entry["paint_input_obj"] = str(paint_input_mesh)
            entry["paint_input_note"] = (
                f"force-generated mode: face_count {input_bbox['face_count']} exceeds "
                f"hunyuan_remesh_face_threshold {args.hunyuan_remesh_face_threshold}; "
                "still sent to Hunyuan3D-Paint with its remesh pre-pass, not basic material fallback"
            )
        if (
            not args.force_hunyuan_paint_dense
            and args.hunyuan_face_limit is not None
            and input_bbox["face_count"] > args.hunyuan_face_limit
        ):
            try:
                export_basic_material_obj(
                    input_mesh,
                    output_obj,
                    reference_image,
                    material_name=f"{part_id}_dense_basic",
                    default_color=(0.62, 0.58, 0.52),
                )
                output_bbox = mesh_bbox(output_obj)
                delta = geometry_delta(input_bbox, output_bbox)
                output_glb = output_obj.with_suffix(".glb")
                glb_result = export_glb(output_obj, output_glb) if args.export_glb else None
                entry.update(
                    {
                        "status": "succeeded_basic_material_dense",
                        "input_bbox": input_bbox,
                        "output_bbox": output_bbox,
                        "geometry_delta": delta,
                        "geometry_preserved": geometry_preserved(delta, args.geometry_tolerance),
                        "output_glb": glb_result
                        if glb_result and not str(glb_result).startswith("GLB export failed")
                        else "",
                        "glb_warning": glb_result if str(glb_result).startswith("GLB export failed") else "",
                        "note": (
                            f"face_count {input_bbox['face_count']} exceeds hunyuan_face_limit "
                            f"{args.hunyuan_face_limit}; used geometry-preserving basic material fallback"
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - failures must remain traceable.
                entry.update(
                    {
                        "status": "failed_untextured",
                        "fallback_obj": str(input_mesh),
                        "reason": repr(exc),
                    }
                )
                failures.append(entry)
            entries.append(entry)
            print(f"[{index + 1}/{len(parts)}] {part_id}: {entry['status']}", flush=True)
            continue

        if args.max_faces is not None and input_bbox["face_count"] > args.max_faces:
            entry.update(
                {
                    "status": "skipped_too_many_faces_untextured",
                    "fallback_obj": str(input_mesh),
                    "reason": f"face_count {input_bbox['face_count']} exceeds max_faces {args.max_faces}",
                    "input_bbox": input_bbox,
                }
            )
            failures.append(entry)
            entries.append(entry)
            print(f"[{index + 1}/{len(parts)}] {part_id}: {entry['status']}", flush=True)
            continue

        if output_obj.exists() and not args.force:
            output_bbox = mesh_bbox(output_obj)
            delta = geometry_delta(input_bbox, output_bbox)
            output_glb = output_obj.with_suffix(".glb")
            glb_result = str(output_glb) if output_glb.exists() else export_glb(output_obj, output_glb)
            preserved = geometry_preserved(
                delta,
                args.geometry_tolerance,
                allow_topology_change=target_class == "wall",
            )
            surface_ok = surface_preserved(
                delta,
                args.geometry_tolerance,
                allow_topology_change=target_class == "wall" or remeshed_paint_input_required(input_bbox["face_count"], args),
            )
            entry.update(
                {
                    "status": "succeeded",
                    "input_bbox": input_bbox,
                    "output_bbox": output_bbox,
                    "geometry_delta": delta,
                    "bbox_preserved": bbox_preserved(delta, args.geometry_tolerance),
                    "surface_preserved": surface_ok,
                    "geometry_preserved": preserved,
                    "output_glb": glb_result
                    if glb_result and not str(glb_result).startswith("GLB export failed")
                    else "",
                    "glb_warning": glb_result if str(glb_result).startswith("GLB export failed") else "",
                    "reused_existing": True,
                }
            )
            entries.append(entry)
            continue

        try:
            raw_paint_output_obj = output_obj
            if target_class == "wall" and args.wall_direct_reference_projection:
                wall_export_info = export_box_with_generated_texture_obj(
                    input_mesh,
                    output_obj,
                    reference_image,
                    material_name=f"{part_id}_wall_reference_facade",
                    tile_size=args.wall_texture_tile_size,
                    min_thickness=args.wall_display_min_thickness,
                    facade_projection=True,
                )
                wall_export_info.update(orient_obj_faces_outward(output_obj))
                entry.update(wall_export_info)
                entry["wall_texture_policy"] = "direct_t2i_reference_facade_projected_to_detected_outer_face"
                entry["wall_texture_source"] = "reference_image_direct_projection"
                entry["note"] = (
                    "wall bypasses Hunyuan Paint diffuse generation: the T2I facade/elevation reference image "
                    "is projected directly once onto the detected exterior wall face; side faces use muted UVs"
                )
            else:
                if pipeline is None:
                    pipeline = build_paint_pipeline(args.paint_repo.resolve(), args.max_num_view, args.resolution)
                if paint_frame_info.get("paint_frame_layout_to_input") and not (
                    target_class == "wall" and args.wall_transfer_generated_texture_to_bbox
                ):
                    raw_paint_output_obj = canonical_paint_asset_path(output_obj)
                if target_class == "wall" and args.wall_transfer_generated_texture_to_bbox:
                    raw_paint_output_obj = part_out_dir / f"{part_id}__hy3dpaint_raw_subdivided.obj"
                pipeline(
                    mesh_path=str(paint_input_mesh),
                    image_path=str(reference_image),
                    output_mesh_path=str(raw_paint_output_obj),
                    use_remesh=use_hunyuan_remesh,
                    save_glb=False,
                )
                if target_class == "wall" and args.wall_transfer_generated_texture_to_bbox:
                    generated_texture = raw_paint_output_obj.with_suffix(".jpg")
                    wall_export_info = export_box_with_generated_texture_obj(
                        input_mesh,
                        output_obj,
                        generated_texture,
                        material_name=f"{part_id}_wall_generated_material",
                        tile_size=args.wall_texture_tile_size,
                        min_thickness=args.wall_display_min_thickness,
                        facade_projection=args.wall_facade_projection,
                    )
                    wall_export_info.update(orient_obj_faces_outward(output_obj))
                    entry.update(wall_export_info)
                    entry["raw_hunyuan_paint_output_obj"] = str(raw_paint_output_obj)
                    entry["raw_hunyuan_paint_texture"] = str(generated_texture)
                    entry["wall_texture_policy"] = (
                        "generated_facade_projected_to_detected_outer_face"
                        if args.wall_facade_projection else
                        "generated_material_tiled_on_stable_bbox"
                    )
                    entry["note"] = (
                        "wall uses Hunyuan Paint to generate a detailed non-repeating facade/elevation texture; "
                        "the generated map is projected once onto the detected exterior wall face, while side faces use muted UVs"
                        if args.wall_facade_projection else
                        "wall uses Hunyuan Paint to generate a material texture from a wall-material reference image; "
                        "the generated map is tiled with stable bbox UVs instead of transferring object-render atlas UVs"
                    )
                elif paint_frame_info.get("paint_frame_layout_to_input"):
                    finalize_canonical_paint_output(raw_paint_output_obj, output_obj, paint_frame_info)
                    entry["raw_hunyuan_paint_output_obj"] = str(raw_paint_output_obj)
                    entry["paint_frame_note"] = (
                        "Hunyuan Paint ran in a canonical part-local frame where image horizontal/up/outward "
                        "match the inferred bbox facade frame; output vertices were restored to layout frame "
                        "without rewriting Hunyuan Paint atlas UVs."
                    )
            output_bbox = mesh_bbox(output_obj)
            uv_info: dict[str, Any] = {}
            if should_apply_world_axis_uv(output_bbox, args):
                uv_info = normalize_obj_uv_world_axes(output_obj, args)
                # UV-only rewrite: refresh GLB after the OBJ texture coordinate
                # contract is normalized, but keep the geometry bbox measured
                # before and after unchanged.
            delta = geometry_delta(input_bbox, output_bbox)
            output_glb = output_obj.with_suffix(".glb")
            glb_result = export_glb(output_obj, output_glb, viewer_frame=True) if args.export_glb else None
            allow_paint_topology_change = (
                (
                    target_class == "wall"
                    and bool(args.wall_transfer_generated_texture_to_bbox)
                    and not entry.get("display_bbox_adjusted")
                )
                or use_hunyuan_remesh
            )
            preserved = geometry_preserved(
                delta,
                args.geometry_tolerance,
                allow_topology_change=allow_paint_topology_change,
            )
            surface_ok = surface_preserved(
                delta,
                args.geometry_tolerance,
                allow_topology_change=allow_paint_topology_change,
            )
            entry.update(
                {
                    "status": "succeeded",
                    "input_bbox": input_bbox,
                    "output_bbox": output_bbox,
                    "geometry_delta": delta,
                    "bbox_preserved": bbox_preserved(delta, args.geometry_tolerance),
                    "surface_preserved": surface_ok,
                    "geometry_preserved": preserved,
                    "output_glb": glb_result
                    if glb_result and not str(glb_result).startswith("GLB export failed")
                    else "",
                    "glb_warning": glb_result if str(glb_result).startswith("GLB export failed") else "",
                }
            )
            if not surface_ok:
                entry["warning"] = "texture output surface bbox or faces differ from source beyond tolerance"
            elif not preserved:
                entry["note"] = (
                    (entry.get("note", "") + " " if entry.get("note") else "")
                    + "Hunyuan Paint UV unwrap changed vertex indexing while preserving bbox and face surface."
                )
            if uv_info:
                entry.update(uv_info)
                if uv_info.get("world_axis_uv_applied"):
                    entry["note"] = (
                        (entry.get("note", "") + " " if entry.get("note") else "")
                        + "Generic world-axis UV normalization applied after Hunyuan Paint to keep directional texture patterns aligned across parts."
                    )
        except Exception as exc:  # noqa: BLE001 - failures must remain traceable.
            if target_class == "wall" and args.wall_failure_fallback:
                try:
                    export_tiled_wall_material_obj(
                        input_mesh,
                        output_obj,
                        reference_image,
                        material_name=f"{part_id}_wall_fallback_after_hunyuan_failure",
                        default_color=(0.74, 0.69, 0.60),
                    )
                    output_bbox = mesh_bbox(output_obj)
                    delta = geometry_delta(input_bbox, output_bbox)
                    output_glb = output_obj.with_suffix(".glb")
                    glb_result = export_glb(output_obj, output_glb) if args.export_glb else None
                    entry.update(
                        {
                            "status": "failed_hunyuan_paint_fallback_tiled_wall",
                            "fallback_obj": str(output_obj),
                            "reason": repr(exc),
                            "input_bbox": input_bbox,
                            "output_bbox": output_bbox,
                            "geometry_delta": delta,
                            "geometry_preserved": geometry_preserved(delta, args.geometry_tolerance),
                            "output_glb": glb_result
                            if glb_result and not str(glb_result).startswith("GLB export failed")
                            else "",
                            "glb_warning": glb_result if str(glb_result).startswith("GLB export failed") else "",
                            "note": "Hunyuan Paint failed for wall; tiled UV wall texture is only a visible fallback",
                        }
                    )
                except Exception as fallback_exc:  # noqa: BLE001
                    entry.update(
                        {
                            "status": "failed_untextured",
                            "fallback_obj": str(input_mesh),
                            "reason": f"Hunyuan Paint failed: {exc!r}; wall fallback also failed: {fallback_exc!r}",
                        }
                    )
            else:
                entry.update(
                    {
                        "status": "failed_untextured",
                        "fallback_obj": str(input_mesh),
                        "reason": repr(exc),
                    }
                )
            failures.append(entry)
        entries.append(entry)
        print(f"[{index + 1}/{len(parts)}] {part_id}: {entry['status']}", flush=True)

    assembly_obj = output_dir / "assemblies" / "textured_or_fallback_assembly.obj"
    assemble_obj(entries, assembly_obj)
    assembly_glb_result = export_glb(assembly_obj, assembly_obj.with_suffix(".glb"), viewer_frame=True) if args.export_glb else None
    status_check_glb_result = (
        export_status_check_glb(entries, output_dir / "assemblies" / "textured_or_fallback_status_check.glb")
        if args.export_glb
        else None
    )
    texture_manifest = {
        "schema_version": "archstudio_s2.texture_manifest.v1",
        "source_run": str(source_run),
        "source_manifest": str(manifest_path),
        "source_agent_trace": load_source_agent_trace(source_run) or {},
        "source_agent_trace_path": str(source_run / "agent_trace" / "s2_agent_trace.json"),
        "baseline_texture_run": str(args.baseline_texture_run.resolve()) if args.baseline_texture_run else "",
        "texture_model": "Hunyuan3D-Paint-v2-1 + direct T2I wall facade projection" if args.wall_direct_reference_projection else "Hunyuan3D-Paint-v2-1",
        "paint_repo": str(args.paint_repo.resolve()),
        "paint_frame_alignment": bool(args.paint_frame_alignment),
        "paint_frame_policy": "bbox_outward_z_up_part_local_frame" if args.paint_frame_alignment else "off",
        "num_paint_frame_aligned": sum(1 for item in entries if item.get("paint_frame_alignment") == "layout_to_paint_canonical"),
        "num_parts": len(entries),
        "num_success": sum(1 for item in entries if is_success_status(str(item["status"]))),
        "num_failures": len(failures),
        "num_wall_skipped": sum(1 for item in entries if item["status"] == "skipped_wall_untextured"),
        "num_basic_material": sum(1 for item in entries if str(item["status"]).startswith("succeeded_basic_material")),
        "world_axis_uv_normalization": args.world_axis_uv_normalization,
        "world_axis_uv_tile_size": args.world_axis_uv_tile_size,
        "num_world_axis_uv_applied": sum(1 for item in entries if item.get("world_axis_uv_applied")),
        "assembly_obj": str(assembly_obj),
        "assembly_glb": assembly_glb_result
        if assembly_glb_result and not str(assembly_glb_result).startswith("GLB export failed")
        else "",
        "assembly_glb_warning": assembly_glb_result
        if str(assembly_glb_result).startswith("GLB export failed")
        else "",
        "assembly_status_check_glb": status_check_glb_result
        if status_check_glb_result and not str(status_check_glb_result).startswith("Status-check GLB export failed")
        else "",
        "assembly_status_check_glb_warning": status_check_glb_result
        if str(status_check_glb_result).startswith("Status-check GLB export failed")
        else "",
        "parts": entries,
    }
    write_json(output_dir / "texture_manifest.json", texture_manifest)
    write_json(output_dir / "texture_failures.json", failures)
    write_texture_report(output_dir, texture_manifest, failures)
    print(json.dumps(texture_manifest, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paint-repo", type=Path, default=DEFAULT_PAINT_REPO)
    parser.add_argument(
        "--baseline-texture-run",
        type=Path,
        default=None,
        help="Optional prior texture run to show as before/final comparison in the report.",
    )
    parser.add_argument("--part-id", action="append", help="Only texture this part id; can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N selected parts.")
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-5)
    parser.add_argument("--export-glb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="Optional safety cap: skip texture generation above this face count. Disabled by default.",
    )
    parser.add_argument(
        "--hunyuan-face-limit",
        type=int,
        default=300000,
        help=(
            "Use geometry-preserving basic material fallback above this face count "
            "instead of sending very dense meshes to Hunyuan Paint."
        ),
    )
    parser.add_argument(
        "--force-hunyuan-paint-dense",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Do not use basic-material fallback for dense meshes. Dense parts are still "
            "sent through Hunyuan3D-Paint, using its remesh pre-pass when above "
            "--hunyuan-remesh-face-threshold."
        ),
    )
    parser.add_argument(
        "--hunyuan-remesh-face-threshold",
        type=int,
        default=300000,
        help=(
            "When --force-hunyuan-paint-dense is enabled, use Hunyuan Paint's remesh "
            "pre-pass for meshes above this face count. Set to -1 to force remesh for all parts, "
            "or omit force mode to keep legacy basic-material behavior."
        ),
    )
    parser.add_argument(
        "--wall-failure-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If Hunyuan Paint fails on a wall placeholder, emit a visible tiled wall fallback and report the failure.",
    )
    parser.add_argument(
        "--wall-subdivide-for-paint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send wall placeholders to Hunyuan Paint as same-bbox subdivided boxes to avoid black texture atlases.",
    )
    parser.add_argument(
        "--wall-subdivision",
        type=int,
        default=24,
        help="Grid subdivisions per wall box face when --wall-subdivide-for-paint is enabled.",
    )
    parser.add_argument(
        "--wall-transfer-generated-texture-to-bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After Hunyuan paints a subdivided wall input, transfer the generated texture "
            "onto the stable original wall bbox mesh for assembly."
        ),
    )
    parser.add_argument(
        "--wall-display-min-thickness",
        type=float,
        default=0.035,
        help=(
            "Minimum final display thickness for wall bbox geometry. The Hunyuan-generated "
            "texture remains unchanged; only the visible assembly bbox is thickened when "
            "an input wall side is thinner than this."
        ),
    )
    parser.add_argument(
        "--wall-texture-tile-size",
        type=float,
        default=0.12,
        help=(
            "Layout-space size of one repeated wall material tile when transferring generated "
            "wall material maps onto stable bbox placeholder geometry. Ignored by facade projection."
        ),
    )
    parser.add_argument(
        "--wall-facade-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For wall placeholders, project a non-repeating generated facade/elevation map once "
            "onto the detected exterior face instead of repeating a small material tile."
        ),
    )
    parser.add_argument(
        "--wall-direct-reference-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For wall placeholders, bypass Hunyuan Paint diffuse generation and project the T2I "
            "reference facade image directly onto the detected exterior face."
        ),
    )
    parser.add_argument(
        "--paint-frame-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before Hunyuan Paint, rotate each part into a generic bbox-derived local frame: "
            "image horizontal, layout Z-up, and outward facade normal are aligned in Paint's "
            "canonical view frame; the textured mesh is then restored to layout coordinates "
            "without rewriting atlas UVs."
        ),
    )
    parser.add_argument(
        "--world-axis-uv-normalization",
        choices=("off", "planar", "all"),
        default="off",
        help=(
            "Generic post-paint UV orientation pass. 'planar' applies to thin/planar "
            "geometry, 'all' applies to every part. It uses world/layout axes, not "
            "semantic class labels, so directional textures such as bricks keep a "
            "consistent up/horizontal direction across parts."
        ),
    )
    parser.add_argument(
        "--world-axis-uv-tile-size",
        type=float,
        default=0.12,
        help="Layout-space texture repeat size for generic world-axis UV normalization.",
    )
    parser.add_argument(
        "--world-axis-uv-planar-thin-ratio",
        type=float,
        default=0.18,
        help=(
            "In planar mode, apply world-axis UV normalization when min_extent / max_extent "
            "is less than or equal to this ratio."
        ),
    )
    parser.add_argument(
        "--world-axis-uv-horizontal-normal-threshold",
        type=float,
        default=0.82,
        help=(
            "Absolute dot(normal, +Z) threshold above which a face uses layout XY UVs; "
            "other faces use world Z as texture V."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.hunyuan_remesh_face_threshold is not None and args.hunyuan_remesh_face_threshold < 0:
        args.hunyuan_remesh_face_threshold = 0
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
