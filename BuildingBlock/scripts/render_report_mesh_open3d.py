#!/usr/bin/env python
"""Render assembled report meshes with Open3D offscreen rendering.

This optional helper is intended for high-quality report hero images.  It uses
the raw assembled OBJ emitted by the BuildingBlock-Hunyuan pipeline and writes
mesh-surface PNGs over the report assets used by ``index.html``.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


VIEWS = {
    "iso": (
        np.array([1.35, -1.55, 0.95], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        2.9,
    ),
    "front": (
        np.array([0.0, -1.0, 0.18], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        2.9,
    ),
    "side": (
        np.array([1.0, 0.0, 0.18], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        2.9,
    ),
    "top": (
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        2.25,
    ),
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--mesh",
        choices=("raw", "placeholder"),
        default="raw",
        help="Which assembled OBJ to render",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    return parser


def mesh_path_for_run(run_dir, mesh_kind):
    filename = (
        "raw_hunyuan_assembly.obj"
        if mesh_kind == "raw"
        else "placeholder_filled_assembly.obj"
    )
    return run_dir / "assemblies" / filename


def setup_lighting(renderer):
    renderer.scene.set_background([0.05, 0.06, 0.08, 1.0])
    renderer.scene.scene.set_sun_light(
        [0.45, -0.7, -1.0],
        [1.0, 1.0, 1.0],
        80000,
    )
    renderer.scene.scene.enable_sun_light(True)


def render_views(run_dir, mesh_kind, width, height):
    run_dir = run_dir.resolve()
    mesh_path = mesh_path_for_run(run_dir, mesh_kind)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    assets_dir = run_dir / "report" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("assembled mesh has no triangles: {}".format(mesh_path))
    mesh.compute_vertex_normals()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    setup_lighting(renderer)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.42, 0.86, 0.48, 1.0]
    renderer.scene.add_geometry("assembly", mesh, material)

    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=np.float32)
    extent = float(max(bbox.get_extent())) or 1.0
    prefix = "assembly_raw_view" if mesh_kind == "raw" else "assembly_placeholder_view"
    written = []

    for name, (direction, up, distance_factor) in VIEWS.items():
        direction = direction / np.linalg.norm(direction)
        eye = center + direction * extent * distance_factor
        renderer.setup_camera(42.0, center, eye.astype(np.float32), up.astype(np.float32))
        image = renderer.render_to_image()
        output_path = assets_dir / "{}_{}.png".format(prefix, name)
        o3d.io.write_image(str(output_path), image)
        written.append(output_path)

    iso_path = assets_dir / "{}_iso.png".format(prefix)
    legacy_path = assets_dir / (
        "assembly_raw.png" if mesh_kind == "raw" else "assembly_placeholder.png"
    )
    if iso_path.exists():
        legacy_path.write_bytes(iso_path.read_bytes())
        written.append(legacy_path)
    return written


def main(argv=None):
    args = build_parser().parse_args(argv)
    written = render_views(args.run_dir, args.mesh, args.width, args.height)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
