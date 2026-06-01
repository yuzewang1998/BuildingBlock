#!/usr/bin/env python
"""Run a single-image single-bbox Hunyuan3D-Omni generation job."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wangyz/project/0working/Hunyuan3D-Omni-main/Hunyuan3D-Omni-main")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover


def save_ply_points(filename, points):
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % len(points))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write("%f %f %f\n" % (point[0], point[1], point[2]))


def export_outputs(mesh, sampled_point, image_file, file_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh.export(save_dir / f"{file_name}.obj")
    mesh.export(save_dir / f"{file_name}.glb")
    save_ply_points(str(save_dir / f"{file_name}.ply"), sampled_point.cpu().numpy())
    shutil.copy(image_file, save_dir / f"{file_name}.png")


def layout_xyz_to_hunyuan_lhw(bbox):
    """Map BuildingBlock bbox [x, y, z-up] to Hunyuan bbox [length, height, width]."""
    if len(bbox) != 3:
        raise ValueError("bbox must contain exactly three values")
    return [float(bbox[0]), float(bbox[2]), float(bbox[1])]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox-json", default=None, help='JSON string like "[sx, sy, sz]"')
    parser.add_argument("--bbox-sx", type=float, default=None)
    parser.add_argument("--bbox-sy", type=float, default=None)
    parser.add_argument("--bbox-sz", type=float, default=None)
    parser.add_argument(
        "--bbox-input-frame",
        choices=("layout_xyz", "hunyuan_lhw"),
        default="layout_xyz",
        help=(
            "Interpret bbox values as BuildingBlock layout [x,y,z-up] by default "
            "and convert to Hunyuan [length,height,width]. Use hunyuan_lhw to pass through."
        ),
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--repo-id", default="tencent/Hunyuan3D-Omni")
    parser.add_argument("--file-name", default="part")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--flashvdm", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="CUDA random seed for the Hunyuan generation call.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of Hunyuan denoising/inference steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale for Hunyuan generation.",
    )
    parser.add_argument(
        "--mock-box",
        action="store_true",
        help="Testing only: emit a simple bbox-shaped OBJ/GLB/PLY without loading Hunyuan.",
    )
    return parser


def export_mock_box_outputs(input_bbox, image_file, file_name, save_dir):
    import trimesh

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = [float(value) for value in input_bbox]
    mesh = trimesh.creation.box(extents=(sx, sz, sy))
    mesh.export(save_dir / f"{file_name}.obj")
    mesh.export(save_dir / f"{file_name}.glb")
    save_ply_points(str(save_dir / f"{file_name}.ply"), torch.zeros((8, 3)))
    if image_file and Path(image_file).exists():
        shutil.copy(image_file, save_dir / f"{file_name}.png")


def main():
    args = build_parser().parse_args()
    if args.bbox_sx is not None and args.bbox_sy is not None and args.bbox_sz is not None:
        bbox = [args.bbox_sx, args.bbox_sy, args.bbox_sz]
    else:
        if args.bbox_json is None:
            raise ValueError("either --bbox-json or all of --bbox-sx/--bbox-sy/--bbox-sz is required")
        bbox = json.loads(args.bbox_json)
    if len(bbox) != 3:
        raise ValueError("bbox-json must be [sx, sy, sz]")
    input_bbox = [float(value) for value in bbox]
    if args.bbox_input_frame == "layout_xyz":
        bbox = layout_xyz_to_hunyuan_lhw(input_bbox)
    else:
        bbox = input_bbox

    if args.mock_box:
        export_mock_box_outputs(input_bbox, args.image, args.file_name, args.save_dir)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.save_dir) / f"{args.file_name}_bbox_metadata.json").write_text(
            json.dumps(
                {
                    "mock_box": True,
                    "bbox_input_frame": args.bbox_input_frame,
                    "input_bbox": input_bbox,
                    "hunyuan_bbox_length_height_width": bbox,
                    "seed": args.seed,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return

    pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
        args.repo_id,
        fast_decode=args.flashvdm,
    )

    bbox_tensor = (
        torch.FloatTensor(bbox)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(pipeline.device)
        .to(pipeline.dtype)
    )

    result = pipeline(
        image=args.image,
        bbox=bbox_tensor,
        num_inference_steps=args.num_inference_steps,
        octree_resolution=512,
        mc_level=0,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    )
    mesh = result["shapes"][0][0]
    sampled_point = result["sampled_point"][0]
    export_outputs(mesh, sampled_point, args.image, args.file_name, args.save_dir)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.save_dir) / f"{args.file_name}_bbox_metadata.json").write_text(
        json.dumps(
            {
                "bbox_input_frame": args.bbox_input_frame,
                "input_bbox": input_bbox,
                "hunyuan_bbox_length_height_width": bbox,
                "seed": args.seed,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
