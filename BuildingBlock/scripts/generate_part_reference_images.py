#!/usr/bin/env python
"""Generate part-level reference images from a layout mesh contract."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.prompting import (
    NEGATIVE_PROMPT,
    build_part_prompt,
    negative_prompt_for_part,
    prompt_hash,
    recommended_t2i_canvas_size,
    visual_ratio_for_part,
)
from scene_synthesis.building_mesh.reference_images import (
    provider_from_name,
    write_t2i_metadata,
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--provider", default="siliconflow_qwen_image")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--width", default=768, type=int)
    parser.add_argument("--height", default=768, type=int)
    parser.add_argument("--steps", default=None, type=int)
    parser.add_argument("--guidance-scale", default=None, type=float)
    parser.add_argument(
        "--ratio-aware",
        action="store_true",
        help="Regenerate prompts and request canvas sizes from each part bbox aspect ratio.",
    )
    parser.add_argument(
        "--skip-wall",
        action="store_true",
        help="Do not regenerate wall reference images for wall-placeholder-only runs.",
    )
    return parser


def main(argv):
    args = build_parser().parse_args(argv)
    contract = json.loads(args.contract.read_text())
    provider = provider_from_name(args.provider)
    for part in contract["parts"]:
        if args.skip_wall and str(part.get("target_class", "")).startswith("wall"):
            continue
        prompt = build_part_prompt(part) if args.ratio_aware else part["part_prompt"]
        negative_prompt = negative_prompt_for_part(part) if args.ratio_aware else part.get("negative_prompt", NEGATIVE_PROMPT)
        image_path = Path(part["reference_image_path"])
        prompt_path = image_path.parent / "prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt + "\n")
        width = args.width
        height = args.height
        ratio_metadata = None
        if args.ratio_aware:
            width, height = recommended_t2i_canvas_size(part.get("target_class", "object"), part["bbox"]["size"])
            ratio, front_axes, front_width, front_height = visual_ratio_for_part(
                part.get("target_class", "object"),
                part["bbox"]["size"],
            )
            ratio_metadata = {
                "target_visual_ratio": ratio,
                "front_axes": list(front_axes),
                "front_width": front_width,
                "front_height": front_height,
                "requested_canvas_width": width,
                "requested_canvas_height": height,
            }
        generate_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_path": image_path,
            "seed": args.seed,
            "width": width,
            "height": height,
        }
        if args.steps is not None:
            generate_kwargs["num_inference_steps"] = args.steps
        if args.guidance_scale is not None:
            generate_kwargs["guidance_scale"] = args.guidance_scale
        provider.generate(**generate_kwargs)
        effective_steps = generate_kwargs.get("num_inference_steps")
        effective_guidance_scale = generate_kwargs.get("guidance_scale")
        write_t2i_metadata(
            image_path.parent / "t2i_metadata.json",
            {
                "part_id": part["part_id"],
                "provider": args.provider,
                "seed": args.seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "prompt_hash": prompt_hash(prompt, negative_prompt),
                "steps": effective_steps,
                "guidance_scale": effective_guidance_scale,
                "status": "succeeded",
                "ratio_aware": bool(args.ratio_aware),
                "ratio_metadata": ratio_metadata,
                "reference_image_path": str(image_path),
            },
        )
    print("generated_reference_images", len(contract["parts"]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
