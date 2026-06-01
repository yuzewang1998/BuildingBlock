#!/usr/bin/env python3
"""Generate missing part reference images from a layout mesh contract.

This is a resumable wrapper for open-semantic experiments: existing reference
images are preserved, provider failures are recorded per part, and the process
continues instead of aborting the whole layout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.prompting import (  # noqa: E402
    NEGATIVE_PROMPT,
    PROMPT_POLICY_VERSION,
    build_part_prompt,
    component_context_tail,
    component_core_description,
    component_source_description,
    component_visual_subject,
    negative_prompt_for_part,
    prompt_hash,
    recommended_t2i_canvas_size,
    visual_ratio_for_part,
)
from scene_synthesis.building_mesh.reference_images import (  # noqa: E402
    provider_from_name,
    write_t2i_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--provider", default="siliconflow_qwen_image")
    parser.add_argument(
        "--fallback-provider",
        action="append",
        default=[],
        help="Provider to try after the primary provider fails. May be repeated.",
    )
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--width", default=768, type=int)
    parser.add_argument("--height", default=768, type=int)
    parser.add_argument("--steps", default=None, type=int)
    parser.add_argument("--guidance-scale", default=None, type=float)
    parser.add_argument("--ratio-aware", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate existing reference.png files.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--part-id",
        action="append",
        default=[],
        help="Regenerate only this part_id. May be repeated. Applied before --limit.",
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional run directory for repair-agent reruns. When set, reference images "
            "and agent_prompt_revision.json are read/written under OUTPUT_RUN_DIR/parts/<part_id>/ "
            "instead of any absolute source paths embedded in the contract."
        ),
    )
    return parser.parse_args()


def load_agent_prompt_revision(part: dict) -> dict:
    """Return per-part prompt override emitted by the S2 repair agent."""
    image_path = Path(part.get("reference_image_path") or part.get("reference_image") or "")
    candidates = []
    explicit = part.get("agent_prompt_revision_path")
    if explicit:
        candidates.append(Path(str(explicit)))
    if image_path:
        candidates.append(image_path.parent / "agent_prompt_revision.json")
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("prompt"):
            payload["prompt_revision_path"] = str(candidate)
            return payload
    return {}


def main() -> int:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    providers = [(args.provider, provider_from_name(args.provider))]
    for fallback_name in args.fallback_provider:
        if fallback_name and fallback_name != args.provider:
            providers.append((fallback_name, provider_from_name(fallback_name)))
    generated = 0
    skipped = 0
    failed = 0
    selected = contract.get("parts", [])
    selected_part_ids = {str(part_id) for part_id in args.part_id if str(part_id).strip()}
    if selected_part_ids:
        selected = [part for part in selected if str(part.get("part_id")) in selected_part_ids]
        missing_part_ids = selected_part_ids.difference(str(part.get("part_id")) for part in selected)
        if missing_part_ids:
            raise SystemExit("Unknown --part-id values: {}".format(", ".join(sorted(missing_part_ids))))
    if args.limit is not None:
        selected = selected[: args.limit]

    if args.output_run_dir is not None:
        output_run_dir = args.output_run_dir.resolve()
        localized = []
        for part in selected:
            item = dict(part)
            part_id = str(item.get("part_id"))
            local_part_dir = output_run_dir / "parts" / part_id
            local_reference = local_part_dir / "reference.png"
            item["reference_image_path"] = str(local_reference)
            item["reference_image"] = str(local_reference)
            item["agent_prompt_revision_path"] = str(local_part_dir / "agent_prompt_revision.json")
            localized.append(item)
        selected = localized

    for index, part in enumerate(selected, start=1):
        image_path = Path(part["reference_image_path"])
        if image_path.exists() and not args.force:
            skipped += 1
            print(f"[{index}/{len(selected)}] skip existing {part['part_id']}", flush=True)
            continue

        prompt_revision = load_agent_prompt_revision(part)
        if prompt_revision:
            prompt = str(prompt_revision["prompt"])
            negative_prompt = str(prompt_revision.get("negative_prompt") or negative_prompt_for_part(part))
        else:
            prompt = build_part_prompt(part) if args.ratio_aware else part["part_prompt"]
            negative_prompt = (
                negative_prompt_for_part(part)
                if args.ratio_aware
                else part.get("negative_prompt", NEGATIVE_PROMPT)
            )
        width = args.width
        height = args.height
        ratio_metadata = None
        if args.ratio_aware:
            width, height = recommended_t2i_canvas_size(
                part.get("target_class", "object"),
                part["bbox"]["size"],
            )
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

        prompt_trace = {
            "prompt_policy_version": PROMPT_POLICY_VERSION if args.ratio_aware else part.get("prompt_policy_version"),
            "part_description_source": component_source_description(part),
            "part_description_core": component_core_description(part),
            "part_visual_subject": component_visual_subject(part),
            "part_context_tail": component_context_tail(part),
            "stage": "t2i_reference",
            "policy": (
                "agent_prompt_revision"
                if prompt_revision
                else "part_only_core_subject_with_context_as_negative_metadata"
                if args.ratio_aware
                else "contract_prompt"
            ),
            "agent_prompt_revision_path": prompt_revision.get("prompt_revision_path"),
            "agent_prompt_revision_attempt": prompt_revision.get("revision_attempt"),
            "agent_repair_instruction": prompt_revision.get("repair_instruction"),
        }

        image_path.parent.mkdir(parents=True, exist_ok=True)
        (image_path.parent / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
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

        last_error = None
        provider_errors = []
        succeeded = False
        succeeded_provider = None
        succeeded_attempt = 0
        for provider_index, (provider_name, provider) in enumerate(providers):
            retry_count = args.retries if provider_index == 0 else 0
            for attempt in range(retry_count + 1):
                try:
                    provider.generate(**generate_kwargs)
                    succeeded = True
                    succeeded_provider = provider_name
                    succeeded_attempt = attempt + 1
                    break
                except Exception as exc:  # noqa: BLE001 - per-part traceability.
                    last_error = repr(exc)
                    provider_errors.append(
                        {
                            "provider": provider_name,
                            "attempt": attempt + 1,
                            "error": last_error,
                        }
                    )
                    if attempt < retry_count:
                        print(
                            f"[{index}/{len(selected)}] retry {attempt + 1} "
                            f"{provider_name} after {last_error}",
                            flush=True,
                        )
                        time.sleep(args.retry_sleep)
            if succeeded:
                break
            if provider_index + 1 < len(providers):
                print(
                    f"[{index}/{len(selected)}] fallback after {provider_name} error: {last_error}",
                    flush=True,
                )
        if succeeded:
            write_t2i_metadata(
                image_path.parent / "t2i_metadata.json",
                {
                    "part_id": part["part_id"],
                    "provider": succeeded_provider,
                    "primary_provider": args.provider,
                    "fallback_providers": list(args.fallback_provider),
                    "provider_errors": provider_errors,
                    "used_fallback": succeeded_provider != args.provider,
                    "seed": args.seed,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "prompt_hash": prompt_hash(prompt, negative_prompt),
                    "steps": generate_kwargs.get("num_inference_steps"),
                    "guidance_scale": generate_kwargs.get("guidance_scale"),
                    "status": "succeeded",
                    "attempts": succeeded_attempt,
                    "ratio_aware": bool(args.ratio_aware),
                    "ratio_metadata": ratio_metadata,
                    "prompt_trace": prompt_trace,
                    "agent_prompt_revision": prompt_revision or None,
                    "reference_image_path": str(image_path),
                },
            )
            generated += 1
            fallback_suffix = " fallback" if succeeded_provider != args.provider else ""
            print(
                f"[{index}/{len(selected)}] generated {part['part_id']} "
                f"provider={succeeded_provider} attempt={succeeded_attempt}{fallback_suffix}",
                flush=True,
            )
        else:
            failed += 1
            write_t2i_metadata(
                image_path.parent / "t2i_metadata.json",
                {
                    "part_id": part["part_id"],
                    "provider": args.provider,
                    "fallback_providers": list(args.fallback_provider),
                    "provider_errors": provider_errors,
                    "seed": args.seed,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "prompt_hash": prompt_hash(prompt, negative_prompt),
                    "status": "failed",
                    "error": last_error,
                    "ratio_aware": bool(args.ratio_aware),
                    "ratio_metadata": ratio_metadata,
                    "prompt_trace": prompt_trace,
                    "agent_prompt_revision": prompt_revision or None,
                    "reference_image_path": str(image_path),
                },
            )
            print(f"[{index}/{len(selected)}] failed {part['part_id']}: {last_error}", flush=True)

    print(json.dumps({"generated": generated, "skipped": skipped, "failed": failed}, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
