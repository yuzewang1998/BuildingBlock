#!/usr/bin/env python3
"""Run the ArchStudio-S2 reject/accept repair agent on an existing S2 run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.s2_repair_agent import (  # noqa: E402
    DEFAULT_ACCEPTANCE_SCORE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_REJECT_SEVERITY,
    S2AgentConfig,
    S2Repairer,
)


DEFAULT_T2I_PYTHON = Path("/home/wangyz/anaconda3/envs/archstudio_qwen_image/bin/python")
DEFAULT_HF_HOME = Path("/mnt/data/wangyz/huggingface")
DEFAULT_HUNYUAN_SCRIPT = REPO_ROOT / "scripts" / "hunyuan_bbox_single.py"
DEFAULT_CONDA_SH = Path("/home/wangyz/anaconda3/etc/profile.d/conda.sh")


def parse_env_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--t2i-env values must be KEY=VALUE")
    key, env_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("--t2i-env key cannot be empty")
    return key, env_value


def split_gpu_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def first_gpu(value: str | None, fallback: str = "0") -> str:
    gpus = split_gpu_list(value)
    return gpus[0] if gpus else fallback


def build_hunyuan_command(args: argparse.Namespace) -> str | None:
    if args.hunyuan_command:
        return args.hunyuan_command
    if args.no_default_hunyuan_command:
        return None
    return (
        'bash -lc "source {conda_sh} && conda activate {env_name} && '
        "CUDA_VISIBLE_DEVICES={gpu} python {script} "
        "--image $reference_image --bbox-sx $bbox_sx --bbox-sy $bbox_sy --bbox-sz $bbox_sz "
        '--save-dir $output_dir --file-name $part_id --seed {seed}"'
    ).format(
        conda_sh=args.conda_sh,
        env_name=args.hunyuan_env,
        gpu=first_gpu(args.hunyuan_gpu, "1"),
        script=args.hunyuan_script,
        seed=args.seed,
    )


def build_t2i_env(args: argparse.Namespace) -> Dict[str, str]:
    env = {
        "HF_ENDPOINT": args.hf_endpoint,
        "HF_HOME": str(args.hf_home),
        "HUGGINGFACE_HUB_CACHE": str(args.hf_home / "hub"),
        "QWEN_IMAGE_MODEL_ID": args.qwen_image_model_id,
        "QWEN_IMAGE_DEVICE": "cuda",
    }
    if args.t2i_gpu:
        env["CUDA_VISIBLE_DEVICES"] = first_gpu(args.t2i_gpu, "0")
    for key, value in args.t2i_env:
        env[key] = value
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Existing ArchStudio-S2 geometry run directory")
    parser.add_argument("--texture-run-dir", type=Path, default=None)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--max-no-improvement-iterations", type=int, default=2)
    parser.add_argument("--min-score-improvement", type=float, default=0.01)
    parser.add_argument("--acceptance-score", type=float, default=DEFAULT_ACCEPTANCE_SCORE)
    parser.add_argument("--reject-severity", type=int, default=DEFAULT_REJECT_SEVERITY)
    parser.add_argument("--max-sample-vertices", type=int, default=50000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--fast-report-refresh", action="store_true")
    parser.add_argument(
        "--evaluator-provider",
        default="metrics",
        choices=["metrics", "mock", "openai", "aimirror", "aicodemirror", "gpt", "gpt55", "multimodal"],
        help="Visual critic provider. Use gpt55/aimirror/openai to invoke GPT-5.5 VLM.",
    )
    parser.add_argument("--evaluator-model", default="gpt-5.5")
    parser.add_argument("--evaluator-base-url", default=None)
    parser.add_argument("--evaluator-auth-path", type=Path, default=None)
    parser.add_argument("--evaluator-max-images", type=int, default=12)
    evidence_group = parser.add_mutually_exclusive_group()
    evidence_group.add_argument("--evaluator-build-evidence", dest="evaluator_build_evidence", action="store_true")
    evidence_group.add_argument("--evaluator-no-evidence", dest="evaluator_build_evidence", action="store_false")
    parser.set_defaults(evaluator_build_evidence=None)
    parser.add_argument(
        "--evaluator-full-evidence-render",
        action="store_true",
        help="Render fresh evidence meshes instead of reusing existing report assets.",
    )
    parser.add_argument("--evaluator-mock-response", default=None)
    parser.add_argument(
        "--evaluator-required",
        action="store_true",
        help="Fail the whole agent loop if the VLM call fails; default is to trace error and continue deterministic-only.",
    )
    parser.add_argument(
        "--evaluator-mode",
        default="focused",
        choices=["focused", "omnibus"],
        help="focused runs one single-scope VLM request per stage/case; omnibus keeps the legacy all-evidence prompt.",
    )
    parser.add_argument("--focused-critic-max-parts", type=int, default=0, help="0 means focused critics audit every part.")
    parser.add_argument(
        "--focused-critic-scope",
        action="append",
        choices=["scene_assembly", "part_image", "part_mesh"],
        default=[],
        help="Repeat to restrict focused critics. Default runs scene_assembly, part_image, and part_mesh.",
    )
    parser.add_argument("--focused-critic-score-threshold", type=float, default=0.72)
    parser.add_argument(
        "--disable-sequential-part-gate",
        action="store_true",
        help="Legacy focused behavior: review all part images/meshes in a batch each iteration. Default blocks on one part-stage until accepted.",
    )
    parser.add_argument(
        "--agent-scope",
        default="full_s2_repair",
        choices=["full_s2_repair", "assemble_geometry", "assembly_geometry", "assemble_only"],
        help="Restrict the repair loop. assemble_geometry runs iteration-0 per-part image/mesh prechecks, then overall assembled-mesh geometry review and repair routing.",
    )

    parser.add_argument("--provider", default="qwen_image_local")
    parser.add_argument("--t2i-python", type=Path, default=DEFAULT_T2I_PYTHON)
    parser.add_argument("--t2i-gpu", default="0", help="GPU id or comma list. Comma list runs per-part T2I subprocesses in parallel.")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--qwen-image-model-id", default="Qwen/Qwen-Image")
    parser.add_argument("--t2i-env", action="append", type=parse_env_assignment, default=[])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--t2i-retries", type=int, default=1)
    parser.add_argument("--t2i-retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--t2i-min-free-gpu-mem-mib",
        type=int,
        default=None,
        help="Minimum free GPU memory for Qwen T2I scheduling. Defaults to S2_T2I_MIN_FREE_GPU_MEM_MIB or 30000 MiB.",
    )
    parser.add_argument(
        "--t2i-max-parallel",
        type=int,
        default=None,
        help="Cap concurrent per-part T2I subprocesses after GPU filtering. Defaults to number of eligible GPUs.",
    )

    parser.add_argument("--hunyuan-command", default=None)
    parser.add_argument("--no-default-hunyuan-command", action="store_true")
    parser.add_argument("--hunyuan-gpu", default="1", help="GPU id or comma list. Comma list runs per-part Hunyuan subprocesses in parallel.")
    parser.add_argument("--hunyuan-env", default="hunyuan_omni_clean")
    parser.add_argument("--hunyuan-script", type=Path, default=DEFAULT_HUNYUAN_SCRIPT)
    parser.add_argument("--conda-sh", type=Path, default=DEFAULT_CONDA_SH)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--retry-count", type=int, default=0)

    parser.add_argument("--disable-t2i", action="store_true")
    parser.add_argument("--disable-geometry", action="store_true")
    parser.add_argument("--disable-snap", action="store_true")
    parser.add_argument("--disable-split", action="store_true")
    parser.add_argument("--no-split-child-generation", action="store_true")
    parser.add_argument("--allow-procedural-t2i", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = S2AgentConfig(
        run_dir=args.run_dir.resolve(),
        texture_run_dir=args.texture_run_dir.resolve() if args.texture_run_dir else None,
        max_iterations=args.max_iterations,
        max_no_improvement_iterations=args.max_no_improvement_iterations,
        min_score_improvement=args.min_score_improvement,
        acceptance_score=args.acceptance_score,
        reject_severity=args.reject_severity,
        max_sample_vertices=args.max_sample_vertices,
        dry_run=args.dry_run,
        force_report=not args.no_report,
        fast_report_refresh=args.fast_report_refresh,
        evaluator_provider=args.evaluator_provider,
        evaluator_model=args.evaluator_model,
        evaluator_base_url=args.evaluator_base_url,
        evaluator_auth_path=args.evaluator_auth_path,
        evaluator_max_images=args.evaluator_max_images,
        evaluator_build_evidence=args.evaluator_build_evidence,
        evaluator_fast_evidence=not args.evaluator_full_evidence_render,
        evaluator_mock_response=args.evaluator_mock_response,
        evaluator_required=args.evaluator_required,
        evaluator_mode=args.evaluator_mode,
        focused_critic_max_parts=args.focused_critic_max_parts,
        focused_critic_scopes=args.focused_critic_scope or None,
        focused_critic_score_threshold=args.focused_critic_score_threshold,
        agent_scope=args.agent_scope,
        sequential_part_gate=not args.disable_sequential_part_gate,
        provider=args.provider,
        t2i_python=args.t2i_python,
        t2i_env=build_t2i_env(args),
        t2i_gpus=split_gpu_list(args.t2i_gpu),
        t2i_seed=args.seed,
        t2i_steps=args.steps,
        t2i_guidance_scale=args.guidance_scale,
        t2i_retries=args.t2i_retries,
        t2i_retry_sleep=args.t2i_retry_sleep,
        hunyuan_command=build_hunyuan_command(args),
        hunyuan_gpus=split_gpu_list(args.hunyuan_gpu),
        hunyuan_timeout_seconds=args.timeout_seconds,
        hunyuan_retry_count=args.retry_count,
        hunyuan_seed=args.seed,
        enable_t2i=not args.disable_t2i,
        enable_geometry=not args.disable_geometry,
        enable_snap=not args.disable_snap,
        enable_split=not args.disable_split,
        split_then_generate=not args.no_split_child_generation,
        forbid_procedural_t2i=not args.allow_procedural_t2i,
    )
    if args.t2i_min_free_gpu_mem_mib is not None:
        os.environ["S2_T2I_MIN_FREE_GPU_MEM_MIB"] = str(max(0, args.t2i_min_free_gpu_mem_mib))
    if args.t2i_max_parallel is not None:
        os.environ["S2_T2I_MAX_PARALLEL"] = str(max(1, args.t2i_max_parallel))
    # Keep ordinary Python subprocesses deterministic with the same project path.
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))
    result = S2Repairer(config).run()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.accepted or result.stop_condition != "rejected_but_no_enabled_operations" else 2


if __name__ == "__main__":
    raise SystemExit(main())
