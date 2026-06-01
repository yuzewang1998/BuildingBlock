#!/usr/bin/env python3
"""Run ArchStudio-S2 open-semantic layout experiments.

This is a resumable orchestration wrapper for the Stage-1 open-vocabulary
layout prototypes.  It intentionally composes the existing one-step scripts
instead of changing the geometry generation logic:

1. prepare S2 contract/run directories;
2. generate missing part reference images;
3. run Hunyuan-Omni bbox geometry for every part;
4. build the static report consumed by the 9999 harness.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT_ROOT = Path(
    "/home/wangyz/project/0working/ArchStudio/stage1/experiment_results/"
    "stage1_quality/V9-P2d-open-semantic-ground-plane-alignment"
)
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2")
DEFAULT_HUNYUAN_SCRIPT = REPO_ROOT / "scripts" / "hunyuan_bbox_single.py"
DEFAULT_CONDA_SH = Path("/home/wangyz/anaconda3/etc/profile.d/conda.sh")
DEFAULT_TEXTURE_SCRIPT = REPO_ROOT / "scripts" / "texture_archstudio_s2_hunyuan_paint.py"
DEFAULT_TEXTURE_OUTPUT_ROOT = Path("/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2_texture")
DEFAULT_T2I_PYTHON = Path("/home/wangyz/anaconda3/envs/archstudio_qwen_image/bin/python")
DEFAULT_HF_HOME = Path("/mnt/data/wangyz/huggingface")


@dataclass(frozen=True)
class LayoutCase:
    index: int
    case_dir: Path
    layout_path: Path
    layout_id: str
    part_count: int

    @property
    def case_name(self) -> str:
        return self.case_dir.name

    @property
    def run_name(self) -> str:
        tag = "v9p2d" if "V9-P2d" in str(self.case_dir) else "v9p2"
        return "{}_{}".format(tag, self.case_name.replace("-", "_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--texture-output-root", type=Path, default=DEFAULT_TEXTURE_OUTPUT_ROOT)
    parser.add_argument("--run-suffix", default="", help="Append a suffix to every output run name.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Case selector: numeric prefix such as 02, full directory name, "
            "or run name. May be repeated. Default: all cases."
        ),
    )
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", default="4", help="CUDA_VISIBLE_DEVICES for Hunyuan geometry.")
    parser.add_argument("--hunyuan-env", default="hunyuan_omni_clean")
    parser.add_argument("--conda-sh", type=Path, default=DEFAULT_CONDA_SH)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--retry-count", type=int, default=0)
    parser.add_argument("--provider", default="qwen_image_local")
    parser.add_argument(
        "--t2i-python",
        type=Path,
        default=DEFAULT_T2I_PYTHON,
        help="Python executable used only for text-to-image reference generation.",
    )
    parser.add_argument(
        "--t2i-gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES for local T2I. Defaults to --gpu when omitted.",
    )
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--qwen-image-model-id", default="Qwen/Qwen-Image")
    parser.add_argument(
        "--fallback-provider",
        action="append",
        default=[],
        help="Fallback T2I provider after the primary provider fails. May be repeated.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--t2i-retries", type=int, default=3)
    parser.add_argument("--t2i-retry-sleep", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument(
        "--reference-limit",
        type=int,
        default=None,
        help="Debug option: only generate references for the first N parts.",
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-t2i", action="store_true")
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-report", action="store_true")
    parser.add_argument("--with-texture", action="store_true", help="Run Hunyuan3D-Paint texture stage after geometry.")
    parser.add_argument("--texture-env", default="hunyuan3d_paint_21")
    parser.add_argument("--texture-gpu", default="1")
    parser.add_argument("--force-texture", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--t2i-only", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--texture-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: List[str], *, dry_run: bool = False, env: Optional[dict] = None) -> None:
    print("\n$ {}".format(" ".join(subprocess.list2cmdline([token]) for token in command)), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(REPO_ROOT), check=True, env=env)


def t2i_env(args: argparse.Namespace) -> dict:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}:{pythonpath}"
    env["HF_ENDPOINT"] = args.hf_endpoint
    env["HF_HOME"] = str(args.hf_home)
    env["HUGGINGFACE_HUB_CACHE"] = str(args.hf_home / "hub")
    env.setdefault("QWEN_IMAGE_MODEL_ID", args.qwen_image_model_id)
    env.setdefault("QWEN_IMAGE_DEVICE", "cuda")
    t2i_gpu = args.t2i_gpu if args.t2i_gpu is not None else args.gpu
    if t2i_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(t2i_gpu)
    return env


def load_cases(layout_root: Path) -> List[LayoutCase]:
    cases: List[LayoutCase] = []
    for case_dir in sorted(layout_root.glob("[0-9][0-9]-*")):
        layout_path = case_dir / "layout_s2_open_semantic.json"
        if not layout_path.exists():
            continue
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        match = re.match(r"^(\d+)-", case_dir.name)
        index = int(match.group(1)) if match else len(cases) + 1
        cases.append(
            LayoutCase(
                index=index,
                case_dir=case_dir,
                layout_path=layout_path,
                layout_id=str(payload.get("layout_id") or ""),
                part_count=len(payload.get("parts") or []),
            )
        )
    return cases


def selector_matches(case: LayoutCase, selectors: Iterable[str]) -> bool:
    selectors = [selector.strip() for selector in selectors if selector.strip()]
    if not selectors:
        return True
    aliases = {
        "{:02d}".format(case.index),
        str(case.index),
        case.case_name,
        case.run_name,
        case.layout_id,
    }
    return any(selector in aliases for selector in selectors)


def filtered_cases(cases: List[LayoutCase], args: argparse.Namespace) -> List[LayoutCase]:
    selected = [
        case
        for case in cases
        if selector_matches(case, args.case)
        and (args.start_index is None or case.index >= args.start_index)
        and (args.end_index is None or case.index <= args.end_index)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def contract_path(run_dir: Path) -> Path:
    return run_dir / "contract" / "layout_mesh_contract.json"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def report_path(run_dir: Path) -> Path:
    return run_dir / "report" / "index.html"


def count_existing_references(contract_file: Path) -> tuple[int, int]:
    if not contract_file.exists():
        return 0, 0
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    parts = contract.get("parts") or []
    existing = 0
    for part in parts:
        image_path = Path(part.get("reference_image_path") or part.get("reference_image") or "")
        if image_path.exists():
            existing += 1
    return existing, len(parts)


def count_existing_geometry(run_dir: Path) -> tuple[int, int]:
    cpath = contract_path(run_dir)
    if not cpath.exists():
        return 0, 0
    contract = json.loads(cpath.read_text(encoding="utf-8"))
    parts = contract.get("parts") or []
    existing = 0
    for part in parts:
        part_id = part.get("part_id")
        if part_id and any((run_dir / "parts" / part_id).glob("*.obj")):
            existing += 1
    return existing, len(parts)


def hunyuan_command(args: argparse.Namespace) -> str:
    return (
        'bash -lc "source {conda_sh} && conda activate {env_name} && '
        "CUDA_VISIBLE_DEVICES={gpu} python {script} "
        "--image $reference_image --bbox-sx $bbox_sx --bbox-sy $bbox_sy --bbox-sz $bbox_sz "
        '--save-dir $output_dir --file-name $part_id"'
    ).format(
        conda_sh=args.conda_sh,
        env_name=args.hunyuan_env,
        gpu=args.gpu,
        script=DEFAULT_HUNYUAN_SCRIPT,
    )


def prepare_case(case: LayoutCase, run_dir: Path, args: argparse.Namespace) -> None:
    if contract_path(run_dir).exists() and not args.force_prepare:
        print(f"[{case.case_name}] prepare skip: {contract_path(run_dir)}", flush=True)
        return
    command = [
        sys.executable,
        "scripts/generate_building_mesh_hunyuan.py",
        str(case.layout_path),
        "--run-dir",
        str(run_dir),
        "--prepare-only",
        "--force",
    ]
    run_command(command, dry_run=args.dry_run)


def generate_references(case: LayoutCase, run_dir: Path, args: argparse.Namespace) -> None:
    existing, total = count_existing_references(contract_path(run_dir))
    if total and existing == total and not args.force_t2i:
        print(f"[{case.case_name}] t2i skip: {existing}/{total} reference images exist", flush=True)
        return
    command = [
        str(args.t2i_python),
        "scripts/generate_missing_reference_images.py",
        "--contract",
        str(contract_path(run_dir)),
        "--provider",
        args.provider,
        "--seed",
        str(args.seed),
        "--ratio-aware",
        "--retries",
        str(args.t2i_retries),
        "--retry-sleep",
        str(args.t2i_retry_sleep),
    ]
    for fallback_provider in args.fallback_provider:
        command.extend(["--fallback-provider", fallback_provider])
    if args.force_t2i:
        command.append("--force")
    if args.steps is not None:
        command.extend(["--steps", str(args.steps)])
    if args.guidance_scale is not None:
        command.extend(["--guidance-scale", str(args.guidance_scale)])
    if args.reference_limit is not None:
        command.extend(["--limit", str(args.reference_limit)])
    run_command(command, dry_run=args.dry_run, env=t2i_env(args))


def run_geometry(case: LayoutCase, run_dir: Path, args: argparse.Namespace) -> None:
    existing, total = count_existing_geometry(run_dir)
    if manifest_path(run_dir).exists() and existing == total and total and not args.force_geometry:
        print(f"[{case.case_name}] geometry skip: manifest exists and {existing}/{total} objs exist", flush=True)
        return
    command = [
        sys.executable,
        "scripts/generate_building_mesh_hunyuan.py",
        str(run_dir / "input" / "layout.json"),
        "--contract-file",
        str(contract_path(run_dir)),
        "--run-dir",
        str(run_dir),
        "--force",
        "--hunyuan-command",
        hunyuan_command(args),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--retry-count",
        str(args.retry_count),
    ]
    run_command(command, dry_run=args.dry_run)


def generate_report(case: LayoutCase, run_dir: Path, args: argparse.Namespace, *, force: bool = False) -> None:
    if report_path(run_dir).exists() and not (args.force_report or force):
        print(f"[{case.case_name}] report skip: {report_path(run_dir)}", flush=True)
        return
    command = [
        sys.executable,
        "scripts/generate_run_report.py",
        "--run-dir",
        str(run_dir),
    ]
    run_command(command, dry_run=args.dry_run)


def texture_dir_for_case(case: LayoutCase, args: argparse.Namespace) -> Path:
    return args.texture_output_root / f"{case.run_name}{args.run_suffix}_partfocus_hy3dpaint"


def run_texture(case: LayoutCase, run_dir: Path, args: argparse.Namespace) -> None:
    texture_dir = texture_dir_for_case(case, args)
    if (texture_dir / "texture_manifest.json").exists() and not args.force_texture:
        print(f"[{case.case_name}] texture skip: {texture_dir / 'texture_manifest.json'}", flush=True)
        return
    command_text = (
        "source {conda_sh} && conda activate {env_name} && "
        "CUDA_VISIBLE_DEVICES={gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "python {script} --source-run {source_run} --output-dir {output_dir} "
        "--force-hunyuan-paint-dense --world-axis-uv-normalization planar "
        "--force"
    ).format(
        conda_sh=args.conda_sh,
        env_name=args.texture_env,
        gpu=args.texture_gpu,
        script=DEFAULT_TEXTURE_SCRIPT,
        source_run=run_dir,
        output_dir=texture_dir,
    )
    run_command(["bash", "-lc", command_text], dry_run=args.dry_run)


def write_agent_trace(case: LayoutCase, run_dir: Path, args: argparse.Namespace) -> None:
    texture_dir = texture_dir_for_case(case, args) if (args.with_texture or args.texture_only) else None
    command = [
        sys.executable,
        "scripts/write_archstudio_s2_agent_trace.py",
        "--run-dir",
        str(run_dir),
    ]
    if texture_dir and (texture_dir / "texture_manifest.json").exists():
        command.extend(["--texture-dir", str(texture_dir)])
    run_command(command, dry_run=args.dry_run)


def run_case(case: LayoutCase, args: argparse.Namespace) -> None:
    run_dir = args.output_root / f"{case.run_name}{args.run_suffix}"
    print(
        "\n=== {name} parts={parts} layout_id={layout_id} run={run} ===".format(
            name=case.case_name,
            parts=case.part_count,
            layout_id=case.layout_id,
            run=run_dir,
        ),
        flush=True,
    )

    if args.report_only:
        write_agent_trace(case, run_dir, args)
        generate_report(case, run_dir, args)
        return
    if args.texture_only:
        run_texture(case, run_dir, args)
        write_agent_trace(case, run_dir, args)
        generate_report(case, run_dir, args, force=True)
        return
    if args.geometry_only:
        run_geometry(case, run_dir, args)
        write_agent_trace(case, run_dir, args)
        generate_report(case, run_dir, args)
        return
    if args.t2i_only:
        prepare_case(case, run_dir, args)
        generate_references(case, run_dir, args)
        return

    prepare_case(case, run_dir, args)
    if args.prepare_only:
        return
    generate_references(case, run_dir, args)
    run_geometry(case, run_dir, args)
    write_agent_trace(case, run_dir, args)
    generate_report(case, run_dir, args)
    if args.with_texture:
        run_texture(case, run_dir, args)
        write_agent_trace(case, run_dir, args)
        generate_report(case, run_dir, args, force=True)


def main() -> int:
    args = parse_args()
    cases = filtered_cases(load_cases(args.layout_root), args)
    if not cases:
        raise SystemExit("No layout cases matched the selection")

    args.output_root.mkdir(parents=True, exist_ok=True)
    print("ArchStudio-S2 open-semantic batch")
    print("layout_root:", args.layout_root)
    print("output_root:", args.output_root)
    print("selected:", ", ".join(case.case_name for case in cases))

    for case in cases:
        run_case(case, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
