#!/usr/bin/env python3
"""Run ArchStudio-S2 part-focused experiments over Stage1 open-semantic layouts.

This orchestrator is resumable and can run multiple scenes concurrently.  Each
scene still writes its own report as soon as it finishes, so completed runs are
immediately visible on the 9999 harness.  The texture stage is optional and can
be resumed later.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT_ROOT = Path(
    "/home/wangyz/project/0working/ArchStudio/stage1/experiment_results/"
    "stage1_quality/V9-P2d-open-semantic-ground-plane-alignment"
)
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2")
DEFAULT_TEXTURE_ROOT = Path("/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2_texture")
DEFAULT_RUN_SUFFIX = "_partfocus_v10"
DEFAULT_T2I_PYTHON = Path("/home/wangyz/anaconda3/envs/archstudio_qwen_image/bin/python")
DEFAULT_HF_HOME = Path("/mnt/data/wangyz/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--texture-root", type=Path, default=DEFAULT_TEXTURE_ROOT)
    parser.add_argument("--run-suffix", default=DEFAULT_RUN_SUFFIX)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=10)
    parser.add_argument("--gpu-list", default="0,1,2,3")
    parser.add_argument("--texture-gpu-list", default="4,5,6,7")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum concurrently running scenes. Default: number of GPUs in --gpu-list, or 1.",
    )
    parser.add_argument("--provider", default="qwen_image_local")
    parser.add_argument(
        "--fallback-provider",
        action="append",
        default=[],
        help="Fallback T2I provider after the primary provider fails. May be repeated.",
    )
    parser.add_argument("--t2i-python", type=Path, default=DEFAULT_T2I_PYTHON)
    parser.add_argument(
        "--t2i-gpu-list",
        default=None,
        help="CUDA_VISIBLE_DEVICES list for local T2I. Defaults to --gpu-list.",
    )
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--qwen-image-model-id", default="Qwen/Qwen-Image")
    parser.add_argument(
        "--siliconflow-env-file",
        type=Path,
        default=Path("/mnt/data/wangyz/BuildingBlock/.secrets/siliconflow.env"),
        help="Optional shell-style env file containing SILICONFLOW_API_KEY for SiliconFlow provider.",
    )
    parser.add_argument("--with-texture", action="store_true")
    parser.add_argument("--texture-only", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force-t2i", action="store_true")
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-texture", action="store_true")
    parser.add_argument("--force-report", action="store_true")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_cases(layout_root: Path, args: argparse.Namespace) -> list[tuple[int, Path]]:
    cases = []
    selectors = {item.strip() for item in args.case if item.strip()}
    for case_dir in sorted(layout_root.glob("[0-9][0-9]-*")):
        layout_path = case_dir / "layout_s2_open_semantic.json"
        if not layout_path.exists():
            continue
        index = int(case_dir.name.split("-", 1)[0])
        if index < args.start_index or index > args.end_index:
            continue
        aliases = {str(index), f"{index:02d}", case_dir.name}
        if selectors and not (selectors & aliases):
            continue
        cases.append((index, case_dir))
    return cases


def run_name(case_dir: Path, suffix: str) -> str:
    return "v9p2d_{}{}".format(case_dir.name.replace("-", "_"), suffix)


def texture_name(case_dir: Path, suffix: str) -> str:
    return run_name(case_dir, suffix) + "_partfocus_hy3dpaint"


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path or not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def build_case_command(
    args: argparse.Namespace,
    case_index: int,
    case_dir: Path,
    gpu: str,
    texture_gpu: str,
    t2i_gpu: str,
) -> tuple[list[str], Path, Path]:
    run_dir = args.output_root / run_name(case_dir, args.run_suffix)
    texture_dir = args.texture_root / texture_name(case_dir, args.run_suffix)
    cmd = [
        sys.executable,
        "scripts/run_open_semantic_s2_batch.py",
        "--layout-root",
        str(args.layout_root),
        "--output-root",
        str(args.output_root),
        "--texture-output-root",
        str(args.texture_root),
        "--case",
        f"{case_index:02d}",
        "--run-suffix",
        args.run_suffix,
        "--gpu",
        gpu,
        "--t2i-gpu",
        t2i_gpu,
        "--texture-gpu",
        texture_gpu,
        "--provider",
        args.provider,
        "--t2i-python",
        str(args.t2i_python),
        "--hf-endpoint",
        args.hf_endpoint,
        "--hf-home",
        str(args.hf_home),
        "--qwen-image-model-id",
        args.qwen_image_model_id,
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--retry-count",
        "0",
    ]
    for fallback_provider in args.fallback_provider:
        cmd.extend(["--fallback-provider", fallback_provider])
    if args.with_texture:
        cmd.append("--with-texture")
    if args.texture_only:
        cmd.append("--texture-only")
    if args.geometry_only:
        cmd.append("--geometry-only")
    if args.report_only:
        cmd.append("--report-only")
    if args.force_t2i:
        cmd.append("--force-t2i")
    if args.force_geometry:
        cmd.append("--force-geometry")
    if args.force_texture:
        cmd.append("--force-texture")
    if args.force_report:
        cmd.append("--force-report")
    return cmd, run_dir, texture_dir


def main() -> int:
    args = parse_args()
    cases = discover_cases(args.layout_root, args)
    if not cases:
        raise SystemExit("no cases selected")

    gpu_list = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    texture_gpu_list = [item.strip() for item in args.texture_gpu_list.split(",") if item.strip()]
    env = dict(os.environ)
    for key, value in load_env_file(args.siliconflow_env_file).items():
        env.setdefault(key, value)
    provider_needs_siliconflow = args.provider == "siliconflow_qwen_image" or any(
        provider == "siliconflow_qwen_image" for provider in args.fallback_provider
    )
    if provider_needs_siliconflow and not env.get("SILICONFLOW_API_KEY") and not (
        args.dry_run or args.report_only or args.geometry_only or args.texture_only
    ):
        raise SystemExit(
            "SILICONFLOW_API_KEY is not set. Export it or provide --siliconflow-env-file."
        )

    index_rows = []
    pending = []
    for ordinal, (case_index, case_dir) in enumerate(cases):
        t2i_gpu_list = [
            item.strip()
            for item in (args.t2i_gpu_list or args.gpu_list).split(",")
            if item.strip()
        ]
        gpu = gpu_list[ordinal % len(gpu_list)] if gpu_list else "0"
        t2i_gpu = t2i_gpu_list[ordinal % len(t2i_gpu_list)] if t2i_gpu_list else gpu
        texture_gpu = texture_gpu_list[ordinal % len(texture_gpu_list)] if texture_gpu_list else gpu
        cmd, run_dir, texture_dir = build_case_command(args, case_index, case_dir, gpu, texture_gpu, t2i_gpu)
        pending.append((case_dir, cmd, run_dir, texture_dir))
        index_rows.append(
            {
                "case": case_dir.name,
                "status": "pending",
                "geometry_run": str(run_dir),
                "geometry_report": str(run_dir / "report" / "index.html"),
                "texture_run": str(texture_dir),
                "texture_report": str(texture_dir / "report" / "index.html"),
            }
        )
        write_index(args.output_root / f"archstudio_s2{args.run_suffix}_10scene_index.html", index_rows)

    if args.dry_run:
        for _case_dir, cmd, _run_dir, _texture_dir in pending:
            run_command(cmd, env=env, dry_run=True)
        return 0

    max_parallel = args.max_parallel
    if max_parallel is None:
        max_parallel = max(1, len(gpu_list) if gpu_list else 1)
    max_parallel = max(1, max_parallel)
    running: list[dict[str, object]] = []
    next_index = 0
    failures: list[tuple[str, int]] = []
    index_path = args.output_root / f"archstudio_s2{args.run_suffix}_10scene_index.html"

    def update_status(case_name: str, status: str) -> None:
        for row in index_rows:
            if row["case"] == case_name:
                row["status"] = status
                break
        write_index(index_path, index_rows)

    while next_index < len(pending) or running:
        while next_index < len(pending) and len(running) < max_parallel:
            case_dir, cmd, _run_dir, _texture_dir = pending[next_index]
            next_index += 1
            print("\n$ " + " ".join(cmd), flush=True)
            update_status(case_dir.name, "running")
            process = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            running.append({"case": case_dir, "process": process})

        time.sleep(5)
        still_running = []
        for item in running:
            process = item["process"]
            case_dir = item["case"]
            assert isinstance(process, subprocess.Popen)
            assert isinstance(case_dir, Path)
            returncode = process.poll()
            if returncode is None:
                still_running.append(item)
                continue
            if returncode == 0:
                update_status(case_dir.name, "complete")
                print(f"[{case_dir.name}] complete", flush=True)
            else:
                update_status(case_dir.name, f"failed({returncode})")
                failures.append((case_dir.name, int(returncode)))
                print(f"[{case_dir.name}] failed returncode={returncode}", flush=True)
        running = still_running

    if failures:
        failed_text = ", ".join(f"{name}:{code}" for name, code in failures)
        raise SystemExit(f"failed cases: {failed_text}")
    return 0


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html_rows = []
    for row in rows:
        geom = Path(row["geometry_report"])
        tex = Path(row["texture_report"])
        status = row.get("status", "")
        html_rows.append(
            "<tr>"
            f"<td>{row['case']}</td>"
            f"<td>{status}</td>"
            f"<td>{'<a href=\"' + geom.as_uri() + '\">file</a>' if geom.exists() else 'pending'}</td>"
            f"<td>{'<a href=\"' + tex.as_uri() + '\">file</a>' if tex.exists() else 'pending'}</td>"
            f"<td><code>{row['geometry_run']}</code></td>"
            "</tr>"
        )
    path.write_text(
        """<!doctype html><meta charset='utf-8'><title>ArchStudio-S2 10-scene partfocus index</title>
        <style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f7f7f5}table{border-collapse:collapse;width:100%;background:#fff}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}code{font-size:12px}</style>
        <h1>ArchStudio-S2 · 10-scene part-focused V10 index</h1>
        <p>每个 scene 完成后会单独出现在 9999 的 ArchStudio-S2 项目列表里。</p>
        <table><thead><tr><th>case</th><th>status</th><th>geometry report</th><th>texture report</th><th>geometry run</th></tr></thead><tbody>"""
        + "".join(html_rows)
        + "</tbody></table>\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
