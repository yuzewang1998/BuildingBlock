#!/usr/bin/env python3
"""Run ArchStudio-S2 geometry-only repair agent over Stage1 open-semantic cases.

This wrapper is resumable and report-first: each case writes/refreshes its own
S2 report and agent_loop/index.html as soon as it completes so the 9999 harness
can expose partial progress.  It can either reuse existing geometry runs or run
prepare/T2I/geometry first via run_open_semantic_s2_batch.py.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT_ROOT = Path(
    "/home/wangyz/project/0working/ArchStudio/stage1/experiment_results/"
    "stage1_quality/V9-P2d-open-semantic-ground-plane-alignment"
)
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2_agent_loop")
DEFAULT_EXISTING_OUTPUT_ROOT = Path("/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2")
DEFAULT_RUN_SUFFIX = "_agentloop_v1"
DEFAULT_T2I_PYTHON = Path("/home/wangyz/anaconda3/envs/archstudio_qwen_image/bin/python")
DEFAULT_HF_HOME = Path("/mnt/data/wangyz/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--existing-output-root",
        type=Path,
        default=DEFAULT_EXISTING_OUTPUT_ROOT,
        help="Root for existing geometry runs when --reuse-existing-run-suffix is set.",
    )
    parser.add_argument("--run-suffix", default=DEFAULT_RUN_SUFFIX)
    parser.add_argument(
        "--reuse-existing-run-suffix",
        default="_partfocus_v10",
        help="Copy/reuse existing geometry run suffix from --existing-output-root. Empty string disables reuse.",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu-list", default="0,1,2,3")
    parser.add_argument("--t2i-gpu-list", default=None)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--provider", default="qwen_image_local")
    parser.add_argument("--t2i-python", type=Path, default=DEFAULT_T2I_PYTHON)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--qwen-image-model-id", default="Qwen/Qwen-Image")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-no-improvement-iterations", type=int, default=2)
    parser.add_argument("--acceptance-score", type=float, default=0.90)
    parser.add_argument("--reject-severity", type=int, default=3)
    parser.add_argument(
        "--evaluator-provider",
        default="metrics",
        choices=["metrics", "mock", "openai", "aimirror", "aicodemirror", "gpt", "gpt55", "multimodal"],
    )
    parser.add_argument("--evaluator-model", default="gpt-5.5")
    parser.add_argument("--evaluator-max-images", type=int, default=12)
    parser.add_argument("--evaluator-mock-response", default=None)
    parser.add_argument("--evaluator-build-evidence", action="store_true", help="Force evidence render even for mock evaluator.")
    parser.add_argument("--evaluator-no-evidence", action="store_true", help="Skip visual evidence render for VLM smoke/debug.")
    parser.add_argument("--evaluator-full-evidence-render", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent-dry-run", action="store_true", help="Run agent planning/trace without model subprocess writes.")
    parser.add_argument("--skip-generate", action="store_true", help="Do not call initial S2 generation; require/reuse existing runs.")
    parser.add_argument("--force-initial", action="store_true")
    parser.add_argument("--force-agent", action="store_true", help="Run agent even if agent_loop/agent_trace.json exists.")
    parser.add_argument("--fast-report-refresh", action="store_true")
    return parser.parse_args()


def discover_cases(args: argparse.Namespace) -> List[tuple[int, Path]]:
    selectors = {item.strip() for item in args.case if item.strip()}
    cases: List[tuple[int, Path]] = []
    for case_dir in sorted(args.layout_root.glob("[0-9][0-9]-*")):
        if not (case_dir / "layout_s2_open_semantic.json").exists():
            continue
        index = int(case_dir.name.split("-", 1)[0])
        if index < args.start_index or index > args.end_index:
            continue
        aliases = {str(index), f"{index:02d}", case_dir.name}
        if selectors and not (selectors & aliases):
            continue
        cases.append((index, case_dir))
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def base_run_name(case_dir: Path, suffix: str) -> str:
    return "v9p2d_{}{}".format(case_dir.name.replace("-", "_"), suffix)


def shell_join(command: Sequence[str]) -> str:
    return " ".join(subprocess.list2cmdline([token]) for token in command)


def run_command(command: Sequence[str], *, env: Dict[str, str], dry_run: bool) -> int:
    print("\n$ " + shell_join(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=str(REPO_ROOT), env=env).returncode


def env_for_case(args: argparse.Namespace, gpu: str, t2i_gpu: str) -> Dict[str, str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}:{pythonpath}"
    env["HF_ENDPOINT"] = args.hf_endpoint
    env["HF_HOME"] = str(args.hf_home)
    env["HUGGINGFACE_HUB_CACHE"] = str(args.hf_home / "hub")
    env["QWEN_IMAGE_MODEL_ID"] = args.qwen_image_model_id
    env["QWEN_IMAGE_DEVICE"] = "cuda"
    env["CUDA_VISIBLE_DEVICES"] = str(t2i_gpu or gpu)
    return env


def initial_generation_command(args: argparse.Namespace, case_index: int, gpu: str, t2i_gpu: str, run_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "scripts/run_open_semantic_s2_batch.py",
        "--layout-root",
        str(args.layout_root),
        "--output-root",
        str(args.output_root),
        "--case",
        f"{case_index:02d}",
        "--run-suffix",
        args.run_suffix,
        "--gpu",
        gpu,
        "--t2i-gpu",
        t2i_gpu,
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
    if args.force_initial:
        command.extend(["--force-t2i", "--force-geometry", "--force-report"])
    return command


def ensure_reused_run(args: argparse.Namespace, case_dir: Path, run_dir: Path) -> Dict[str, Any]:
    if not args.reuse_existing_run_suffix:
        return {"reused": False}
    source = args.existing_output_root / base_run_name(case_dir, args.reuse_existing_run_suffix)
    if not source.exists():
        return {"reused": False, "source": str(source), "reason": "source_missing"}
    if run_dir.exists():
        return {"reused": False, "source": str(source), "reason": "target_exists"}
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(source, run_dir, symlinks=True)
    return {"reused": True, "source": str(source), "target": str(run_dir)}


def agent_command(args: argparse.Namespace, run_dir: Path, gpu: str, t2i_gpu: str) -> List[str]:
    command = [
        sys.executable,
        "scripts/run_archstudio_s2_agent_loop.py",
        "--run-dir",
        str(run_dir),
        "--max-iterations",
        str(args.max_iterations),
        "--max-no-improvement-iterations",
        str(args.max_no_improvement_iterations),
        "--acceptance-score",
        str(args.acceptance_score),
        "--reject-severity",
        str(args.reject_severity),
        "--evaluator-provider",
        args.evaluator_provider,
        "--evaluator-model",
        args.evaluator_model,
        "--evaluator-max-images",
        str(args.evaluator_max_images),
        "--provider",
        args.provider,
        "--t2i-python",
        str(args.t2i_python),
        "--t2i-gpu",
        str(t2i_gpu),
        "--hunyuan-gpu",
        str(gpu),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.evaluator_mock_response:
        command.extend(["--evaluator-mock-response", args.evaluator_mock_response])
    if args.evaluator_build_evidence:
        command.append("--evaluator-build-evidence")
    if args.evaluator_no_evidence:
        command.append("--evaluator-no-evidence")
    if args.evaluator_full_evidence_render:
        command.append("--evaluator-full-evidence-render")
    if args.agent_dry_run:
        command.append("--dry-run")
    if args.fast_report_refresh:
        command.append("--fast-report-refresh")
    return command


def write_index(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html_rows = []
    for row in rows:
        report = Path(str(row.get("report", "")))
        agent = Path(str(row.get("agent_report", "")))
        trace = Path(str(row.get("trace", "")))
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('case', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('stop_condition', '')))}</td>"
            f"<td>{html.escape(str(row.get('accepted', '')))}</td>"
            f"<td>{html.escape(str(row.get('score', '')))}</td>"
            f"<td>{'<a href="' + report.as_uri() + '">report</a>' if report.exists() else 'pending'}</td>"
            f"<td>{'<a href="' + agent.as_uri() + '">agent</a>' if agent.exists() else 'pending'}</td>"
            f"<td>{'<a href="' + trace.as_uri() + '">trace</a>' if trace.exists() else 'pending'}</td>"
            f"<td><code>{html.escape(str(row.get('run_dir', '')))}</code></td>"
            "</tr>"
        )
    path.write_text(
        """<!doctype html><meta charset='utf-8'><title>ArchStudio-S2 geometry agent batch</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f7f7f5;color:#1f2630}table{border-collapse:collapse;width:100%;background:#fff}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left;vertical-align:top}th{color:#46628d;text-transform:uppercase;font-size:12px}code{font-size:12px}</style>
<h1>ArchStudio-S2 · geometry-only VLM agent batch</h1>
<p>每个 case 完成后会单独出现在 9999 的 ArchStudio-S2 项目列表里；本页是批量状态索引。</p>
<table><thead><tr><th>case</th><th>status</th><th>stop</th><th>accepted</th><th>score</th><th>S2 report</th><th>agent report</th><th>trace</th><th>run dir</th></tr></thead><tbody>"""
        + "".join(html_rows)
        + "</tbody></table>\n",
        encoding="utf-8",
    )


def read_agent_result(stdout_path: Path, run_dir: Path) -> Dict[str, Any]:
    trace_path = run_dir / "agent_loop" / "agent_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
    return {
        "accepted": trace.get("accepted"),
        "stop_condition": trace.get("stop_condition"),
        "score": trace.get("final_scene_score"),
        "trace": str(trace_path),
        "report": str(run_dir / "report" / "index.html"),
        "agent_report": str(run_dir / "agent_loop" / "index.html"),
        "stdout": str(stdout_path),
    }


def main() -> int:
    args = parse_args()
    cases = discover_cases(args)
    if not cases:
        raise SystemExit("no cases selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    gpu_list = [item.strip() for item in args.gpu_list.split(",") if item.strip()] or ["0"]
    t2i_gpu_list = [item.strip() for item in (args.t2i_gpu_list or args.gpu_list).split(",") if item.strip()] or gpu_list
    index_path = args.output_root / f"archstudio_s2_geometry_agent{args.run_suffix}_index.html"
    rows: List[Dict[str, Any]] = []
    for ordinal, (case_index, case_dir) in enumerate(cases):
        run_dir = args.output_root / base_run_name(case_dir, args.run_suffix)
        rows.append({
            "case": case_dir.name,
            "status": "pending",
            "run_dir": str(run_dir),
            "report": str(run_dir / "report" / "index.html"),
            "agent_report": str(run_dir / "agent_loop" / "index.html"),
            "trace": str(run_dir / "agent_loop" / "agent_trace.json"),
        })
    write_index(index_path, rows)

    failures: List[str] = []
    for ordinal, (case_index, case_dir) in enumerate(cases):
        row = rows[ordinal]
        run_dir = Path(str(row["run_dir"]))
        gpu = gpu_list[ordinal % len(gpu_list)]
        t2i_gpu = t2i_gpu_list[ordinal % len(t2i_gpu_list)]
        env = env_for_case(args, gpu, t2i_gpu)
        row["status"] = "running"
        write_index(index_path, rows)
        try:
            reuse_info = ensure_reused_run(args, case_dir, run_dir)
            row["reuse"] = reuse_info
            if not args.skip_generate and not (run_dir / "manifest.json").exists():
                rc = run_command(initial_generation_command(args, case_index, gpu, t2i_gpu, run_dir), env=env, dry_run=args.dry_run)
                if rc != 0:
                    raise RuntimeError(f"initial_generation_failed:{rc}")
            trace_path = run_dir / "agent_loop" / "agent_trace.json"
            if trace_path.exists() and not args.force_agent:
                row.update(read_agent_result(run_dir / "agent_loop" / "agent_stdout.json", run_dir))
                row["status"] = "complete(existing-agent)"
                write_index(index_path, rows)
                continue
            if not (run_dir / "manifest.json").exists():
                raise RuntimeError("missing manifest.json after reuse/generation")
            stdout_path = run_dir / "agent_loop" / "agent_stdout.json"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = agent_command(args, run_dir, gpu, t2i_gpu)
            print("\n$ " + shell_join(cmd), flush=True)
            if args.dry_run:
                rc = 0
            else:
                completed = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
                stdout_path.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
                rc = completed.returncode
            if rc != 0:
                raise RuntimeError(f"agent_failed:{rc}")
            row.update(read_agent_result(stdout_path, run_dir))
            row["status"] = "complete"
        except Exception as exc:  # noqa: BLE001 - keep batch diagnosable.
            row["status"] = f"failed({exc})"
            failures.append(f"{case_dir.name}:{exc}")
        write_index(index_path, rows)
        time.sleep(0.2)
    if failures:
        raise SystemExit("failed cases: " + ", ".join(failures))
    print(json.dumps({"index": str(index_path), "cases": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
