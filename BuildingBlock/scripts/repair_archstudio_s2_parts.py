#!/usr/bin/env python3
"""Create a dry-run ArchStudio-S2 V9 local repair plan.

This first implementation does not execute T2I/Hunyuan/Paint.  It consumes
``quality_findings.json`` and ``metrics_summary.json`` and writes a traceable,
additive repair plan that later executor code can use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.quality_agent import (  # noqa: E402
    DEFAULT_QUALITY_SUBDIR,
    build_repair_plan,
    load_json,
    make_quality_summary,
    write_json,
    write_quality_trace,
)
from scene_synthesis.building_mesh.report_bundle import build_report, refresh_quality_agent_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--quality-dir", type=Path, default=None)
    parser.add_argument("--max-repair-parts", type=int, default=5)
    parser.add_argument("--max-attempts-per-part", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--fast-report-refresh",
        action="store_true",
        help="Update only the V9 QA section in an existing report instead of rebuilding mesh assets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    quality_dir = (args.quality_dir or (run_dir / DEFAULT_QUALITY_SUBDIR)).resolve()
    metrics_path = quality_dir / "metrics_summary.json"
    findings_path = quality_dir / "quality_findings.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics_summary.json: {metrics_path}")
    if not findings_path.exists():
        raise FileNotFoundError(f"missing quality_findings.json: {findings_path}")

    metrics = load_json(metrics_path)
    findings = load_json(findings_path)
    plan = build_repair_plan(
        findings,
        metrics,
        max_parts=args.max_repair_parts,
        max_attempts_per_part=args.max_attempts_per_part,
        dry_run=True,
    )
    write_json(quality_dir / "repair_plan.json", plan)
    trace = write_quality_trace(
        quality_dir,
        run_dir=run_dir,
        texture_run_dir=metrics.get("texture_run_dir"),
        metrics=metrics,
        evidence=load_json(quality_dir / "evidence_manifest.json") if (quality_dir / "evidence_manifest.json").exists() else None,
        findings=findings,
        repair_plan=plan,
        provider="repair-dry-run",
    )
    summary = make_quality_summary(
        run_dir,
        quality_dir,
        metrics,
        load_json(quality_dir / "evidence_manifest.json") if (quality_dir / "evidence_manifest.json").exists() else None,
        findings,
        plan,
    )
    write_json(quality_dir / "quality_summary.json", summary)
    if not args.no_report:
        if args.fast_report_refresh:
            refresh_quality_agent_report(run_dir)
        else:
            build_report(run_dir)
    print(json.dumps({
        "quality_dir": str(quality_dir),
        "repair_plan": str(quality_dir / "repair_plan.json"),
        "planned_actions": plan["num_planned_actions"],
        "trace": str(quality_dir / "quality_agent_trace.json"),
        "trace_stop_condition": trace["stop_condition"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
