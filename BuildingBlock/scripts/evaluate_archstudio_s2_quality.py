#!/usr/bin/env python3
"""Build ArchStudio-S2 V9 quality evidence, metrics, and agent findings.

By default this evaluator is deterministic/metrics-only.  Use ``--provider
mock`` for schema/testing flow or ``--provider openai``/``aimirror`` for the
multimodal agent evaluator.  It never invokes T2I, Hunyuan-Omni, or Paint.
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
    build_evidence_pack,
    build_repair_plan,
    collect_quality_metrics,
    evaluate_quality,
    make_quality_summary,
    write_json,
    write_quality_trace,
)
from scene_synthesis.building_mesh.report_bundle import build_report, refresh_quality_agent_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Existing ArchStudio-S2 geometry run directory")
    parser.add_argument("--texture-run-dir", type=Path, default=None, help="Optional downstream texture run directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Quality output dir; default <run-dir>/quality/v9_agent")
    parser.add_argument(
        "--provider",
        choices=["metrics", "mock", "openai", "aimirror", "aicodemirror", "gpt", "gpt55", "multimodal"],
        default="metrics",
        help="Evaluator provider. metrics is deterministic-only; openai/aimirror calls a multimodal model.",
    )
    parser.add_argument("--metrics-only", action="store_true", help="Only collect metrics/evidence; skip findings and repair plan")
    parser.add_argument("--mock-response", default=None, help="Optional JSON file/string for mock evaluator findings")
    parser.add_argument("--model", default=None, help="Model name for openai/aimirror providers")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for openai/aimirror providers")
    parser.add_argument("--auth-path", type=Path, default=None, help="Optional auth JSON path containing OPENAI_API_KEY")
    parser.add_argument("--max-images", type=int, default=12, help="Max evidence images sent to multimodal evaluator")
    parser.add_argument("--no-evidence", action="store_true", help="Skip visual evidence rendering")
    parser.add_argument("--no-report", action="store_true", help="Do not rebuild report/index.html")
    parser.add_argument(
        "--fast-report-refresh",
        action="store_true",
        help="Update only the V9 QA section in an existing report instead of rebuilding mesh assets",
    )
    parser.add_argument("--plan-repairs", action="store_true", help="Also write a dry-run repair_plan.json")
    parser.add_argument("--max-repair-parts", type=int, default=5)
    parser.add_argument("--max-attempts-per-part", type=int, default=2)
    parser.add_argument(
        "--max-sample-vertices",
        type=int,
        default=50000,
        help="Per-mesh vertex sample cap for percentile bounds; exact min/max and face count are still streamed",
    )
    parser.add_argument(
        "--deep-mesh-components",
        action="store_true",
        help="Slow path: load dense OBJ faces and compute connected components",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    quality_dir = (args.output_dir or (run_dir / DEFAULT_QUALITY_SUBDIR)).resolve()
    quality_dir.mkdir(parents=True, exist_ok=True)

    metrics = collect_quality_metrics(
        run_dir,
        args.texture_run_dir,
        max_sample_vertices=args.max_sample_vertices,
        deep_mesh_components=args.deep_mesh_components,
    )
    write_json(quality_dir / "metrics_summary.json", metrics)

    evidence = None
    if not args.no_evidence:
        evidence = build_evidence_pack(run_dir, quality_dir, args.texture_run_dir)
        write_json(quality_dir / "evidence_manifest.json", evidence)

    findings = None
    if not args.metrics_only:
        findings = evaluate_quality(
            metrics,
            evidence,
            provider=args.provider,
            mock_response=args.mock_response,
            model=args.model,
            base_url=args.base_url,
            auth_path=args.auth_path,
            max_images=args.max_images,
        )
    if findings:
        write_json(quality_dir / "quality_findings.json", findings)

    repair_plan = None
    if args.plan_repairs and findings:
        repair_plan = build_repair_plan(
            findings,
            metrics,
            max_parts=args.max_repair_parts,
            max_attempts_per_part=args.max_attempts_per_part,
            dry_run=True,
        )
        write_json(quality_dir / "repair_plan.json", repair_plan)

    write_quality_trace(
        quality_dir,
        run_dir=run_dir,
        texture_run_dir=args.texture_run_dir,
        metrics=metrics,
        evidence=evidence,
        findings=findings,
        repair_plan=repair_plan,
        provider=args.provider if not args.metrics_only else "metrics-only",
    )
    summary = make_quality_summary(run_dir, quality_dir, metrics, evidence, findings, repair_plan)
    write_json(quality_dir / "quality_summary.json", summary)

    if not args.no_report:
        if args.fast_report_refresh:
            refresh_quality_agent_report(run_dir)
        else:
            build_report(run_dir)

    print(json.dumps({
        "quality_dir": str(quality_dir),
        "metrics": str(quality_dir / "metrics_summary.json"),
        "findings": str(quality_dir / "quality_findings.json") if findings else None,
        "repair_plan": str(quality_dir / "repair_plan.json") if repair_plan else None,
        "report": str(run_dir / "report" / "index.html"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
