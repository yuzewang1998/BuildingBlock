#!/usr/bin/env python
"""Run the BuildingBlock layout-to-mesh V1 pipeline for one layout JSON."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh import (
    HunyuanAdapter,
    HunyuanAdapterPolicy,
    build_assemblies,
    prepare_run_directories,
    write_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a versioned layout-to-mesh contract and prepare downstream "
            "run directories for Hunyuan and assembly stages."
        )
    )
    parser.add_argument("layout_json", type=Path, help="Raw BuildingBlock layout JSON")
    parser.add_argument(
        "--contract-file",
        type=Path,
        default=None,
        help="Use an existing layout mesh contract instead of emitting one from layout_json",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Output run directory for contract, per-part requests, and assemblies",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty run directory",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only emit the versioned contract and run directory structure",
    )
    parser.add_argument(
        "--hunyuan-command",
        default=None,
        help=(
            "Shell command template for official Hunyuan bbox inference. "
            "Supports $contract_path, $output_dir, $part_id, $target_prompt, $target_class."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20 * 60,
        help="Per-part Hunyuan timeout in seconds",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=1,
        help="Retry count after the initial failed Hunyuan attempt",
    )
    return parser


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    if args.contract_file:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        contract = json.loads(args.contract_file.read_text())
        result = {
            "run_dir": str(args.run_dir.resolve()),
            "contract_path": str(args.contract_file.resolve()),
            "layout_id": contract["layout_id"],
            "part_ids": [part["part_id"] for part in contract["parts"]],
        }
    else:
        result = prepare_run_directories(
            raw_layout_path=args.layout_json,
            run_dir=args.run_dir,
            overwrite=args.force,
        )

    print("prepared run directory:", result["run_dir"])
    print("layout id:", result["layout_id"])
    print("contract path:", result["contract_path"])
    print("parts prepared:", len(result["part_ids"]))

    if args.prepare_only:
        return 0

    if not args.hunyuan_command:
        raise SystemExit(
            "--hunyuan-command is required unless --prepare-only is used"
        )

    policy = HunyuanAdapterPolicy(
        timeout_seconds=args.timeout_seconds,
        retry_count=args.retry_count,
        concurrency=1,
    )
    adapter = HunyuanAdapter(
        command_template=args.hunyuan_command,
        output_root=Path(result["run_dir"]) / "parts",
        policy=policy,
    )

    contract = json.loads(Path(result["contract_path"]).read_text())
    contract_parts = contract["parts"]
    hunyuan_results = adapter.run_many(contract_parts)
    assembly_result = build_assemblies(
        contract_parts,
        hunyuan_results,
        Path(result["run_dir"]) / "assemblies",
    )
    report_paths = write_reports(
        hunyuan_results,
        assembly_result,
        Path(result["run_dir"]),
    )

    print("raw assembly:", assembly_result.raw_assembly_path)
    print("placeholder assembly:", assembly_result.placeholder_assembly_path)
    print("manifest:", report_paths["manifest_path"])
    print("failures:", report_paths["failures_path"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
