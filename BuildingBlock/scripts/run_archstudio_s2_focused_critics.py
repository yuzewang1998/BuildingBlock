#!/usr/bin/env python3
"""Run focused ArchStudio-S2 GPT-5.5 visual critics on an existing run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.quality_agent import DEFAULT_AICODEMIRROR_BASE_URL, run_focused_visual_critics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--provider", default="gpt55")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default=DEFAULT_AICODEMIRROR_BASE_URL)
    parser.add_argument("--auth-path", type=Path, default=Path.home() / ".codex" / "provider_snapshots" / "auth_aicodemirror.json")
    parser.add_argument("--max-parts", type=int, default=0, help="0 means audit every part.")
    parser.add_argument("--part-id", action="append", default=[])
    parser.add_argument(
        "--scope",
        action="append",
        choices=["scene_assembly", "part_image", "part_mesh"],
        default=[],
        help="Repeat to run a subset. Default runs all focused scopes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_focused_visual_critics(
        args.run_dir,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        auth_path=args.auth_path,
        output_dir=args.output_dir,
        max_parts=args.max_parts,
        part_ids=args.part_id or None,
        scopes=args.scope or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
