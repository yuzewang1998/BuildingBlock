#!/usr/bin/env python3
"""Write the lightweight ArchStudio-S2 per-part agent trace for a run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.s2_agent_trace import (  # noqa: E402
    copy_trace_to_texture_dir,
    write_agent_trace,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--texture-dir", type=Path, default=None)
    args = parser.parse_args()
    trace_path = write_agent_trace(args.run_dir, args.texture_dir)
    print("s2_agent_trace", trace_path)
    if args.texture_dir:
        copied = copy_trace_to_texture_dir(args.run_dir, args.texture_dir)
        if copied:
            print("s2_agent_trace_texture_copy", copied)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
