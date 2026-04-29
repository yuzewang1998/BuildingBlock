#!/usr/bin/env python
"""Generate a static HTML report for a BuildingBlock-Hunyuan run."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_synthesis.building_mesh.report_bundle import build_report


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report_dir = build_report(args.run_dir)
    print("report_dir", report_dir)
    print("index_html", report_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
