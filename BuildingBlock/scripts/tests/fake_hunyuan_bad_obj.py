#!/usr/bin/env python
"""Fake runner that emits an invalid OBJ for normalization-failure checks."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--part-id", required=True)
    args = parser.parse_args()
    output_path = Path(args.output_dir) / "{}.obj".format(args.part_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("not an obj mesh\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
