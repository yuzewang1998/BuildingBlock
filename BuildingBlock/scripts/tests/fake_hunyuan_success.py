#!/usr/bin/env python
"""Small fake Hunyuan runner for local integration checks."""

import argparse
from pathlib import Path


def write_unit_obj(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v -0.5 -0.5 -0.5",
                "v 0.5 -0.5 -0.5",
                "v 0.5 0.5 -0.5",
                "v -0.5 0.5 -0.5",
                "v -0.5 -0.5 0.5",
                "v 0.5 -0.5 0.5",
                "v 0.5 0.5 0.5",
                "v -0.5 0.5 0.5",
                "f 1 2 3 4",
                "f 5 8 7 6",
                "f 1 5 6 2",
                "f 2 6 7 3",
                "f 3 7 8 4",
                "f 4 8 5 1",
                "",
            ]
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--part-id", required=True)
    args = parser.parse_args()
    write_unit_obj(Path(args.output_dir) / "{}.obj".format(args.part_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
