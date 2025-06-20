"""Command line interface for stitcher."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import Stitcher, StitchError
from .utils import parse_roi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m stitcher")
    parser.add_argument("input_dir", type=Path, help="Directory with screenshots")
    parser.add_argument("-o", "--output", type=Path, help="Output image path")
    parser.add_argument("--roi", type=str, help="ROI as x,y,w,h")
    parser.add_argument("--threshold", type=float, default=0.8, help="Match threshold")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stitcher = Stitcher(threshold=args.threshold)
    roi = parse_roi(args.roi) if args.roi else None
    try:
        out = stitcher.stitch(args.input_dir, args.output, roi)
    except FileNotFoundError:
        return 11
    except StitchError:
        return 10
    except Exception:
        return 12
    print(out)
    return 0
