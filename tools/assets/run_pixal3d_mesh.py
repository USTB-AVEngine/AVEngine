"""Run the vendored Pixal3D image-to-3D stage on an already-matted candidate.

Pixal3D builds its own background remover eagerly at pipeline construction, and
that remover is a gated remote model this project has no access to.  The mesh
stage never needs it here: the candidate is matted beforehand with the local
ISNet weights, and Pixal3D uses an existing alpha channel directly.  This runner
therefore replaces the remover factory with one that yields nothing, then
executes the vendored entry point unchanged -- no file in the vendored tree is
edited.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixal3d-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="RGBA cut-out")
    parser.add_argument("--output", type=Path, required=True, help="fresh .glb path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--fov", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        print(f"mesh refused: {args.output} exists; choose a fresh path", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"mesh refused: cut-out not found: {args.image}", file=sys.stderr)
        return 2

    from PIL import Image

    with Image.open(args.image) as opened:
        if opened.mode != "RGBA":
            print("mesh refused: the input must carry an alpha channel", file=sys.stderr)
            return 2

    root = args.pixal3d_root.resolve()
    sys.path.insert(0, str(root))
    from pixal3d.pipelines import rembg as vendored_rembg

    def _no_remover(*_args, **_kwargs):
        return None

    for name in dir(vendored_rembg):
        if name.startswith("_"):
            continue
        if isinstance(getattr(vendored_rembg, name), type):
            setattr(vendored_rembg, name, _no_remover)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    argv_for_entry = [
        "inference.py",
        "--image", str(args.image),
        "--output", str(args.output),
        "--seed", str(args.seed),
    ]
    if args.resolution is not None:
        argv_for_entry += ["--resolution", str(args.resolution)]
    if args.fov is not None:
        argv_for_entry += ["--fov", str(args.fov)]
    sys.argv = argv_for_entry
    runpy.run_path(str(root / "inference.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
