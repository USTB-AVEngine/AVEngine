"""Cut the canonical 2D candidate out of its background with the local ISNet model.

The mesh stage skips its own background remover when the input already carries a
real alpha channel, which keeps a gated remote model out of the chain.  The
matte is produced by the ISNet general-use ONNX weights that already sit in the
shared model directory, so nothing is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

DEFAULT_MODEL = Path("/data/models/rembg/isnet-general-use/isnet-general-use.onnx")
INPUT_SIZE = 1024
MEAN = 0.5
STD = 1.0


class SegmentationError(RuntimeError):
    """The cut-out cannot be produced as specified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh RGBA png path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--alpha-floor", type=float, default=0.5,
                        help="matte values below this fraction are forced transparent")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output.exists():
            raise SegmentationError(f"{args.output} exists; choose a fresh path")
        if not args.image.is_file():
            raise SegmentationError(f"image not found: {args.image}")
        if not args.model.is_file():
            raise SegmentationError(f"ISNet weights not found: {args.model}")
    except SegmentationError as error:
        print(f"segmentation refused: {error}", file=sys.stderr)
        return 2

    import numpy as np
    import onnxruntime
    from PIL import Image

    with Image.open(args.image) as opened:
        opened.load()
        source = opened.convert("RGB")

    resized = source.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    tensor = array.transpose(2, 0, 1)[None, ...]

    session = onnxruntime.InferenceSession(
        str(args.model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    matte = outputs[0][0][0]
    span = float(matte.max() - matte.min())
    if span <= 0:
        print("segmentation refused: the matte is constant", file=sys.stderr)
        return 2
    matte = (matte - matte.min()) / span
    matte = np.where(matte < args.alpha_floor, 0.0, matte)

    alpha = Image.fromarray((matte * 255).astype(np.uint8)).resize(source.size, Image.Resampling.LANCZOS)
    cut = source.copy()
    cut.putalpha(alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cut.save(args.output)

    coverage = float((np.asarray(alpha, dtype=np.float32) / 255.0 > 0.5).mean())
    manifest = {
        "schema": "avengine_canonical_2d_cutout_v1",
        "source": {"path": str(args.image), "sha256": _sha256_file(args.image)},
        "model": {"path": str(args.model), "sha256": _sha256_file(args.model)},
        "alpha_floor": args.alpha_floor,
        "foreground_coverage": coverage,
        "output": {"path": str(args.output), "sha256": _sha256_file(args.output)},
    }
    args.output.with_suffix(".cutout_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {args.output} (foreground {coverage:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
