"""Run the local AVEngine Pixal3D image-to-3D stage on an RGBA candidate.

The selected Pixal3D inference closure lives under src/avengine/assets.
All model roots are explicit local directories or files resolved through the
AVEngine model-root registry; remote Hugging Face and Torch Hub fallbacks are
disabled. The input must already contain a non-opaque alpha channel, so this
entry point does not load a background-removal model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY = Path(__file__).resolve().parents[2]
# Production inference is local-only: prevent a missing file from
# turning into an implicit Hugging Face/Datasets network lookup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
SOURCE_ROOT = REPOSITORY / "src"
ASSET_ROOT = SOURCE_ROOT / "avengine/assets"
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (ASSET_ROOT, SOURCE_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_roots import resolve as resolve_model_root  # noqa: E402
from pixal3d_transform_profile import (  # noqa: E402
    DEFAULT_PROFILE,
    load_profile,
    mesh_export_matrix,
)

MODEL_ROOT_NAMES = {
    "pixal3d": "pixal3d",
    "moge": "moge_2_vitl",
    "dinov3": "dinov3_vitl16",
    "naf": "naf_upsampler",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="local Pixal3D model snapshot; defaults to model_roots_v1.json",
    )
    parser.add_argument(
        "--moge-root",
        type=Path,
        default=None,
        help="local MoGe model snapshot; defaults to model_roots_v1.json",
    )
    parser.add_argument(
        "--dinov3-root",
        type=Path,
        default=None,
        help="local DINOv3 model snapshot; defaults to model_roots_v1.json",
    )
    parser.add_argument(
        "--naf-root",
        type=Path,
        default=None,
        help="local NAF checkpoint; defaults to model_roots_v1.json",
    )
    parser.add_argument(
        "--pixal3d-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--image", type=Path, required=True, help="RGBA cut-out")
    parser.add_argument("--output", type=Path, required=True, help="fresh .glb path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--fov", type=float)
    parser.add_argument(
        "--transform-profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="class-level Pixal3D transform profile; never keyed by asset id",
    )
    return parser


def _resolve_model_root(
    override: Path | None, *, name: str, label: str, directory: bool
) -> Path:
    try:
        value = resolve_model_root(name, override=override)
    except SystemExit as error:
        raise ValueError(str(error)) from error
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} {kind} is missing: {path}")
    return path


def _resolve_model_roots(args) -> dict[str, Path]:
    return {
        "pixal3d": _resolve_model_root(
            args.model_root,
            name=MODEL_ROOT_NAMES["pixal3d"],
            label="Pixal3D model root",
            directory=True,
        ),
        "moge": _resolve_model_root(
            args.moge_root,
            name=MODEL_ROOT_NAMES["moge"],
            label="MoGe model root",
            directory=True,
        ),
        "dinov3": _resolve_model_root(
            args.dinov3_root,
            name=MODEL_ROOT_NAMES["dinov3"],
            label="DINOv3 model root",
            directory=True,
        ),
        "naf": _resolve_model_root(
            args.naf_root,
            name=MODEL_ROOT_NAMES["naf"],
            label="NAF checkpoint",
            directory=False,
        ),
    }


def _validate_rgba_cutout(image: Path) -> None:
    from PIL import Image

    with Image.open(image) as opened:
        opened.load()
        if opened.mode != "RGBA":
            raise ValueError("the input must carry an RGBA alpha channel")
        low, high = opened.getchannel("A").getextrema()
        if low >= 255 or high <= 0:
            raise ValueError(
                "the input alpha channel must contain a non-opaque foreground cut-out"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pixal3d_root is not None:
        print(
            "mesh refused: --pixal3d-root is deprecated; Pixal3D source is "
            "AVEngine-local, use --model-root for weights",
            file=sys.stderr,
        )
        return 2
    if args.output.exists():
        print(
            f"mesh refused: {args.output} exists; choose a fresh path",
            file=sys.stderr,
        )
        return 2
    if not args.image.is_file():
        print(f"mesh refused: cut-out not found: {args.image}", file=sys.stderr)
        return 2

    try:
        _validate_rgba_cutout(args.image)
        roots = _resolve_model_roots(args)
        from pixal3d.inference import run_inference

        profile = load_profile(args.transform_profile)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run_inference(
            image_path=str(args.image.resolve()),
            output_path=str(args.output.resolve()),
            seed=args.seed,
            manual_fov=-1.0 if args.fov is None else args.fov,
            model_path=roots["pixal3d"],
            moge_model_path=roots["moge"],
            dinov3_model_path=roots["dinov3"],
            naf_model_path=roots["naf"],
            resolution=-1 if args.resolution is None else args.resolution,
            export_transform=mesh_export_matrix(profile),
        )
    except (OSError, ImportError, ValueError, RuntimeError) as error:
        print(f"mesh refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
