#!/usr/bin/env python3
"""Export UE's approaching_storm HDRI and build a visual-only Habitat GLB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


REPOSITORY = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ue-root",
        type=Path,
        help="UE installation root; required unless --retained-hdri is used.",
    )
    parser.add_argument(
        "--uproject",
        type=Path,
        help="SpearSim .uproject; required unless --retained-hdri is used.",
    )
    parser.add_argument("--blender", default="blender")
    parser.add_argument(
        "--visual-profile",
        type=Path,
        default=(
            REPOSITORY
            / "examples/m6x/fixed_apartment/review_visual_profile.json"
        ),
    )
    parser.add_argument(
        "--retained-hdri",
        type=Path,
        help=(
            "Reuse a user-supplied HDRI instead of launching UE again; this "
            "mode does not independently prove the file came from UE."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "tmp/m6x/assets/approaching_storm_4k_exterior_v2",
    )
    parser.add_argument("--texture-width", type=int, default=2048)
    return parser.parse_args(argv)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"asset preparation command returned {completed.returncode}: {command[0]}"
        )


def _executable(value: str) -> Path:
    candidate = shutil.which(value)
    if candidate is not None:
        return Path(candidate).resolve()
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    raise RuntimeError(f"Blender executable is missing: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    blender = _executable(args.blender)
    visual_profile = args.visual_profile.resolve()
    retained_hdri = (
        args.retained_hdri.resolve() if args.retained_hdri is not None else None
    )
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.staging")
    for owner, path in (
        ("Blender", blender),
        ("review visual profile", visual_profile),
    ):
        if not path.is_file():
            raise RuntimeError(f"{owner} is missing: {path}")
    editor: Path | None = None
    uproject: Path | None = None
    engine_asset: Path | None = None
    if retained_hdri is None:
        if args.ue_root is None or args.uproject is None:
            raise RuntimeError(
                "--ue-root and --uproject are required for a fresh UE export"
            )
        ue_root = args.ue_root.resolve()
        uproject = args.uproject.resolve()
        editor = ue_root / "Engine/Binaries/Linux/UnrealEditor-Cmd"
        engine_asset = (
            ue_root
            / "Engine/Plugins/Runtime/HDRIBackdrop/Content/Textures/approaching_storm_4k.uasset"
        )
        for owner, path in (
            ("UnrealEditor-Cmd", editor),
            ("SpearSim project", uproject),
            ("approaching_storm UAsset", engine_asset),
        ):
            if not path.is_file():
                raise RuntimeError(f"{owner} is missing: {path}")
    if os.path.lexists(output) or os.path.lexists(staging):
        raise RuntimeError(f"refusing to replace output or staging path: {output}")
    staging.mkdir(parents=True)
    hdri = staging / "approaching_storm_4k.hdr"
    proxy = staging / "approaching_storm_4k_exterior.glb"
    try:
        env = os.environ.copy()
        env["AVENGINE_HDRI_EXPORT_PATH"] = str(hdri)
        if retained_hdri is None:
            assert editor is not None and uproject is not None
            _run(
                [
                    str(editor),
                    str(uproject),
                    f"-ExecutePythonScript={REPOSITORY / 'tools/m6x/unreal_export_approaching_storm.py'}",
                    "-unattended",
                    "-nop4",
                    "-nosplash",
                    "-NoSound",
                    "-NullRHI",
                ],
                env=env,
            )
            extraction_mode = "fresh_unreal_asset_export"
            assert engine_asset is not None
            source_record = {
                "asset_uri": "unreal:///HDRIBackdrop/Textures/approaching_storm_4k",
                "uasset_sha256": _sha256(engine_asset),
                "extraction_mode": extraction_mode,
            }
        else:
            if not retained_hdri.is_file():
                raise RuntimeError(f"retained HDRI is missing: {retained_hdri}")
            shutil.copy2(retained_hdri, hdri)
            extraction_mode = "retained_user_supplied_hdri"
            source_record = {
                "asset_uri": "user-supplied retained HDRI",
                "extraction_mode": extraction_mode,
            }
        _run(
            [
                str(blender),
                "--background",
                "--factory-startup",
                "--python",
                str(REPOSITORY / "tools/m6x/blender_build_exterior_proxy.py"),
                "--",
                "--input-hdri",
                str(hdri),
                "--output-glb",
                str(proxy),
                "--visual-profile",
                str(visual_profile),
                "--texture-width",
                str(args.texture_width),
            ]
        )
        manifest = {
            "schema": "avengine_m6x_prepared_exterior_proxy_v1",
            "status": "pass",
            "source": source_record,
            "visual_profile_sha256": _sha256(visual_profile),
            "outputs": {
                "hdri": {"path": hdri.name, "sha256": _sha256(hdri)},
                "proxy_glb": {"path": proxy.name, "sha256": _sha256(proxy)},
            },
            "scope": (
                "Transient visual-capture proxy; excluded from the room "
                "SceneInstance, collision, navmesh, Topdown obstacles, and RLR "
                "acoustics. It renders as semantic background ID 0 and a distant "
                "depth surface."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, output)
    except BaseException:
        # Keep the staging directory and native logs/files for diagnosis.  A
        # failed extraction must never be represented as a synthetic sky.
        raise
    print(output / proxy.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
