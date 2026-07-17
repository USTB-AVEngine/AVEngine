#!/usr/bin/env python3
"""Build one diagnostic-only cross-species appearance lineage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import BinaryIO
from typing import Sequence

from avengine.m2.cross_species_lineage import (
    CrossSpeciesLineageError,
    build_cross_species_appearance_lineage,
    validate_cross_species_appearance_lineage,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--upstream-source-manifest", type=Path, required=True)
    parser.add_argument("--material-realization-report", type=Path, required=True)
    parser.add_argument("--material-normalization-report", type=Path, required=True)
    parser.add_argument("--rebase-report", type=Path, required=True)
    parser.add_argument("--pre-rebase-visual-glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


@dataclass
class _OutputReservation:
    path: Path
    stream: BinaryIO
    parent_fd: int
    device: int
    inode: int

    def entry_is_owned(self) -> bool:
        try:
            current = os.stat(
                self.path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        return (
            stat.S_ISREG(current.st_mode)
            and current.st_dev == self.device
            and current.st_ino == self.inode
        )

    def cleanup_owned_entry(self) -> None:
        if self.entry_is_owned():
            os.unlink(self.path.name, dir_fd=self.parent_fd)

    def close(self) -> None:
        self.stream.close()
        os.close(self.parent_fd)


def _open_directory_chain(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            try:
                following = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o777, dir_fd=current)
                except FileExistsError:
                    pass
                following = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = following
    except OSError as exc:
        os.close(current)
        raise CrossSpeciesLineageError(
            f"output parent must be a real directory chain without symlinks: {exc}"
        ) from exc
    return current


def _unlink_if_owned(
    *,
    parent_fd: int,
    name: str,
    device: int,
    inode: int,
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and current.st_dev == device
        and current.st_ino == inode
    ):
        os.unlink(name, dir_fd=parent_fd)


def _reserve_output(path: Path) -> _OutputReservation:
    parent_fd = _open_directory_chain(path.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o666,
            dir_fd=parent_fd,
        )
        identity = os.fstat(descriptor)
        stream = os.fdopen(descriptor, "w+b")
        descriptor = None
        return _OutputReservation(
            path=path,
            stream=stream,
            parent_fd=parent_fd,
            device=identity.st_dev,
            inode=identity.st_ino,
        )
    except OSError as exc:
        if descriptor is not None:
            identity = os.fstat(descriptor)
            os.close(descriptor)
            _unlink_if_owned(
                parent_fd=parent_fd,
                name=path.name,
                device=identity.st_dev,
                inode=identity.st_ino,
            )
        os.close(parent_fd)
        raise CrossSpeciesLineageError(
            f"unable to reserve output without replacement: {exc}"
        ) from exc


def _write_reserved_json(
    reservation: _OutputReservation,
    value: object,
) -> tuple[dict[str, object], str]:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    reservation.stream.write(payload)
    reservation.stream.flush()
    os.fsync(reservation.stream.fileno())
    reservation.stream.seek(0)
    readback_payload = reservation.stream.read()
    if readback_payload != payload:
        raise CrossSpeciesLineageError("reserved output readback bytes differ")
    readback = json.loads(readback_payload.decode("utf-8"))
    if not isinstance(readback, dict):
        raise CrossSpeciesLineageError("reserved output readback is not an object")
    return readback, hashlib.sha256(readback_payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output = Path(os.path.abspath(args.output))
    if output.suffix.lower() != ".json":
        parser.error("--output must use a .json suffix")
    reservation: _OutputReservation | None = None
    try:
        reservation = _reserve_output(output)
        lineage = build_cross_species_appearance_lineage(
            variant_spec=args.spec,
            upstream_source_manifest=args.upstream_source_manifest,
            material_realization_report=args.material_realization_report,
            material_normalization_report=args.material_normalization_report,
            rebase_report=args.rebase_report,
            pre_rebase_visual_glb=args.pre_rebase_visual_glb,
            lineage_producer=Path(__file__),
            lineage_contract=(
                REPOSITORY_ROOT / "src/avengine/m2/cross_species_lineage.py"
            ),
            material_realization_tool=(
                REPOSITORY_ROOT / "tools/m2/force_matte_materials.py"
            ),
            material_normalization_tool=(
                REPOSITORY_ROOT / "tools/m2/normalize_materials.py"
            ),
            material_algorithm=(REPOSITORY_ROOT / "src/avengine/m2/materials.py"),
            skin_root_rebase_tool=(REPOSITORY_ROOT / "tools/m2/rebase_skin_root.py"),
            rebase_algorithm=(REPOSITORY_ROOT / "src/avengine/m2/rebase.py"),
        )
        readback, output_sha256 = _write_reserved_json(reservation, lineage)
        validate_cross_species_appearance_lineage(
            readback,
            expected_spec_path=args.spec,
            expected_upstream_source_manifest=args.upstream_source_manifest,
            expected_material_normalization_report=(args.material_normalization_report),
            expected_rebase_report=args.rebase_report,
            expected_final_visual=lineage["artifacts"]["final_visual_glb"]["path"],
            expected_repository_root=REPOSITORY_ROOT,
        )
        if not reservation.entry_is_owned():
            raise CrossSpeciesLineageError(
                "reserved output path no longer names the producer-owned inode"
            )
    except (CrossSpeciesLineageError, OSError, ValueError) as exc:
        if reservation is not None:
            reservation.cleanup_owned_entry()
            reservation.close()
        parser.error(str(exc))
    assert reservation is not None
    reservation.close()
    print(
        json.dumps(
            {
                "status": "pass",
                "schema": lineage["schema"],
                "design_kind": lineage["design_kind"],
                "ofat_status": lineage["ofat_status"],
                "qualification_claim": lineage["qualification_claim"],
                "formal_dataset_registration_authorized": lineage[
                    "formal_dataset_registration_authorized"
                ],
                "output": str(output),
                "output_sha256": output_sha256,
                "lineage_content_sha256": lineage["lineage_content_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
