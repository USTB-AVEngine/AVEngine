#!/usr/bin/env python3
"""Render a resumable native-RLR RIR cache from an M6.x job plan."""

from __future__ import annotations

import argparse
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
)
from avengine.m3.runtime import load_compiled_acoustic_scene
from avengine.spatial_audio.runtime import M4SimulationConfig
from avengine.m6x.rir_cache import (
    RIR_CACHE_ACOUSTIC_SELECTION_NAME,
    load_semantic_acoustic_scene,
    render_rir_cache,
    render_semantic_rir_cache,
)


REPOSITORY = Path(__file__).resolve().parents[2]
LEGACY_SIMULATION_REQUEST = (
    REPOSITORY / "examples/m4/blender_custom/multi_source_canary_request.json"
)
DEFAULT_ROOM_REGISTRY = REPOSITORY / "examples/m6/rooms/room_registry.json"
ACOUSTIC_SELECTION_NAME = RIR_CACHE_ACOUSTIC_SELECTION_NAME


@dataclass(frozen=True)
class EffectiveAcousticInputs:
    """Exact package/request inputs selected for one cache invocation."""

    acoustic_package_manifest: Path
    simulation_request: Path
    selection_mode: str
    simulation_profile: str
    profile_selection_receipt: Mapping[str, Any] | None
    explicit_package_override: bool
    explicit_simulation_override: bool

    def receipt(self) -> dict[str, Any]:
        def record(path: Path) -> dict[str, Any]:
            return {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        value = {
            "schema": "avengine_rir_cache_acoustic_selection_v1",
            "selection_mode": self.selection_mode,
            "simulation_profile": self.simulation_profile,
            "explicit_overrides": {
                "acoustic_package_manifest": self.explicit_package_override,
                "simulation_request": self.explicit_simulation_override,
            },
            "registry_selection_applied_to_effective_inputs": {
                "acoustic_package_manifest": (
                    self.profile_selection_receipt is not None
                    and not self.explicit_package_override
                ),
                "simulation_request": (
                    self.profile_selection_receipt is not None
                    and not self.explicit_simulation_override
                ),
            },
            "registry_resolution": (
                dict(self.profile_selection_receipt)
                if self.profile_selection_receipt is not None
                else None
            ),
            "effective_inputs": {
                "acoustic_package_manifest": record(
                    self.acoustic_package_manifest
                ),
                "simulation_request": record(self.simulation_request),
            },
        }
        value["effective_selection_content_sha256"] = canonical_json_sha256(
            value
        )
        return value


def resolve_effective_acoustic_inputs(
    args: argparse.Namespace,
) -> EffectiveAcousticInputs:
    """Resolve manual paths or one exact registry-selected room profile."""

    explicit_package = args.acoustic_package_manifest is not None
    explicit_simulation = args.simulation_request is not None
    if args.room_id is None:
        if args.room_revision is not None:
            raise ValueError("--room-revision requires --room-id")
        if args.acoustic_profile_registry is not None:
            raise ValueError("--acoustic-profile-registry requires --room-id")
        if args.simulation_profile != "production":
            raise ValueError(
                "--simulation-profile reference requires registry selection "
                "through --room-id"
            )
        if not explicit_package:
            raise ValueError(
                "provide --acoustic-package-manifest, or select a room with "
                "--room-id and --room-revision"
            )
        return EffectiveAcousticInputs(
            acoustic_package_manifest=args.acoustic_package_manifest.resolve(),
            simulation_request=(
                args.simulation_request.resolve()
                if explicit_simulation
                else LEGACY_SIMULATION_REQUEST.resolve()
            ),
            selection_mode="explicit",
            simulation_profile=(
                "explicit" if explicit_simulation else "legacy_default"
            ),
            profile_selection_receipt=None,
            explicit_package_override=True,
            explicit_simulation_override=explicit_simulation,
        )

    if args.room_revision is None:
        raise ValueError("--room-id requires --room-revision")

    from avengine.acoustic_profiles import (  # noqa: PLC0415
        load_acoustic_profile_registry,
        load_default_acoustic_profile_registry,
        resolve_acoustic_profile,
    )
    from avengine.m6.rooms import load_room_registry  # noqa: PLC0415

    acoustic_registry = (
        load_acoustic_profile_registry(args.acoustic_profile_registry)
        if args.acoustic_profile_registry is not None
        else load_default_acoustic_profile_registry()
    )
    room_registry = load_room_registry(args.room_registry)
    room_ref = {
        "registry_id": room_registry["registry_id"],
        "room_id": args.room_id,
        "revision": args.room_revision,
    }
    selection = resolve_acoustic_profile(
        acoustic_registry,
        room_registry,
        room_ref,
        repository_root=REPOSITORY,
        verify_paths=True,
    )
    selected_request = Path(
        selection.simulation_path(args.simulation_profile)
    ).resolve()
    selected_package = Path(selection.acoustic_package_manifest_path).resolve()

    def verified_equivalent_override(
        override: Path | None,
        selected: Path,
        *,
        option: str,
    ) -> Path:
        if override is None:
            return selected
        resolved = override.resolve()
        if not resolved.is_file():
            raise ValueError(f"{option} override does not exist: {resolved}")
        if sha256_file(resolved) != sha256_file(selected):
            raise ValueError(
                f"{option} override SHA-256 differs from the registry-selected "
                "physical file"
            )
        return resolved

    effective_package = verified_equivalent_override(
        args.acoustic_package_manifest,
        selected_package,
        option="--acoustic-package-manifest",
    )
    effective_simulation = verified_equivalent_override(
        args.simulation_request,
        selected_request,
        option="--simulation-request",
    )
    return EffectiveAcousticInputs(
        acoustic_package_manifest=effective_package,
        simulation_request=effective_simulation,
        selection_mode=(
            "registry_with_verified_equivalent_overrides"
            if explicit_package or explicit_simulation
            else "registry"
        ),
        simulation_profile=args.simulation_profile,
        profile_selection_receipt=selection.receipt(args.simulation_profile),
        explicit_package_override=explicit_package,
        explicit_simulation_override=explicit_simulation,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir-job-plan", type=Path, required=True)
    parser.add_argument(
        "--semantic-no-file-evidence",
        action="store_true",
        help=(
            "Render one fresh structural/sample cache from explicit inputs "
            "without entering legacy file-digest or byte-size evidence paths"
        ),
    )
    parser.add_argument(
        "--acoustic-package-manifest",
        type=Path,
        help=(
            "Explicit package manifest. Required unless --room-id and "
            "--room-revision select it from the acoustic profile registry. "
            "With registry selection, an explicit path is accepted only when "
            "its SHA-256 equals the selected package."
        ),
    )
    parser.add_argument(
        "--simulation-request",
        type=Path,
        help=(
            "Explicit simulation request. Manual package mode keeps the "
            "historical M4 request default; room selection uses its production "
            "or reference request. With registry selection, an explicit path "
            "is accepted only when its SHA-256 equals the selected request."
        ),
    )
    parser.add_argument(
        "--room-id",
        help="Room identity for automatic acoustic profile selection",
    )
    parser.add_argument(
        "--room-revision",
        help="Exact room revision for automatic acoustic profile selection",
    )
    parser.add_argument(
        "--room-registry",
        type=Path,
        default=DEFAULT_ROOM_REGISTRY,
    )
    parser.add_argument(
        "--acoustic-profile-registry",
        type=Path,
        help="Override the installed/default acoustic profile registry",
    )
    parser.add_argument(
        "--simulation-profile",
        choices=("production", "reference"),
        default="production",
    )
    parser.add_argument(
        "--hrtf",
        type=Path,
        default=Path("/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"),
    )
    parser.add_argument(
        "--runtime-prefix",
        type=Path,
        help="Explicit non-Git installed Habitat runtime prefix",
    )
    parser.add_argument(
        "--magnum-python-site",
        type=Path,
        help="Explicit non-Git Corrade/Magnum Python site",
    )
    parser.add_argument(
        "--rlr-sdk-root",
        type=Path,
        help="Explicit non-Git RLRAudioPropagationPkg root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=("binaural", "ambisonics"), default="binaural"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thread-count", type=int)
    parser.add_argument("--job-offset", type=int, default=0)
    parser.add_argument("--job-limit", type=int)
    parser.add_argument(
        "--coordinate-translation-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--source-radius-m", type=float, default=0.0)
    parser.add_argument("--listener-radius-m", type=float, default=0.0)
    parser.add_argument("--uncompressed", action="store_true")
    return parser.parse_args(argv)


def _semantic_cli_regular_file(path: Path | None, *, owner: str) -> Path:
    if path is None:
        raise ValueError(f"{owner} is required in semantic mode")
    raw = Path(path)
    if ".." in raw.parts:
        raise ValueError(f"{owner} may not contain parent traversal")
    absolute = Path(os.path.abspath(raw))
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{owner} may not contain symlink components")
    if not absolute.is_file() or absolute.resolve(strict=True) != absolute:
        raise ValueError(f"{owner} must be a selected regular file")
    return absolute


def _run_semantic(args: argparse.Namespace) -> Path:
    if (
        args.room_id is not None
        or args.room_revision is not None
        or args.acoustic_profile_registry is not None
        or args.simulation_profile != "production"
        or args.room_registry != DEFAULT_ROOM_REGISTRY
    ):
        raise ValueError(
            "semantic no-file-evidence mode requires explicit package and "
            "simulation inputs; registry selection remains a legacy mode"
        )
    if args.job_offset != 0 or args.job_limit is not None:
        raise ValueError(
            "semantic no-file-evidence mode renders the complete selected plan"
        )
    if args.layout != "binaural":
        raise ValueError("semantic no-file-evidence mode supports binaural only")
    plan_path = _semantic_cli_regular_file(args.rir_job_plan, owner="--rir-job-plan")
    package_path = _semantic_cli_regular_file(
        args.acoustic_package_manifest, owner="--acoustic-package-manifest"
    )
    simulation_path = _semantic_cli_regular_file(
        args.simulation_request, owner="--simulation-request"
    )
    hrtf_path = _semantic_cli_regular_file(args.hrtf, owner="--hrtf")
    simulation_document = load_json(simulation_path)
    if not isinstance(simulation_document, ABCMapping) or not isinstance(
        simulation_document.get("simulation"), ABCMapping
    ):
        raise ValueError("semantic simulation request is invalid")
    simulation_value = dict(simulation_document["simulation"])
    if args.thread_count is not None:
        simulation_value["thread_count"] = args.thread_count
    simulation = M4SimulationConfig.from_mapping(simulation_value)
    scene = load_semantic_acoustic_scene(package_path)
    result = render_semantic_rir_cache(
        plan_path=plan_path,
        scene=scene,
        simulation_request_path=simulation_path,
        simulation=simulation,
        output=args.output,
        acoustic_selection={
            "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
            "selection_mode": "explicit_legacy_unbound",
            "registry_selection_applied": False,
            "room_ref": None,
            "profile_ref": None,
            "binding_id": None,
        },
        layout_type="binaural",
        hrtf_file_path=hrtf_path,
        batch_size=args.batch_size,
        coordinate_translation_m=args.coordinate_translation_m,
        source_radius_m=args.source_radius_m,
        listener_radius_m=args.listener_radius_m,
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_python_site,
        rlr_sdk_root=args.rlr_sdk_root,
        compressed=not args.uncompressed,
    )
    print(
        "SEMANTIC_RIR_CACHE_OK "
        f"output={result.output} "
        f"jobs={result.receipt['selected_job_count']} "
        f"full_plan_complete={result.receipt['full_plan_complete']}",
        flush=True,
    )
    return result.output


def run(args: argparse.Namespace) -> Path:
    if args.semantic_no_file_evidence:
        return _run_semantic(args)
    inputs = resolve_effective_acoustic_inputs(args)
    selection_receipt = inputs.receipt()
    simulation_request = inputs.simulation_request
    simulation_value = load_json(simulation_request)["simulation"]
    if args.thread_count is not None:
        simulation_value = dict(simulation_value)
        simulation_value["thread_count"] = args.thread_count
    simulation = M4SimulationConfig.from_mapping(simulation_value)
    scene = load_compiled_acoustic_scene(
        inputs.acoustic_package_manifest,
        allow_nonpassing_research_qa=True,
    )
    result = render_rir_cache(
        plan_path=args.rir_job_plan,
        scene=scene,
        simulation_request_path=simulation_request,
        simulation=simulation,
        output=args.output,
        acoustic_selection_receipt=selection_receipt,
        layout_type=args.layout,
        hrtf_file_path=args.hrtf if args.layout == "binaural" else None,
        batch_size=args.batch_size,
        job_offset=args.job_offset,
        job_limit=args.job_limit,
        coordinate_translation_m=args.coordinate_translation_m,
        source_radius_m=args.source_radius_m,
        listener_radius_m=args.listener_radius_m,
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_python_site,
        rlr_sdk_root=args.rlr_sdk_root,
        compressed=not args.uncompressed,
    )
    print(
        "RIR_CACHE_OK "
        f"output={result.output} "
        f"jobs={result.receipt['selected_job_count']} "
        f"full_plan_complete={result.receipt['full_plan_complete']} "
        f"acoustic_selection={inputs.selection_mode}",
        flush=True,
    )
    return result.output


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
