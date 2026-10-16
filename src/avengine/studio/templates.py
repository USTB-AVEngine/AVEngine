"""Argv builders for the render task templates the Studio may queue.

Defaults come from the deployment config; a submission may override only
the authoring-facing keys of its template. Every input path is validated
at submit time so a bad request fails as HTTP 400 instead of a queued
task failure. Every chain is the engine's own CLI — the Studio assembles
argv, never renders anything itself.
"""

from __future__ import annotations

from pathlib import Path

from avengine.studio.config import StudioConfig

MP3D_DYNAMIC_AUDIO_TEMPLATE = "mp3d_dynamic_audio"
MP3D_END_TO_END_TEMPLATE = "mp3d_end_to_end"
MP3D_ROUTE_AUTHOR_TEMPLATE = "mp3d_route_author"
APARTMENT_AUTHOR_TEMPLATE = "apartment_author"
APARTMENT_END_TO_END_TEMPLATE = "apartment_end_to_end"
PAIRED_ABLATION_TEMPLATE = "paired_ablation"
HM3D_ROOM_PREPARE_TEMPLATE = "hm3d_room_prepare"
SEMANTIC_PACKAGE_TEMPLATE = "semantic_acoustic_package"
HM3D_ROUTE_BANK_TEMPLATE = "hm3d_route_bank"
HM3D_EPISODE_TEMPLATE = "hm3d_episode"
HM3D_END_TO_END_TEMPLATE = "hm3d_end_to_end"
KUJIALE_ROUTE_BANK_TEMPLATE = "kujiale_route_bank"
KUJIALE_VISUAL_EPISODE_TEMPLATE = "kujiale_visual_episode"
KUJIALE_ACOUSTIC_PACKAGE_TEMPLATE = "kujiale_acoustic_package"

# The Habitat runtime activation triplet every headless HM3D tool needs.
_HM3D_RUNTIME_KEYS = ("runtime_prefix", "magnum_site", "rlr_sdk_root")
_HM3D_EPISODE_PATH_KEYS = (
    *_HM3D_RUNTIME_KEYS,
    "audio_python",
    "bank",
    "asset_dir",
    "dataset_config",
    "materials_json",
    "hrtf",
)
_HM3D_SPLITS = frozenset({"train", "val", "minival", "test"})

# The one-click chain takes the union of the four stages' inputs; the only
# authoring-facing choice left is which house (the chain picks the room).
_HM3D_E2E_PATH_KEYS = (
    "scene_dir",
    "hm3d_root",
    *_HM3D_RUNTIME_KEYS,
    "material_rules",
    "audio_python",
    "asset_dir",
    "dataset_config",
    "materials_json",
    "hrtf",
)

_MP3D_AUDIO_PATH_KEYS = (
    "visual_capture_dir",
    "m1_request",
    "simulation_request",
    "package_manifest",
    "audio_program",
    "source_endpoint_registry",
    "sound_asset_registry",
    "beagle_audio",
    "hrtf",
    "runtime_prefix",
    "rlr_sdk_root",
)
_MP3D_AUDIO_OPTIONAL_PATH_KEYS = ("hrtf_license", "magnum_python_site")

_MP3D_E2E_PATH_KEYS = (
    "source_animal_manifest",
    "source_m2_request",
    "room_manifest",
    "simulation_request",
    "package_manifest",
    "audio_program",
    "source_endpoint_registry",
    "sound_asset_registry",
    "beagle_audio",
    "hrtf",
    "runtime_prefix",
    "mp3d_root",
    "magnum_python_site",
    "rlr_sdk_root",
)

_APARTMENT_AUTHOR_POINT_KEYS = (
    "camera_position_ue_cm",
    "human_start_ue_cm",
    "human_end_ue_cm",
    "beagle_start_ue_cm",
    "beagle_end_ue_cm",
)

_APARTMENT_E2E_PATH_KEYS = (
    "actor_selection",
    "source_asset_registry",
    "closure_report",
    "stage_root",
    "spear_executable",
    "m1_request",
    "simulation_request",
    "package_manifest",
    "audio_program",
    "source_endpoint_registry",
    "sound_asset_registry",
    "beagle_audio",
    "hrtf",
    "runtime_prefix",
    "rlr_sdk_root",
    "magnum_python_site",
)

TEMPLATE_OVERRIDABLE_KEYS: dict[str, frozenset[str]] = {
    # The MP3D templates accept the room-identity paths as overrides so a
    # submission can target any MP3D room on disk, not only the configured
    # default. Identity travels together: a room manifest from one room with
    # the acoustic package of another is exactly the mismatch the compiled
    # package's own hash checks exist to reject downstream.
    MP3D_DYNAMIC_AUDIO_TEMPLATE: frozenset(
        {
            "visual_capture_dir",
            "audio_program",
            "variant",
            "rir_stride_frames",
            "m1_request",
            "package_manifest",
        }
    ),
    MP3D_END_TO_END_TEMPLATE: frozenset(
        {
            "seed",
            "camera_selection",
            "audio_program",
            "variant",
            "rir_stride_frames",
            "room_manifest",
            "package_manifest",
            "mp3d_root",
            "source_m2_request",
            "source_animal_manifest",
        }
    ),
    MP3D_ROUTE_AUTHOR_TEMPLATE: frozenset(
        {
            "seed",
            "camera_selection",
            "room_manifest",
            "package_manifest",
            "mp3d_root",
            "source_m2_request",
            "source_animal_manifest",
        }
    ),
    HM3D_ROOM_PREPARE_TEMPLATE: frozenset(
        {"scene_dir", "split", "connectivity_samples", "connectivity_seed"}
    ),
    SEMANTIC_PACKAGE_TEMPLATE: frozenset(
        {"room_manifest", "seed", "package_id", "verify_frame_parity"}
    ),
    HM3D_ROUTE_BANK_TEMPLATE: frozenset(
        {
            "scene",
            "navmesh",
            "room_bounds",
            "room_label",
            "seed",
            "episodes_per_motion_case",
            "minimum_route_distance_m",
            "maximum_route_distance_m",
            "source1_height_m",
            "source2_height_m",
        }
    ),
    HM3D_EPISODE_TEMPLATE: frozenset(
        {
            "bank",
            "scene_id",
            "asset_dir",
            "episode_index",
            "episode_id",
            "motion_case",
            "slot",
            "seed",
            "frame_stride",
            "aim_open",
            "overhead_m",
            "place_at_emitter",
            "width",
            "height",
        }
    ),
    HM3D_END_TO_END_TEMPLATE: frozenset({"scene_dir", "split", "seed"}),
    KUJIALE_ROUTE_BANK_TEMPLATE: frozenset(
        {
            "scene_metadata",
            "seed",
            "episodes_per_motion_case",
            "minimum_route_distance_m",
            "maximum_route_distance_m",
            "minimum_clearance_m",
        }
    ),
    KUJIALE_VISUAL_EPISODE_TEMPLATE: frozenset(
        {"episode_root", "rpc_port", "graphics_adapter", "visual_only_research"}
    ),
    KUJIALE_ACOUSTIC_PACKAGE_TEMPLATE: frozenset(
        {
            "source_usd",
            "room_id",
            "transform_profile",
            "interior_origins",
            "source_revision",
            "seed",
        }
    ),
    APARTMENT_AUTHOR_TEMPLATE: frozenset(
        {*_APARTMENT_AUTHOR_POINT_KEYS, "camera_yaw_deg"}
    ),
    APARTMENT_END_TO_END_TEMPLATE: frozenset(
        {
            *_APARTMENT_AUTHOR_POINT_KEYS,
            "camera_yaw_deg",
            "audio_program",
            "variant",
            "rir_stride_frames",
        }
    ),
    PAIRED_ABLATION_TEMPLATE: frozenset({"audio_dir", "pair_id", "mute_stems"}),
}


class StudioTemplateError(ValueError):
    """Raised when a task submission cannot be turned into a valid argv."""


def _input_path(key: str, value: object, repository_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    path = path.resolve()
    if not path.exists():
        raise StudioTemplateError(f"template input {key!r} does not exist: {path}")
    return path


def _merged(
    config: StudioConfig, template_name: str, overrides: dict[str, object]
) -> dict[str, object]:
    defaults = config.task_templates.get(template_name)
    if defaults is None:
        raise StudioTemplateError(
            f"unknown template {template_name!r}; configured templates: "
            f"{sorted(config.task_templates)}"
        )
    overridable = TEMPLATE_OVERRIDABLE_KEYS.get(template_name)
    if overridable is None:
        raise StudioTemplateError(f"no argv builder for template {template_name!r}")
    unknown = sorted(set(overrides) - overridable)
    if unknown:
        raise StudioTemplateError(
            f"overrides not allowed for {template_name!r}: {unknown}; "
            f"overridable keys: {sorted(overridable)}"
        )
    merged: dict[str, object] = dict(defaults)
    merged.update(overrides)
    return merged


def _required(merged: dict[str, object], template_name: str, key: str) -> object:
    value = merged.get(key)
    if value is None:
        raise StudioTemplateError(
            f"template {template_name!r} is missing required input {key!r}"
        )
    return value


def _stride(merged: dict[str, object]) -> str:
    stride = int(merged.get("rir_stride_frames", 3))
    if stride < 1:
        raise StudioTemplateError(f"rir_stride_frames must be >= 1, got {stride}")
    return str(stride)


def _fresh_output(output_path: Path) -> str:
    resolved = Path(output_path).resolve()
    if resolved.exists():
        raise StudioTemplateError(
            f"output path already exists (fresh/no-clobber): {resolved}"
        )
    return str(resolved)


def _point_cm(merged: dict[str, object], template_name: str, key: str) -> list[str]:
    value = _required(merged, template_name, key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise StudioTemplateError(f"{key} must be a [x, y, z] triple in UE cm")
    return [str(float(item)) for item in value]


def _append_paths(
    argv: list[str],
    merged: dict[str, object],
    template_name: str,
    keys: tuple[str, ...],
    repository_root: Path,
    *,
    optional: tuple[str, ...] = (),
) -> None:
    for key in keys:
        value = _required(merged, template_name, key)
        path = _input_path(key, value, repository_root)
        argv += [f"--{key.replace('_', '-')}", str(path)]
    for key in optional:
        value = merged.get(key)
        if value is not None:
            path = _input_path(key, value, repository_root)
            argv += [f"--{key.replace('_', '-')}", str(path)]


def build_template_argv(
    config: StudioConfig,
    template_name: str,
    overrides: dict[str, object],
    output_path: Path,
) -> list[str]:
    merged = _merged(config, template_name, overrides)
    python = str(config.python_executable)
    repo = config.repository_root

    if template_name == MP3D_DYNAMIC_AUDIO_TEMPLATE:
        argv = [python, "-m", "avengine.cli", "m5", "render-current-mp3d-dynamic-audio"]
        _append_paths(
            argv,
            merged,
            template_name,
            _MP3D_AUDIO_PATH_KEYS,
            repo,
            optional=_MP3D_AUDIO_OPTIONAL_PATH_KEYS,
        )
        argv += ["--rir-stride-frames", _stride(merged)]
        argv += ["--variant", str(merged.get("variant", "A"))]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name in (MP3D_END_TO_END_TEMPLATE, MP3D_ROUTE_AUTHOR_TEMPLATE):
        seed = int(_required(merged, template_name, "seed"))
        argv = [python, str(repo / "tools/studio/run_mp3d_end_to_end.py")]
        argv += ["--seed", str(seed)]
        argv += [
            "--camera-selection",
            str(merged.get("camera_selection", "lateral_sweep")),
        ]
        _append_paths(
            argv, merged, template_name, _MP3D_E2E_PATH_KEYS, repo,
            optional=("hrtf_license",),
        )
        argv += ["--rir-stride-frames", _stride(merged)]
        argv += ["--variant", str(merged.get("variant", "A"))]
        if template_name == MP3D_ROUTE_AUTHOR_TEMPLATE:
            argv += ["--author-only"]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == APARTMENT_AUTHOR_TEMPLATE:
        argv = [
            python,
            "-m",
            "avengine.cli",
            "m5",
            "author-current-apartment-visual-timeline",
        ]
        _append_paths(
            argv,
            merged,
            template_name,
            ("actor_selection", "source_asset_registry"),
            repo,
        )
        for key in _APARTMENT_AUTHOR_POINT_KEYS:
            argv += [f"--{key.replace('_', '-')}", *_point_cm(merged, template_name, key)]
        argv += ["--camera-yaw-deg", str(float(_required(merged, template_name, "camera_yaw_deg")))]
        for key, fallback in (("width", 1280), ("height", 720), ("hfov_degrees", 105.0)):
            value = merged.get(key, fallback)
            argv += [f"--{key.replace('_', '-')}", str(value)]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == APARTMENT_END_TO_END_TEMPLATE:
        argv = [python, str(repo / "tools/studio/run_apartment_end_to_end.py")]
        _append_paths(
            argv,
            merged,
            template_name,
            _APARTMENT_E2E_PATH_KEYS,
            repo,
            optional=("hrtf_license", "spear_ext_dir"),
        )
        for key in _APARTMENT_AUTHOR_POINT_KEYS:
            argv += [f"--{key.replace('_', '-')}", *_point_cm(merged, template_name, key)]
        argv += ["--camera-yaw-deg", str(float(_required(merged, template_name, "camera_yaw_deg")))]
        argv += ["--rir-stride-frames", _stride(merged)]
        argv += ["--variant", str(merged.get("variant", "A"))]
        if merged.get("rpc_port") is not None:
            argv += ["--rpc-port", str(int(merged["rpc_port"]))]
        if merged.get("graphics_adapter") is not None:
            argv += ["--graphics-adapter", str(int(merged["graphics_adapter"]))]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == PAIRED_ABLATION_TEMPLATE:
        audio_dir = _input_path(
            "audio_dir", _required(merged, template_name, "audio_dir"), repo
        )
        pair_id = str(_required(merged, template_name, "pair_id"))
        argv = [
            python,
            str(repo / "tools/studio/make_paired_ablation.py"),
            "--audio-dir", str(audio_dir),
            "--pair-id", pair_id,
        ]
        mute_stems = merged.get("mute_stems") or []
        if not isinstance(mute_stems, (list, tuple)):
            raise StudioTemplateError("mute_stems must be a list of stem names")
        for stem in mute_stems:
            argv += ["--mute-stem", str(stem)]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == HM3D_ROOM_PREPARE_TEMPLATE:
        split = str(merged.get("split", "val"))
        if split not in _HM3D_SPLITS:
            raise StudioTemplateError(
                f"split must be one of {sorted(_HM3D_SPLITS)}, got {split!r}"
            )
        argv = [python, str(repo / "tools/rooms/emit_hm3d_room_manifest.py")]
        _append_paths(
            argv,
            merged,
            template_name,
            ("scene_dir", "hm3d_root", *_HM3D_RUNTIME_KEYS),
            repo,
        )
        argv += ["--split", split]
        argv += [
            "--connectivity-samples",
            str(int(merged.get("connectivity_samples", 64))),
            "--connectivity-seed",
            str(int(merged.get("connectivity_seed", 20260826))),
        ]
        argv += ["--output-dir", _fresh_output(output_path)]
        return argv

    if template_name == SEMANTIC_PACKAGE_TEMPLATE:
        argv = [
            python,
            str(repo / "tools/acoustics/compile_semantic_research_package.py"),
        ]
        _append_paths(
            argv,
            merged,
            template_name,
            ("room_manifest", "material_rules"),
            repo,
            optional=("hm3d_root", "mp3d_root"),
        )
        if merged.get("verify_frame_parity"):
            _append_paths(
                argv,
                merged,
                template_name,
                ("runtime_prefix", "magnum_site", "rlr_sdk_root"),
                repo,
            )
            argv.append("--verify-frame-parity")
        argv += ["--seed", str(int(merged.get("seed", 20260826)))]
        if merged.get("package_id") is not None:
            argv += ["--package-id", str(merged["package_id"])]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == HM3D_ROUTE_BANK_TEMPLATE:
        argv = [
            python,
            str(repo / "tools/routes/compile_hm3d_dynamic_source_bank.py"),
        ]
        _append_paths(
            argv,
            merged,
            template_name,
            (*_HM3D_RUNTIME_KEYS, "scene", "navmesh"),
            repo,
        )
        for key, fallback in (
            ("seed", 20260826),
            ("episodes_per_motion_case", 8),
        ):
            argv += [f"--{key.replace('_', '-')}", str(int(merged.get(key, fallback)))]
        for key, fallback in (
            ("minimum_route_distance_m", 3.5),
            ("maximum_route_distance_m", 5.5),
            ("source1_height_m", 1.2),
            ("source2_height_m", 0.35),
        ):
            argv += [
                f"--{key.replace('_', '-')}",
                str(float(merged.get(key, fallback))),
            ]
        bounds = merged.get("room_bounds")
        if bounds is not None:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
                raise StudioTemplateError(
                    "room_bounds must be [x0, z0, x1, z1] from the room-prepare "
                    "stage's rooms.json"
                )
            argv += ["--room-bounds", *[str(float(value)) for value in bounds]]
            if merged.get("room_label") is not None:
                argv += ["--room-label", str(merged["room_label"])]
        # All three outputs live under one fresh directory; the tool creates
        # them itself with parents=True, so the directory need not pre-exist.
        out = Path(_fresh_output(output_path))
        argv += ["--bank-dir", str(out / "bank")]
        argv += ["--topdown-dir", str(out / "topdown")]
        argv += ["--report", str(out / "route_report.json")]
        return argv

    if template_name == HM3D_EPISODE_TEMPLATE:
        argv = [python, str(repo / "tools/studio/run_hm3d_episode.py")]
        _append_paths(argv, merged, template_name, _HM3D_EPISODE_PATH_KEYS, repo)
        argv += ["--scene-id", str(_required(merged, template_name, "scene_id"))]
        if merged.get("episode_id") is not None:
            argv += ["--episode-id", str(merged["episode_id"])]
        else:
            argv += ["--episode-index", str(int(merged.get("episode_index", 0)))]
        argv += ["--motion-case", str(merged.get("motion_case", "source1_moving_source2_static"))]
        argv += ["--slot", str(merged.get("slot", "source1"))]
        for key, fallback in (
            ("seed", 20260826),
            ("frame_stride", 1),
            ("width", 960),
            ("height", 540),
        ):
            argv += [f"--{key.replace('_', '-')}", str(int(merged.get(key, fallback)))]
        if merged.get("aim_open"):
            argv.append("--aim-open")
        if merged.get("overhead_m") is not None:
            argv += ["--overhead-m", str(float(merged["overhead_m"]))]
        if merged.get("place_at_emitter"):
            argv.append("--place-at-emitter")
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == HM3D_END_TO_END_TEMPLATE:
        split = str(merged.get("split", "val"))
        if split not in _HM3D_SPLITS:
            raise StudioTemplateError(
                f"split must be one of {sorted(_HM3D_SPLITS)}, got {split!r}"
            )
        argv = [python, str(repo / "tools/studio/run_hm3d_end_to_end.py")]
        _append_paths(argv, merged, template_name, _HM3D_E2E_PATH_KEYS, repo)
        argv += ["--split", split]
        argv += ["--seed", str(int(merged.get("seed", 20260826)))]
        argv += [
            "--connectivity-samples",
            str(int(merged.get("connectivity_samples", 64))),
            "--episodes-per-motion-case",
            str(int(merged.get("episodes_per_motion_case", 8))),
        ]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == KUJIALE_ROUTE_BANK_TEMPLATE:
        argv = [
            python,
            str(repo / "tools/routes/compile_kujiale_feasibility_bank.py"),
        ]
        _append_paths(argv, merged, template_name, ("scene_metadata",), repo)
        argv += ["--seed", str(int(merged.get("seed", 20260721)))]
        argv += [
            "--episodes-per-motion-case",
            str(int(merged.get("episodes_per_motion_case", 50))),
        ]
        for key, fallback in (
            ("minimum_route_distance_m", 3.5),
            ("maximum_route_distance_m", 5.5),
        ):
            argv += [
                f"--{key.replace('_', '-')}",
                str(float(merged.get(key, fallback))),
            ]
        if merged.get("minimum_clearance_m") is not None:
            argv += [
                "--minimum-clearance-m",
                str(float(merged["minimum_clearance_m"])),
            ]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == KUJIALE_ACOUSTIC_PACKAGE_TEMPLATE:
        argv = [python, str(repo / "tools/studio/run_kujiale_acoustic_package.py")]
        _append_paths(
            argv, merged, template_name, ("source_usd", "material_rules"), repo
        )
        argv += ["--room-id", str(_required(merged, template_name, "room_id"))]
        argv += [
            "--transform-profile",
            str(_required(merged, template_name, "transform_profile")),
        ]
        origins = _required(merged, template_name, "interior_origins")
        if (
            not isinstance(origins, (list, tuple))
            or len(origins) < 2
            or not all(
                isinstance(origin, (list, tuple)) and len(origin) == 3
                for origin in origins
            )
        ):
            raise StudioTemplateError(
                "interior_origins must be a list of at least two [x, y, z] "
                "points reviewed to lie inside the room"
            )
        for origin in origins:
            argv += ["--interior-origin", *[str(float(v)) for v in origin]]
        argv += [
            "--source-revision",
            str(_required(merged, template_name, "source_revision")),
            "--dataset-id",
            str(_required(merged, template_name, "dataset_id")),
            "--source-license",
            str(_required(merged, template_name, "source_license")),
        ]
        argv += ["--seed", str(int(merged.get("seed", 20260827)))]
        argv += ["--output", _fresh_output(output_path)]
        return argv

    if template_name == KUJIALE_VISUAL_EPISODE_TEMPLATE:
        argv = [python, str(repo / "tools/rooms/run_spear_residential_episode.py")]
        _append_paths(
            argv,
            merged,
            template_name,
            ("episode_root", "uproject", "unreal_editor"),
            repo,
            optional=("spear_ext_dir",),
        )
        if merged.get("rpc_port") is not None:
            argv += ["--rpc-port", str(int(merged["rpc_port"]))]
        if merged.get("graphics_adapter") is not None:
            argv += ["--graphics-adapter", str(int(merged["graphics_adapter"]))]
        # An episode root without an audio/ directory can only be replayed as
        # visual-only research; muxing against the absent mixture is how the
        # first live run of this template died at its final step.
        if merged.get("visual_only_research"):
            argv.append("--visual-only-research")
        argv += ["--output", _fresh_output(output_path)]
        return argv

    raise StudioTemplateError(f"no argv builder for template {template_name!r}")
