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
    MP3D_DYNAMIC_AUDIO_TEMPLATE: frozenset(
        {"visual_capture_dir", "audio_program", "variant", "rir_stride_frames"}
    ),
    MP3D_END_TO_END_TEMPLATE: frozenset(
        {"seed", "camera_selection", "audio_program", "variant", "rir_stride_frames"}
    ),
    MP3D_ROUTE_AUTHOR_TEMPLATE: frozenset({"seed", "camera_selection"}),
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

    raise StudioTemplateError(f"no argv builder for template {template_name!r}")
