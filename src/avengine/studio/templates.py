"""Argv builders for the render task templates the Studio may queue.

Session 1 supports one template: the MP3D motion-following dynamic audio
verb. Defaults come from the deployment config; a submission may override
only the authoring-facing keys. Every input path is validated at submit
time so a bad request fails as HTTP 400 instead of a queued task failure.
"""

from __future__ import annotations

from pathlib import Path

from avengine.studio.config import StudioConfig

MP3D_DYNAMIC_AUDIO_TEMPLATE = "mp3d_dynamic_audio"

_MP3D_REQUIRED_PATH_KEYS = (
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
_MP3D_OPTIONAL_PATH_KEYS = ("hrtf_license", "magnum_python_site")

TEMPLATE_OVERRIDABLE_KEYS: dict[str, frozenset[str]] = {
    MP3D_DYNAMIC_AUDIO_TEMPLATE: frozenset(
        {"visual_capture_dir", "audio_program", "variant", "rir_stride_frames"}
    ),
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


def build_template_argv(
    config: StudioConfig,
    template_name: str,
    overrides: dict[str, object],
    output_path: Path,
) -> list[str]:
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

    argv = [
        str(config.python_executable),
        "-m",
        "avengine.cli",
        "m5",
        "render-current-mp3d-dynamic-audio",
    ]
    for key in _MP3D_REQUIRED_PATH_KEYS:
        value = merged.get(key)
        if value is None:
            raise StudioTemplateError(
                f"template {template_name!r} is missing required input {key!r}"
            )
        path = _input_path(key, value, config.repository_root)
        argv += [f"--{key.replace('_', '-')}", str(path)]
    for key in _MP3D_OPTIONAL_PATH_KEYS:
        value = merged.get(key)
        if value is not None:
            path = _input_path(key, value, config.repository_root)
            argv += [f"--{key.replace('_', '-')}", str(path)]

    stride = int(merged.get("rir_stride_frames", 3))
    if stride < 1:
        raise StudioTemplateError(f"rir_stride_frames must be >= 1, got {stride}")
    argv += ["--rir-stride-frames", str(stride)]
    argv += ["--variant", str(merged.get("variant", "A"))]

    resolved_output = Path(output_path).resolve()
    if resolved_output.exists():
        raise StudioTemplateError(
            f"output path already exists (fresh/no-clobber): {resolved_output}"
        )
    argv += ["--output", str(resolved_output)]
    return argv
