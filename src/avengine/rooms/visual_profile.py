"""Review-only visual profile for the fixed SPEAR Apartment canary.

This module deliberately does not mutate the Apartment room package.  It owns
three replaceable review concerns instead: the native capture resolution, a
Habitat light setup, and an optional transient exterior proxy.  The proxy is
instantiated only inside the RGB/depth/semantic capture simulator.  It renders
as a distant surface in RGB/depth and as background semantic ID 0, but is not
present in the room SceneInstance, placement simulator, Topdown obstacle map,
navmesh, or RLR acoustic package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from avengine.contracts.json_io import load_json, sha256_file
from avengine.timeline.video import (
    _encode_h264_video_profile,
    _mux_binaural_wav_profile,
)
from avengine.capture.review import SourceOverlayTrack, compose_annotated_frames


VISUAL_PROFILE_SCHEMA = "avengine_m6x_review_visual_profile_v1"


class M6XVisualProfileError(ValueError):
    """A review visual profile or its runtime realization is invalid."""


def resolve_review_capture_rgb_path(capture_directory: str | Path) -> Path:
    """Resolve the producer's declared RGB artifact, including legacy layouts."""
    directory = Path(capture_directory).resolve()
    receipt_path = directory / "research_receipt.json"
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        if not isinstance(receipt, Mapping):
            raise M6XVisualProfileError("capture receipt must be an object")
        artifacts = receipt.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise M6XVisualProfileError("capture artifacts must be an object")
        declared = artifacts.get("rgb")
        if declared is not None:
            if not isinstance(declared, str) or not declared:
                raise M6XVisualProfileError("declared RGB artifact must be a file path")
            path = Path(declared).expanduser()
            path = path if path.is_absolute() else directory / path
            if not path.is_file():
                raise M6XVisualProfileError(f"declared RGB artifact is missing: {path}")
            return path.resolve()
    for relative in ("arrays/rgb.npy", "rgb.npy"):
        path = directory / relative
        if path.is_file():
            return path.resolve()
    raise M6XVisualProfileError(f"capture has no declared or legacy RGB artifact: {directory}")


def resolve_review_capture_channel_order(
    capture_directory: str | Path, override: str | None = None,
) -> str:
    """Use producer metadata; explicit legacy ordering cannot contradict it."""
    receipt = Path(capture_directory) / "research_receipt.json"
    declared = None
    if receipt.is_file():
        value = load_json(receipt)
        if not isinstance(value, Mapping):
            raise M6XVisualProfileError("capture receipt must be an object")
        capture = value.get("capture", {})
        if not isinstance(capture, Mapping):
            raise M6XVisualProfileError("capture receipt capture must be an object")
        declared = capture.get("rgb_channel_order")
    for owner, value in (("declared", declared), ("override", override)):
        if value is not None and (not isinstance(value, str) or value not in {"rgb", "bgr"}):
            raise M6XVisualProfileError(f"{owner} channel order must be rgb or bgr")
    if declared is not None and override is not None and declared != override:
        raise M6XVisualProfileError("channel-order override differs from capture receipt")
    # Historical Habitat/raw inputs used RGB; legacy UE callers already pass BGR.
    return override or declared or "rgb"


@dataclass(frozen=True)
class ReviewVisualProfile:
    """Validated values used by capture and review-video assembly."""

    path: Path
    raw: Mapping[str, Any]
    profile_id: str
    capture_resolution_hw: tuple[int, int]
    diagnostic_panel_resolution_hw: tuple[int, int]
    capture_frame_rate_hz: float = 15.0

    @property
    def capture_height(self) -> int:
        return self.capture_resolution_hw[0]

    @property
    def capture_width(self) -> int:
        return self.capture_resolution_hw[1]

    @property
    def diagnostic_height(self) -> int:
        return self.diagnostic_panel_resolution_hw[0]

    @property
    def diagnostic_width(self) -> int:
        return self.diagnostic_panel_resolution_hw[1]


def _nonempty_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise M6XVisualProfileError(f"{owner} must be a nonempty string")
    return value.strip()


def _resolution(value: Any, *, owner: str, minimum: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise M6XVisualProfileError(f"{owner} must be [height,width] integers")
    height, width = int(value[0]), int(value[1])
    if height < minimum[0] or width < minimum[1] or height % 2 or width % 2:
        raise M6XVisualProfileError(
            f"{owner} must be even and at least [{minimum[0]},{minimum[1]}]"
        )
    return height, width


def _finite_vector(value: Any, *, owner: str, length: int) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M6XVisualProfileError(
            f"{owner} must contain {length} finite numbers"
        ) from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise M6XVisualProfileError(f"{owner} must contain {length} finite numbers")
    return tuple(float(item) for item in array)


def load_review_visual_profile(path: str | Path) -> ReviewVisualProfile:
    """Load the small, human-editable M6.x review profile."""

    resolved = Path(path).resolve()
    value = load_json(resolved)
    if value.get("schema") != VISUAL_PROFILE_SCHEMA:
        raise M6XVisualProfileError(f"visual profile schema must be {VISUAL_PROFILE_SCHEMA!r}")
    profile_id = _nonempty_string(value.get("profile_id"), owner="profile_id")
    capture = value.get("capture")
    if not isinstance(capture, Mapping):
        raise M6XVisualProfileError("capture must be an object")
    capture_resolution = _resolution(
        capture.get("resolution_hw"), owner="capture.resolution_hw", minimum=(480, 640)
    )
    capture_frame_rate = capture.get("frame_rate_hz")
    if (
        isinstance(capture_frame_rate, bool)
        or not isinstance(capture_frame_rate, (int, float))
        or not math.isfinite(float(capture_frame_rate))
        or float(capture_frame_rate) <= 0.0
    ):
        raise M6XVisualProfileError(
            "capture.frame_rate_hz must be a finite positive number"
        )
    clean = capture.get("clean_video")
    if not isinstance(clean, Mapping) or clean != {
        "codec": "h264",
        "crf": 18,
        "preset": "medium",
    }:
        raise M6XVisualProfileError(
            "capture.clean_video must freeze h264/CRF18/medium for review"
        )
    diagnostic = capture.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise M6XVisualProfileError("capture.diagnostic must be an object")
    diagnostic_resolution = _resolution(
        diagnostic.get("panel_resolution_hw"),
        owner="capture.diagnostic.panel_resolution_hw",
        minimum=(240, 320),
    )
    if diagnostic_resolution != (480, 640):
        raise M6XVisualProfileError(
            "the current annotated diagnostic layout requires [480,640] panels"
        )
    if diagnostic.get("main_panel_policy") != (
        "aspect_preserving_downscale_no_upscale"
    ):
        raise M6XVisualProfileError(
            "capture.diagnostic.main_panel_policy must forbid upscaling"
        )
    if diagnostic.get("composition") != "main_plus_topdown":
        raise M6XVisualProfileError(
            "capture.diagnostic.composition must be 'main_plus_topdown'"
        )
    if (
        diagnostic_resolution[0] > capture_resolution[0]
        or diagnostic_resolution[1] > capture_resolution[1]
    ):
        raise M6XVisualProfileError(
            "diagnostic panel cannot exceed the native capture resolution"
        )

    lighting = value.get("lighting")
    if not isinstance(lighting, Mapping) or lighting.get("hbao") is not True:
        raise M6XVisualProfileError("lighting must explicitly keep HBAO enabled")
    _nonempty_string(lighting.get("setup_id"), owner="lighting.setup_id")
    _nonempty_string(lighting.get("claim_boundary"), owner="lighting.claim_boundary")
    lights = lighting.get("lights")
    if not isinstance(lights, list) or not lights:
        raise M6XVisualProfileError("lighting.lights must be a nonempty array")
    light_ids: set[str] = set()
    directional_count = 0
    for index, light in enumerate(lights):
        owner = f"lighting.lights[{index}]"
        if not isinstance(light, Mapping):
            raise M6XVisualProfileError(f"{owner} must be an object")
        light_id = _nonempty_string(light.get("light_id"), owner=f"{owner}.light_id")
        if light_id in light_ids:
            raise M6XVisualProfileError("lighting light_id values must be unique")
        light_ids.add(light_id)
        light_type = light.get("type")
        if light_type == "directional":
            direction = np.asarray(
                _finite_vector(light.get("direction"), owner=f"{owner}.direction", length=3)
            )
            if float(np.linalg.norm(direction)) <= 1.0e-12:
                raise M6XVisualProfileError(f"{owner}.direction must be nonzero")
            directional_count += 1
        elif light_type == "point":
            _finite_vector(light.get("position"), owner=f"{owner}.position", length=3)
        else:
            raise M6XVisualProfileError(f"{owner}.type must be directional or point")
        color = _finite_vector(light.get("color"), owner=f"{owner}.color", length=3)
        if any(channel < 0.0 for channel in color):
            raise M6XVisualProfileError(f"{owner}.color cannot be negative")
        intensity = light.get("intensity")
        if (
            isinstance(intensity, bool)
            or not isinstance(intensity, (int, float))
            or not math.isfinite(float(intensity))
            or float(intensity) <= 0.0
        ):
            raise M6XVisualProfileError(f"{owner}.intensity must be positive")
        if light.get("position_model") not in {"global", "camera", "object"}:
            raise M6XVisualProfileError(
                f"{owner}.position_model must be global, camera, or object"
            )
    if directional_count < 2:
        raise M6XVisualProfileError(
            "review lighting must declare a directional key and directional fill"
        )

    exterior = value.get("exterior_proxy")
    if not isinstance(exterior, Mapping) or exterior.get("enabled") is not True:
        raise M6XVisualProfileError("exterior_proxy must be explicitly enabled")
    if exterior.get("runtime_asset_argument") != "exterior_proxy_glb_path":
        raise M6XVisualProfileError(
            "exterior_proxy.runtime_asset_argument must be exterior_proxy_glb_path"
        )
    if exterior.get("proxy_kind") != (
        "inward_uv_sphere_with_direction_projected_window_panels"
    ):
        raise M6XVisualProfileError(
            "exterior proxy must use listener-direction-projected window panels"
        )
    if exterior.get("center") != "camera_listener":
        raise M6XVisualProfileError("exterior proxy must be centered on camera_listener")
    scale = _finite_vector(
        exterior.get("scale_xyz"), owner="exterior_proxy.scale_xyz", length=3
    )
    if any(item <= 0.0 for item in scale):
        raise M6XVisualProfileError("exterior_proxy.scale_xyz must be positive")
    if scale != (1.0, 1.0, 1.0):
        raise M6XVisualProfileError(
            "the room-aligned exterior GLB must use unit runtime scale"
        )
    radius = exterior.get("sphere_radius_m")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(float(radius))
        or float(radius) <= 0.0
    ):
        raise M6XVisualProfileError("exterior_proxy.sphere_radius_m must be positive")
    panels = exterior.get("window_panels")
    if not isinstance(panels, list) or not panels:
        raise M6XVisualProfileError("exterior_proxy.window_panels must be nonempty")
    panel_ids: set[str] = set()
    for index, panel in enumerate(panels):
        owner = f"exterior_proxy.window_panels[{index}]"
        if not isinstance(panel, Mapping):
            raise M6XVisualProfileError(f"{owner} must be an object")
        panel_id = _nonempty_string(panel.get("panel_id"), owner=f"{owner}.panel_id")
        if panel_id in panel_ids:
            raise M6XVisualProfileError("exterior window panel IDs must be unique")
        panel_ids.add(panel_id)
        center = _finite_vector(
            panel.get("center_from_listener_m"),
            owner=f"{owner}.center_from_listener_m",
            length=3,
        )
        del center
        width_axis = np.asarray(
            _finite_vector(
                panel.get("width_axis"), owner=f"{owner}.width_axis", length=3
            )
        )
        height_axis = np.asarray(
            _finite_vector(
                panel.get("height_axis"), owner=f"{owner}.height_axis", length=3
            )
        )
        if (
            not np.isclose(np.linalg.norm(width_axis), 1.0, atol=1.0e-6)
            or not np.isclose(np.linalg.norm(height_axis), 1.0, atol=1.0e-6)
            or not np.isclose(np.dot(width_axis, height_axis), 0.0, atol=1.0e-6)
        ):
            raise M6XVisualProfileError(
                f"{owner} width/height axes must be orthonormal"
            )
        size = _finite_vector(
            panel.get("size_wh_m"), owner=f"{owner}.size_wh_m", length=2
        )
        if any(item <= 0.0 for item in size):
            raise M6XVisualProfileError(f"{owner}.size_wh_m must be positive")
        subdivisions = panel.get("grid_subdivisions_wh")
        if (
            not isinstance(subdivisions, list)
            or len(subdivisions) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 2
                or item > 128
                for item in subdivisions
            )
        ):
            raise M6XVisualProfileError(
                f"{owner}.grid_subdivisions_wh must be two integers in [2,128]"
            )
        if panel.get("uv_projection") != "listener_direction_equirectangular":
            raise M6XVisualProfileError(
                f"{owner}.uv_projection must use listener-direction mapping"
            )
    capture_objects = exterior.get("capture_scene_objects")
    if not isinstance(capture_objects, Mapping):
        raise M6XVisualProfileError(
            "exterior_proxy.capture_scene_objects must be an object"
        )
    prefixes = capture_objects.get("remove_handle_prefixes")
    if prefixes != ["source_marker_"]:
        raise M6XVisualProfileError(
            "capture scene must remove only the legacy source_marker_ objects"
        )
    if capture_objects.get("expected_removed_count") != 2:
        raise M6XVisualProfileError(
            "capture scene must remove exactly the two legacy source markers"
        )
    if (
        capture_objects.get("logical_source_representation")
        != "topdown_timeline_and_audio_only"
    ):
        raise M6XVisualProfileError(
            "removed source markers must remain represented by logical evidence"
        )
    if exterior.get("shader_type") != "flat" or exterior.get("semantic_id") != 0:
        raise M6XVisualProfileError(
            "exterior proxy must use flat shading and background semantic ID 0"
        )
    if exterior.get("collidable") is not False:
        raise M6XVisualProfileError("exterior proxy must be non-collidable")
    required_exclusions = {
        "room_scene_instance",
        "collision",
        "navmesh",
        "topdown_obstacle_map",
        "rlr_acoustic_geometry",
    }
    if set(exterior.get("excluded_from", ())) != required_exclusions:
        raise M6XVisualProfileError(
            "exterior_proxy.excluded_from must name every out-of-capture subsystem"
        )
    if (
        exterior.get("semantic_behavior") != "background_id_zero"
        or exterior.get("depth_behavior")
        != "renders_as_far_exterior_surface"
    ):
        raise M6XVisualProfileError(
            "exterior proxy must declare its semantic/depth capture behavior"
        )
    return ReviewVisualProfile(
        path=resolved,
        raw=value,
        profile_id=profile_id,
        capture_resolution_hw=capture_resolution,
        diagnostic_panel_resolution_hw=diagnostic_resolution,
        capture_frame_rate_hz=float(capture_frame_rate),
    )


def validate_profile_capture_request(
    profile: ReviewVisualProfile, request: Mapping[str, Any]
) -> None:
    """Require the native sensor request to match the review profile exactly."""

    try:
        resolution = request["primary_camera_rig"]["shared_calibration"][
            "resolution_hw"
        ]
    except (KeyError, TypeError) as exc:
        raise M6XVisualProfileError(
            "capture request has no primary camera shared resolution"
        ) from exc
    if tuple(resolution) != profile.capture_resolution_hw:
        raise M6XVisualProfileError(
            "capture request resolution differs from the review visual profile"
        )


def validate_realized_review_profile(
    evidence: Mapping[str, Any],
    *,
    profile: ReviewVisualProfile,
    exterior_proxy_glb_path: str | Path,
) -> None:
    """Bind a fresh or retained capture to the selected visual inputs."""

    realized = evidence.get("review_visual_profile")
    proxy_path = Path(exterior_proxy_glb_path).resolve()
    if not isinstance(realized, Mapping):
        raise M6XVisualProfileError(
            "capture lacks realized review_visual_profile evidence"
        )
    exterior = realized.get("exterior_proxy")
    capture_objects = realized.get("capture_scene_objects")
    configured = realized.get("configuration")
    final_lighting = realized.get("final_light_readback")
    expected_excluded = profile.raw["exterior_proxy"]["excluded_from"]
    if (
        realized.get("status") != "pass"
        or realized.get("profile_id") != profile.profile_id
        or realized.get("profile_sha256") != sha256_file(profile.path)
        or realized.get("native_capture_resolution_hw")
        != list(profile.capture_resolution_hw)
        or not isinstance(configured, Mapping)
        or configured.get("status") != "pass"
        or configured.get("setup_id") != profile.raw["lighting"]["setup_id"]
        or configured.get("override_scene_light_defaults") is not True
        or not isinstance(exterior, Mapping)
        or exterior.get("prepared_glb_sha256") != sha256_file(proxy_path)
        or exterior.get("collidable_readback") is not False
        or exterior.get("semantic_id_readback") != 0
        or exterior.get("semantic_behavior") != "background_id_zero"
        or exterior.get("depth_behavior") != "renders_as_far_exterior_surface"
        or exterior.get("excluded_from") != expected_excluded
        or exterior.get("scope") != "transient_visual_capture_simulator_only"
        or exterior.get("proxy_kind")
        != profile.raw["exterior_proxy"]["proxy_kind"]
        or exterior.get("window_panel_ids")
        != [
            panel["panel_id"]
            for panel in profile.raw["exterior_proxy"]["window_panels"]
        ]
        or not isinstance(capture_objects, Mapping)
        or capture_objects.get("removed_handle_prefixes")
        != profile.raw["exterior_proxy"]["capture_scene_objects"][
            "remove_handle_prefixes"
        ]
        or capture_objects.get("removed_count")
        != profile.raw["exterior_proxy"]["capture_scene_objects"][
            "expected_removed_count"
        ]
        or capture_objects.get("remaining_matching_count") != 0
        or capture_objects.get("logical_source_representation")
        != profile.raw["exterior_proxy"]["capture_scene_objects"][
            "logical_source_representation"
        ]
        or not isinstance(final_lighting, Mapping)
        or final_lighting.get("status") != "pass"
        or final_lighting.get("expected_light_count")
        != len(profile.raw["lighting"]["lights"])
        or final_lighting.get("current_matches_profile") is not True
        or final_lighting.get("actor_setup_matches_profile") is not True
    ):
        raise M6XVisualProfileError(
            "capture review_visual_profile evidence differs from current inputs"
        )


def light_setup_records(profile: ReviewVisualProfile) -> tuple[dict[str, Any], ...]:
    """Return normalized light vectors/colors without importing Habitat."""

    records: list[dict[str, Any]] = []
    for raw in profile.raw["lighting"]["lights"]:
        if raw["type"] == "directional":
            direction = np.asarray(raw["direction"], dtype=np.float64)
            direction /= np.linalg.norm(direction)
            vector = (*[float(item) for item in direction], 0.0)
        else:
            vector = (*[float(item) for item in raw["position"]], 1.0)
        color = tuple(
            float(channel) * float(raw["intensity"]) for channel in raw["color"]
        )
        records.append(
            {
                "light_id": raw["light_id"],
                "type": raw["type"],
                "vector_xyzw": vector,
                "color_rgb": color,
                "position_model": raw["position_model"],
            }
        )
    return tuple(records)


def _habitat_light_setup(
    profile: ReviewVisualProfile, habitat_sim: Any
) -> tuple[Any, ...]:
    models = {
        "global": habitat_sim.gfx.LightPositionModel.Global,
        "camera": habitat_sim.gfx.LightPositionModel.Camera,
        "object": habitat_sim.gfx.LightPositionModel.Object,
    }
    return tuple(
        habitat_sim.gfx.LightInfo(
            vector=record["vector_xyzw"],
            color=record["color_rgb"],
            model=models[record["position_model"]],
        )
        for record in light_setup_records(profile)
    )


def validate_runtime_review_light_readback(
    simulator: Any,
    *,
    profile: ReviewVisualProfile,
    habitat_sim: Any,
    actor_light_setup_key: str,
) -> dict[str, Any]:
    """Prove the final scene and articulated-actor keys use profile lights.

    This check intentionally runs after the M5.1 actor-light binding.  It
    catches either a later scene reset or actors retaining the Apartment's old
    point-light setup while the room uses the review profile.
    """

    if not isinstance(actor_light_setup_key, str) or not actor_light_setup_key:
        raise M6XVisualProfileError("actor light setup key must be nonempty")
    expected = list(_habitat_light_setup(profile, habitat_sim))
    current = list(simulator.get_current_light_setup())
    actors = list(simulator.get_light_setup(actor_light_setup_key))
    current_matches = current == expected
    actors_match = actors == expected
    if not current_matches or not actors_match:
        raise M6XVisualProfileError(
            "final scene/actor light setup differs from the review profile"
        )
    return {
        "status": "pass",
        "expected_light_count": len(expected),
        "current_light_count": len(current),
        "actor_light_count": len(actors),
        "current_matches_profile": current_matches,
        "actor_setup_matches_profile": actors_match,
        "actor_light_setup_key": actor_light_setup_key,
    }


def configure_runtime_review_profile(
    configuration: Any,
    *,
    profile: ReviewVisualProfile,
    habitat_sim: Any,
) -> dict[str, Any]:
    """Bind the stage to Habitat's mutable default light key before loading.

    Dataset-authored light layouts are installed as finalized Magnum
    resources.  They cannot be replaced after Simulator construction.  The
    documented Habitat route is to override the scene default in the
    configuration, construct the stage against ``DEFAULT_LIGHTING_KEY``, and
    then update that mutable setup.
    """

    sim_cfg = configuration.sim_cfg
    previous_key = str(sim_cfg.scene_light_setup)
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    sim_cfg.override_scene_light_defaults = True
    if (
        str(sim_cfg.scene_light_setup)
        != str(habitat_sim.gfx.DEFAULT_LIGHTING_KEY)
        or bool(sim_cfg.override_scene_light_defaults) is not True
    ):
        raise M6XVisualProfileError(
            "review lighting configuration override did not read back"
        )
    return {
        "status": "pass",
        "setup_id": profile.raw["lighting"]["setup_id"],
        "previous_scene_light_setup_key": previous_key,
        "configured_scene_light_setup_key": str(sim_cfg.scene_light_setup),
        "override_scene_light_defaults": True,
    }


def apply_runtime_review_profile(
    simulator: Any,
    *,
    profile: ReviewVisualProfile,
    exterior_proxy_glb_path: str | Path,
    camera_listener_position_m: Sequence[float],
    habitat_sim: Any,
    mn: Any,
) -> dict[str, Any]:
    """Realize review lights, clean capture objects, and the exterior proxy."""

    proxy_path = Path(exterior_proxy_glb_path).resolve()
    if not proxy_path.is_file() or proxy_path.suffix.casefold() != ".glb":
        raise M6XVisualProfileError(
            "prepared approaching_storm exterior proxy GLB is missing; run "
            "tools/rooms/prepare_spear_apartment_exterior.py first"
        )
    camera = _finite_vector(
        camera_listener_position_m, owner="camera_listener_position_m", length=3
    )
    records = light_setup_records(profile)
    lights = list(_habitat_light_setup(profile, habitat_sim))
    scene_key = str(simulator.config.sim_cfg.scene_light_setup)
    if (
        scene_key != str(habitat_sim.gfx.DEFAULT_LIGHTING_KEY)
        or bool(simulator.config.sim_cfg.override_scene_light_defaults) is not True
    ):
        raise M6XVisualProfileError(
            "review Simulator was not constructed with the mutable light override"
        )
    simulator.set_light_setup(lights, scene_key)
    if list(simulator.get_current_light_setup()) != lights:
        raise M6XVisualProfileError("review scene light setup did not read back")

    exterior = profile.raw["exterior_proxy"]
    object_manager = simulator.get_rigid_object_manager()
    capture_objects = exterior["capture_scene_objects"]
    prefixes = tuple(capture_objects["remove_handle_prefixes"])
    existing = tuple(
        item
        for item in object_manager.get_objects_by_handle_substring().values()
        if str(item.handle).startswith(prefixes)
    )
    if len(existing) != int(capture_objects["expected_removed_count"]):
        raise M6XVisualProfileError(
            "capture simulator does not contain the expected legacy source markers"
        )
    removed = [
        {"object_id": int(item.object_id), "handle": str(item.handle)}
        for item in existing
    ]
    for item in existing:
        object_manager.remove_object_by_id(int(item.object_id))
    remaining = tuple(
        item
        for item in object_manager.get_objects_by_handle_substring().values()
        if str(item.handle).startswith(prefixes)
    )
    if remaining:
        raise M6XVisualProfileError(
            "legacy source markers remained visible in the capture simulator"
        )

    manager = simulator.get_object_template_manager()
    attributes = manager.create_new_template(str(proxy_path), False)
    if attributes is None:
        raise M6XVisualProfileError("Habitat could not create the exterior proxy template")
    # ``create_new_template(path, False)`` already seeds both asset handles.
    # Reassigning the same GLB here makes the pinned Habitat resource manager
    # attempt to replace a finalized resource while the furnished stage is
    # loaded, which aborts in native code before Python can raise an error.
    attributes.scale = mn.Vector3(exterior["scale_xyz"])
    attributes.is_collidable = False
    attributes.shader_type = "flat"
    attributes.semantic_id = 0
    template_handle = f"{profile.profile_id}.visual_only_exterior"
    template_id = int(manager.register_template(attributes, template_handle))
    if template_id < 0:
        raise M6XVisualProfileError("Habitat could not register the exterior proxy")
    proxy = simulator.get_rigid_object_manager().add_object_by_template_id(
        template_id, light_setup_key=scene_key
    )
    if proxy is None:
        raise M6XVisualProfileError("Habitat could not instantiate the exterior proxy")
    proxy.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    proxy.collidable = False
    proxy.translation = mn.Vector3(camera)
    proxy.rotation = mn.Quaternion.rotation(
        mn.Deg(float(exterior["yaw_degrees"])), mn.Vector3.y_axis()
    )
    proxy.semantic_id = 0
    for node in proxy.visual_scene_nodes:
        node.semantic_id = 0
    if bool(proxy.collidable) or int(proxy.semantic_id) != 0:
        raise M6XVisualProfileError(
            "exterior proxy collision/semantic readback violates the visual-only policy"
        )
    return {
        "status": "pass",
        "profile_id": profile.profile_id,
        "profile_sha256": sha256_file(profile.path),
        "native_capture_resolution_hw": list(profile.capture_resolution_hw),
        "lighting": {
            "setup_id": profile.raw["lighting"]["setup_id"],
            "scene_light_setup_key": scene_key,
            "override_scene_light_defaults_readback": True,
            "lights": list(records),
            "hbao": True,
            "claim_boundary": profile.raw["lighting"]["claim_boundary"],
        },
        "capture_scene_objects": {
            "removed_handle_prefixes": list(prefixes),
            "removed_count": len(removed),
            "removed_objects": removed,
            "remaining_matching_count": len(remaining),
            "logical_source_representation": capture_objects[
                "logical_source_representation"
            ],
        },
        "exterior_proxy": {
            "source_asset_uri": exterior["source_asset_uri"],
            "proxy_kind": exterior["proxy_kind"],
            "window_panel_ids": [
                panel["panel_id"] for panel in exterior["window_panels"]
            ],
            "prepared_glb_path": str(proxy_path),
            "prepared_glb_sha256": sha256_file(proxy_path),
            "object_id": int(proxy.object_id),
            "collidable_readback": bool(proxy.collidable),
            "semantic_id_readback": int(proxy.semantic_id),
            "excluded_from": list(exterior["excluded_from"]),
            "semantic_behavior": exterior["semantic_behavior"],
            "depth_behavior": exterior["depth_behavior"],
            "scope": "transient_visual_capture_simulator_only",
        },
    }


def aspect_preserving_downscale_no_upscale(
    frames: Any, *, target_resolution_hw: tuple[int, int]
) -> np.ndarray:
    """Letterbox native frames into a panel while refusing to invent pixels."""

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise M6XVisualProfileError(
            "review frames must be uint8 [frame,height,width,3] RGB"
        )
    target_h, target_w = target_resolution_hw
    source_h, source_w = int(array.shape[1]), int(array.shape[2])
    scale = min(target_w / source_w, target_h / source_h)
    if scale > 1.0 + 1.0e-12:
        raise M6XVisualProfileError(
            "diagnostic composition refuses to upscale a low-resolution source"
        )
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    result = np.zeros((array.shape[0], target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - resized_w) // 2
    y0 = (target_h - resized_h) // 2
    for index, frame in enumerate(array):
        resized = Image.fromarray(frame, mode="RGB").resize(
            (resized_w, resized_h), Image.Resampling.LANCZOS
        )
        result[index, y0 : y0 + resized_h, x0 : x0 + resized_w] = np.asarray(
            resized, dtype=np.uint8
        )
    return np.ascontiguousarray(result)


def letterbox_marker_coordinates(
    markers_xy: Any,
    *,
    source_resolution_hw: tuple[int, int],
    target_resolution_hw: tuple[int, int],
) -> np.ndarray:
    """Apply the exact RGB letterbox transform to pixel marker coordinates."""

    markers = np.asarray(markers_xy, dtype=np.float64)
    if markers.ndim != 2 or markers.shape[1] != 2:
        raise M6XVisualProfileError("marker coordinates must have shape [frame,2]")
    finite_or_nan = np.logical_or(
        np.all(np.isfinite(markers), axis=1), np.all(np.isnan(markers), axis=1)
    )
    if not np.all(finite_or_nan):
        raise M6XVisualProfileError("markers must be finite pairs or NaN pairs")
    source_h, source_w = source_resolution_hw
    target_h, target_w = target_resolution_hw
    scale = min(target_w / source_w, target_h / source_h)
    if scale > 1.0 + 1.0e-12:
        raise M6XVisualProfileError("marker transform refuses to upscale")
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    offset = np.asarray(
        ((target_w - resized_w) // 2, (target_h - resized_h) // 2),
        dtype=np.float64,
    )
    result = markers.copy()
    finite = np.all(np.isfinite(markers), axis=1)
    result[finite] = markers[finite] * scale + offset
    return np.ascontiguousarray(result)


def compose_profiled_annotated_frames(
    *, profile: ReviewVisualProfile, main_rgb: Any, topdown_rgb: Any, **kwargs: Any
) -> np.ndarray:
    """Compose the existing annotation grammar without low-resolution upscaling."""

    main = np.asarray(main_rgb)
    expected = (profile.capture_height, profile.capture_width, 3)
    if main.ndim != 4 or tuple(main.shape[1:]) != expected:
        raise M6XVisualProfileError(
            f"main_rgb must use native profile shape [frame,{expected[0]},{expected[1]},3]"
        )
    prepared_main = aspect_preserving_downscale_no_upscale(
        main, target_resolution_hw=profile.diagnostic_panel_resolution_hw
    )
    topdown = np.asarray(topdown_rgb)
    if tuple(topdown.shape[1:]) != (
        profile.diagnostic_height,
        profile.diagnostic_width,
        3,
    ):
        raise M6XVisualProfileError(
            "Topdown frames must be rendered at the declared diagnostic panel size"
        )
    tracks = kwargs.get("tracks")
    if not isinstance(tracks, Sequence):
        raise M6XVisualProfileError("annotated review tracks must be a sequence")
    transformed_tracks: list[SourceOverlayTrack] = []
    for track in tracks:
        if not isinstance(track, SourceOverlayTrack):
            raise M6XVisualProfileError(
                "annotated review tracks must contain SourceOverlayTrack values"
            )
        markers = track.main_marker_xy
        transformed_tracks.append(
            track
            if markers is None
            else replace(
                track,
                main_marker_xy=letterbox_marker_coordinates(
                    markers,
                    source_resolution_hw=profile.capture_resolution_hw,
                    target_resolution_hw=profile.diagnostic_panel_resolution_hw,
                ),
            )
        )
    prepared_kwargs = dict(kwargs)
    prepared_kwargs["tracks"] = tuple(transformed_tracks)
    return compose_annotated_frames(
        main_rgb=prepared_main, topdown_rgb=topdown, **prepared_kwargs
    )


def encode_profiled_h264_base_video(
    frames: Any,
    output_path: str | Path,
    *,
    profile: ReviewVisualProfile,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
) -> dict[str, Any]:
    """Encode a clean review video at the native profile resolution."""

    array = np.asarray(frames)
    expected_shape = (
        profile.capture_height,
        profile.capture_width,
        3,
    )
    if array.ndim != 4 or tuple(array.shape[1:]) != expected_shape:
        raise M6XVisualProfileError(
            f"clean review frames must have shape [frames,{expected_shape[0]},"
            f"{expected_shape[1]},3], got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise M6XVisualProfileError(
            f"clean review frames must be uint8, got {array.dtype}"
        )
    actual_count = int(array.shape[0])
    if frame_count is not None:
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 1
            or frame_count != actual_count
        ):
            raise M6XVisualProfileError(
                f"requested frame count {frame_count!r} differs from "
                f"RGB frame count {actual_count}"
            )
    if frame_rate_hz is None:
        frame_rate_hz = profile.capture_frame_rate_hz
    return _encode_h264_video_profile(
        array,
        output_path,
        width=profile.capture_width,
        height=profile.capture_height,
        profile_name=f"M6.x {profile.profile_id}",
        frame_count=actual_count,
        frame_rate_hz=frame_rate_hz,
    )


def mux_profiled_binaural_wav(
    base_video_path: str | Path,
    authoritative_wav_path: str | Path,
    output_path: str | Path,
    *,
    profile: ReviewVisualProfile,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
    sample_rate_hz: int | None = None,
    sample_count: int | None = None,
    audio_channel_count: int | None = 2,
) -> dict[str, Any]:
    """Mux a profiled clean stream with its authoritative audio clock."""

    return _mux_binaural_wav_profile(
        base_video_path,
        authoritative_wav_path,
        output_path,
        expected_width=profile.capture_width,
        expected_height=profile.capture_height,
        profile_name=f"M6.x {profile.profile_id}",
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        sample_rate_hz=sample_rate_hz,
        sample_count=sample_count,
        audio_channel_count=audio_channel_count,
    )


__all__ = [
    "M6XVisualProfileError",
    "ReviewVisualProfile",
    "VISUAL_PROFILE_SCHEMA",
    "apply_runtime_review_profile",
    "aspect_preserving_downscale_no_upscale",
    "compose_profiled_annotated_frames",
    "configure_runtime_review_profile",
    "encode_profiled_h264_base_video",
    "light_setup_records",
    "letterbox_marker_coordinates",
    "load_review_visual_profile",
    "mux_profiled_binaural_wav",
    "validate_profile_capture_request",
    "validate_realized_review_profile",
    "validate_runtime_review_light_readback",
]
