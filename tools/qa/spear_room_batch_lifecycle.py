#!/usr/bin/env python3
"""SPEAR same-room Episode lifecycle gates staged for a two-Episode canary.

The functions use only APIs already present in the current Apartment runner
and SPEAR examples.  They have not been runtime-qualified for segmentation
terminate/reinitialize on the production package; the two-Episode canary is a
hard prerequisite before a ten-Episode request may execute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

RESET_SETTLE_FRAMES = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _handle(value: Any) -> int:
    raw = getattr(value, "uobject", getattr(value, "_uobject", None))
    _require(raw is not None and not isinstance(raw, bool), "Unreal handle is missing")
    return int(raw)


def _actor_handles(game: Any) -> set[int]:
    return {int(actor) for actor in game.unreal_service.find_actors(as_handle=True)}


def _clear_proxy_filters(manager: Any) -> None:
    manager.SetAllowedActors(AllowedActors=[])
    manager.SetAllowedComponents(AllowedComponents=[])
    manager.SetIgnoredActors(IgnoredActors=[])
    manager.SetIgnoredComponents(IgnoredComponents=[])


def _normal_filter_snapshot(*, game: Any, depth_component: Any) -> dict[str, Any]:
    """Read back every mutable filter instead of trusting setter completion."""

    segmentation = game.segmentation_service
    primitive_mode = str(
        depth_component.get_property_value(
            property_name="PrimitiveRenderMode", as_value=True
        )
    )
    snapshot = {
        "allowed_actor_count": len(segmentation.get_allowed_actors()),
        "allowed_component_count": len(segmentation.get_allowed_components()),
        "ignored_actor_count": len(segmentation.get_ignored_actors()),
        "ignored_component_count": len(segmentation.get_ignored_components()),
        "depth_primitive_render_mode": primitive_mode,
        "depth_show_only_actor_count": len(
            depth_component.get_property_value(
                property_name="ShowOnlyActors", as_value=True
            )
        ),
    }
    _require(
        all(
            snapshot[key] == 0
            for key in (
                "allowed_actor_count",
                "allowed_component_count",
                "ignored_actor_count",
                "ignored_component_count",
                "depth_show_only_actor_count",
            )
        ),
        f"segmentation/show-only filters were not cleared: {snapshot}",
    )
    _require(
        snapshot["depth_primitive_render_mode"].endswith("PRM_RenderScenePrimitives"),
        f"depth component remains show-only: {snapshot}",
    )
    return snapshot


def configure_normal_pass(*, game: Any, depth_component: Any) -> None:
    """Reset every mutable show-only/segmentation filter for a normal pass."""

    manager = game.segmentation_service.proxy_component_manager
    _clear_proxy_filters(manager)
    depth_component.PrimitiveRenderMode = "PRM_RenderScenePrimitives"
    depth_component.ShowOnlyActors = []


def configure_target_only_pass(
    *, game: Any, depth_component: Any, visual_actor: Any
) -> None:
    manager = game.segmentation_service.proxy_component_manager
    _clear_proxy_filters(manager)
    manager.SetAllowedActors(AllowedActors=[visual_actor])
    depth_component.PrimitiveRenderMode = "PRM_UseShowOnlyList"
    depth_component.ShowOnlyActors = [visual_actor]


def begin_episode_segmentation(
    *,
    instance: Any,
    game: Any,
    depth_component: Any,
    prior_stable_names: Sequence[str],
) -> dict[str, Any]:
    """Initialize a fresh proxy manager and reject prior-Episode descriptors."""

    with instance.begin_frame():
        game.segmentation_service.initialize()
        configure_normal_pass(game=game, depth_component=depth_component)
    with instance.end_frame():
        pass
    instance.step(num_frames=RESET_SETTLE_FRAMES)
    with instance.begin_frame():
        descriptors = game.segmentation_service.get_mesh_proxy_geometry_descs(
            include_debug_info=False, as_global=True
        )
        filter_snapshot = _normal_filter_snapshot(
            game=game, depth_component=depth_component
        )
    with instance.end_frame():
        pass
    leaked = [
        descriptor
        for descriptor in descriptors
        if descriptor.get("actorStableName") in set(prior_stable_names)
    ]
    _require(not leaked, "prior Episode segmentation proxy still exists")
    return {
        "status": "pass",
        "segmentation_initialized": True,
        "prior_stable_names_checked": list(prior_stable_names),
        "prior_proxy_descriptor_match_count": len(leaked),
        "prior_stable_names_absent": True,
        "normal_filter_readback": filter_snapshot,
        "proxy_filters_cleared": True,
        "show_only_list_cleared": True,
    }


def teardown_episode(
    *,
    instance: Any,
    game: Any,
    runner: Any,
    runtimes: Mapping[str, Mapping[str, Any]],
    depth_component: Any,
    stable_names: Sequence[str],
) -> dict[str, Any]:
    """Destroy actors/proxies and prove negative existence before RAW_READY.

    A blank segmentation reinitialize is intentional.  It makes the absence
    of the just-destroyed stable names observable in the same Episode receipt,
    including for the final Episode in a batch.
    """

    controlled_handles = {
        _handle(runtime[owner])
        for runtime in runtimes.values()
        for owner in ("visual_actor", "anchor")
    }
    with instance.begin_frame():
        configure_normal_pass(game=game, depth_component=depth_component)
        filter_snapshot = _normal_filter_snapshot(
            game=game, depth_component=depth_component
        )
        game.segmentation_service.terminate()
    with instance.end_frame():
        pass
    runner._destroy_runtime_actors(instance, runtimes)
    instance.step(num_frames=RESET_SETTLE_FRAMES)

    with instance.begin_frame():
        remaining_handles = _actor_handles(game)
        stable_actor_map = game.unreal_service.find_actors_as_dict(
            include_unreal_name=False
        )
    with instance.end_frame():
        pass
    leaked_handles = sorted(controlled_handles & remaining_handles)
    leaked_names = sorted(set(stable_names) & set(stable_actor_map))
    _require(
        not leaked_handles, f"destroyed actor handles remain live: {leaked_handles}"
    )
    _require(
        not leaked_names, f"destroyed stable actor names remain live: {leaked_names}"
    )

    with instance.begin_frame():
        game.segmentation_service.initialize()
        configure_normal_pass(game=game, depth_component=depth_component)
    with instance.end_frame():
        pass
    instance.step(num_frames=RESET_SETTLE_FRAMES)
    with instance.begin_frame():
        descriptors = game.segmentation_service.get_mesh_proxy_geometry_descs(
            include_debug_info=False, as_global=True
        )
    with instance.end_frame():
        pass
    leaked_descriptors = [
        {
            "rawId": item.get("rawId"),
            "actorStableName": item.get("actorStableName"),
            "actorName": item.get("actorName"),
        }
        for item in descriptors
        if item.get("actorStableName") in set(stable_names)
    ]
    _require(not leaked_descriptors, "destroyed segmentation descriptors remain live")
    with instance.begin_frame():
        game.segmentation_service.terminate()
    with instance.end_frame():
        pass
    return {
        "status": "pass",
        "actors_destroyed": True,
        "segmentation_terminated": True,
        "prior_actor_handles_absent": True,
        "prior_stable_actor_names_absent": True,
        "prior_proxy_descriptors_absent": True,
        # Kept as the compact receipt key required by the controller.
        "prior_stable_names_absent": True,
        "proxy_filters_cleared": True,
        "show_only_list_cleared": True,
        "normal_filter_readback_before_terminate": filter_snapshot,
        "controlled_actor_handle_count": len(controlled_handles),
        "remaining_controlled_actor_handle_count": len(leaked_handles),
        "remaining_controlled_stable_name_count": len(leaked_names),
        "remaining_controlled_proxy_descriptor_count": len(leaked_descriptors),
        "reset_settle_frames_after_actor_destroy": RESET_SETTLE_FRAMES,
        "blank_reinitialize_negative_check": True,
    }


def close_shared_camera(
    *, instance: Any, game: Any, camera: Any, components: Mapping[str, Any]
) -> None:
    """Close shared render components and the camera before Instance.close()."""

    with instance.begin_frame():
        pass
    with instance.end_frame():
        for component in components.values():
            component.terminate_sp_funcs()
            component.Terminate()
        game.unreal_service.destroy_actor(actor=camera)
