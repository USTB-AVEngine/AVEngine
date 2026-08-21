"""Selected lighting helpers for AVEngine's external SPEAR/UE runtime."""

# Adapted from the Eastforward/spear fork.
# Original path: examples/render_in_gpurir_room.py.
# Fork behavior origin: a5168b8c357afa494f6200dedb03b93c3a59be57.
# Selected bytes are carried by local MIT transition snapshot 251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7
# (SPEAR-lead-b); that local carrier is not represented as a public fork ref.
# AVEngine modifications: retain only the two runner-facing RPC helpers;
# omit room construction, assets, CLI, render setup, and all other examples.
# License: MIT; see LICENSES/SPEAR-MIT.txt.

from __future__ import annotations

__all__ = ["spawn_directional_light", "spawn_sky"]


def spawn_directional_light(game, *, yaw_deg, pitch_deg, intensity_lux):
    light = game.unreal_service.spawn_actor(
        uclass="ADirectionalLight",
        location={"X": 0.0, "Y": 0.0, "Z": 500.0},
    )
    root = light.K2_GetRootComponent()
    root.SetMobility(NewMobility="Movable")
    light.K2_SetActorLocationAndRotation(
        NewLocation={"X": 0.0, "Y": 0.0, "Z": 500.0},
        NewRotation={"Roll": 0.0, "Pitch": float(pitch_deg), "Yaw": float(yaw_deg)},
        bSweep=False,
        bTeleport=True,
    )
    comp = game.unreal_service.get_component_by_class(
        actor=light, uclass="UDirectionalLightComponent"
    )
    comp.SetIntensity(NewIntensity=float(intensity_lux))
    return light


def spawn_sky(game):
    """Spawn native SkyAtmosphere + SkyLight for outside-the-window ambient.

    We deliberately do NOT try to spawn BP_LightStudio because that BP is not
    packaged in the standalone build (only its /Game/... referrer is). Native
    ASkyAtmosphere + ASkyLight compile into every UE build without asset deps.
    """
    actors = {}
    for uclass in ("ASkyAtmosphere", "ASkyLight", "AExponentialHeightFog"):
        try:
            actor = game.unreal_service.spawn_actor(
                uclass=uclass,
                location={"X": 0.0, "Y": 0.0, "Z": 100.0},
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            actors[uclass] = actor
        except Exception as e:
            print(f"[gpurir-room] spawn_sky: skip {uclass} ({e})", flush=True)
    return actors
