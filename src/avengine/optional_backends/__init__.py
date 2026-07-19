"""Pure planning boundaries for non-authoritative optional backends."""

from avengine.optional_backends.spear_apartment import (
    NATIVE_APARTMENT_MAP,
    SpearApartmentError,
    build_native_apartment_scenario,
    build_native_apartment_suite,
)
from avengine.optional_backends.spear_visual import (
    BACKEND_ROLE,
    PLAN_SCHEMA,
    SpearActorBinding,
    SpearVisualPlanError,
    actor_ue_yaw_degrees,
    build_spear_visual_plan,
    build_spear_visual_plan_from_files,
    camera_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)

__all__ = [
    "BACKEND_ROLE",
    "NATIVE_APARTMENT_MAP",
    "PLAN_SCHEMA",
    "SpearApartmentError",
    "SpearActorBinding",
    "SpearVisualPlanError",
    "actor_ue_yaw_degrees",
    "build_spear_visual_plan",
    "build_spear_visual_plan_from_files",
    "build_native_apartment_scenario",
    "build_native_apartment_suite",
    "camera_ue_yaw_degrees",
    "habitat_point_to_apartment_ue_cm",
]
