"""Pure planning boundaries for non-authoritative optional backends."""

from avengine.optional_backends.interioragent_kujiale import (
    DATASET_ID as INTERIORAGENT_DATASET_ID,
    InteriorAgentPlanError,
    build_kujiale_review_plan,
    load_room_metadata as load_interioragent_room_metadata,
    preview_material_parameters as interioragent_preview_material_parameters,
    usd_meters_to_unreal_cm,
)
from avengine.optional_backends.residential_episode import (
    ResidentialEpisodeError,
    build_residential_source_episode,
    classify_object_bounds,
    dataset_z_up_to_habitat,
    point_in_polygon_xy,
)
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
    "INTERIORAGENT_DATASET_ID",
    "InteriorAgentPlanError",
    "NATIVE_APARTMENT_MAP",
    "PLAN_SCHEMA",
    "ResidentialEpisodeError",
    "SpearApartmentError",
    "SpearActorBinding",
    "SpearVisualPlanError",
    "actor_ue_yaw_degrees",
    "build_spear_visual_plan",
    "build_spear_visual_plan_from_files",
    "build_native_apartment_scenario",
    "build_native_apartment_suite",
    "build_kujiale_review_plan",
    "build_residential_source_episode",
    "camera_ue_yaw_degrees",
    "classify_object_bounds",
    "dataset_z_up_to_habitat",
    "habitat_point_to_apartment_ue_cm",
    "interioragent_preview_material_parameters",
    "load_interioragent_room_metadata",
    "point_in_polygon_xy",
    "usd_meters_to_unreal_cm",
]
