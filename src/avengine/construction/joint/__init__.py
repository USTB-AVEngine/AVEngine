"""One shared geometry/cache/route/certification service for family plugins."""
from .types import (ActorPlan, BodySpec, CameraPose, Certificate, CheckResult,
                    FamilyPlugin, FamilyRequest, ScenePlan, SolveResult, SoundEvent)
from .solver import solve

__all__ = ["ActorPlan", "BodySpec", "CameraPose", "Certificate", "CheckResult",
           "FamilyPlugin", "FamilyRequest", "ScenePlan", "SolveResult", "SoundEvent", "solve"]

from .context import RoomContext, CandidateGrid, build_room_context, candidate_cameras
from .geometry import (DepthBuffer, MeshRaycaster, CompositeRaycaster, VisibilityPrediction,
                       build_depth_buffer, framing, predict_visibility, project_points)
from .routes import ConstructionError, construct_route, route_between, static_actor, validate_track
from .layouts import enumerate_static_layouts
from .certification import certify_plan, motion_witness_checks
__all__ += ["RoomContext", "CandidateGrid", "build_room_context", "candidate_cameras",
            "DepthBuffer", "MeshRaycaster", "CompositeRaycaster", "VisibilityPrediction",
            "build_depth_buffer", "framing", "predict_visibility", "project_points",
            "ConstructionError", "construct_route", "route_between", "static_actor", "validate_track",
            "enumerate_static_layouts", "certify_plan", "motion_witness_checks"]

from .mesh_prediction import MeshInstance, raster_mesh_depth, predict_mesh_visibility
__all__ += ["MeshInstance", "raster_mesh_depth", "predict_mesh_visibility"]

from .scheduling import SoundRequirement, ScheduleResult, schedule_sound_events
__all__ += ["SoundRequirement", "ScheduleResult", "schedule_sound_events"]

from .geometry import framing_vertices
from .native import strict_native_bindings, native_world_vertices, native_rest_framing, native_static_checks
__all__ += ["framing_vertices", "strict_native_bindings", "native_world_vertices", "native_rest_framing", "native_static_checks"]

from .types import PreparedCandidate
__all__ += ["PreparedCandidate"]

from .native import PreparedStaticScene, prepare_static_native
__all__ += ["PreparedStaticScene", "prepare_static_native"]

from .routes import bind_distance_clock, straight_then_arc, repeat_bouts
__all__ += ["bind_distance_clock", "straight_then_arc", "repeat_bouts"]

from .native import PreparedNativeScene, native_plan_checks, prepare_native
__all__ += ["PreparedNativeScene", "native_plan_checks", "prepare_native"]

from .entry import EntryEnvelope, tangent_entry_anchors, entry_timing
from .gait import admitted_human_clock_source, complete_human_gait_clock
__all__ += ["EntryEnvelope", "tangent_entry_anchors", "entry_timing", "admitted_human_clock_source", "complete_human_gait_clock"]

from .grounding import floor_contact_variants
__all__ += ["floor_contact_variants"]

from .acoustics import native_acoustic_checks
__all__ += ["native_acoustic_checks"]

from .physics import native_full_mesh_sweep
__all__ += ["native_full_mesh_sweep"]

from .routes import out_and_back, out_and_back_shape
__all__ += ["out_and_back", "out_and_back_shape"]

from .paired import paired_parallel_routes, independent_straight_pair_routes
__all__ += ["paired_parallel_routes", "independent_straight_pair_routes"]

from .certification import track_framing_check
__all__ += ["track_framing_check"]

from .registered import PreparedNativeBundle, prepare_native_bundle, prepare_registered_bundle, swapped_visual_plan
__all__ += ["PreparedNativeBundle", "prepare_native_bundle", "prepare_registered_bundle", "swapped_visual_plan"]

from .static_solve import enumerate_joint_static_layouts
from .registered import registered_row_for_plan
__all__ += ["enumerate_joint_static_layouts", "registered_row_for_plan"]
