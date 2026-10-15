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
