from types import SimpleNamespace
import json
import numpy as np

from avengine.assets import qualification_geometry as geometry
from avengine.assets import actions


def test_envelope_encloses_all_allowed_poses_and_cache_keeps_asset_identity(tmp_path, monkeypatch):
    visual = tmp_path / "visual.glb"; visual.write_bytes(b"test")
    mapping = tmp_path / "joint_mapping.json"
    mapping.write_text(json.dumps({"runtime_joint_order": ["joint"], "actor_from_skin_root": np.eye(4).tolist()}))
    poses = tmp_path / "actions.npz"; poses.write_bytes(b"test")
    package = {"joint_mapping": str(mapping), "actions_npz": str(poses),
               "package_root": str(tmp_path), "visual_glb": str(visual)}
    monkeypatch.setattr(geometry, "discover_articulated_package", lambda _: package)
    visited = []
    tool = SimpleNamespace(
        load_glb=lambda _: SimpleNamespace(sha256="fixture"),
        _geometry=lambda _: (None, None, None, None, None, None, None),
        _matrix_from_mapping=lambda value: np.asarray(value),
        _pose_joint_matrices=lambda mapping, pose: {"height": float(pose[0, 0])},
    )
    def skin(**kwargs):
        height = kwargs["joint_matrices"]["height"]
        visited.append(height)
        return np.array([[-1., 0., -2.], [1., height, 2.]])
    tool._skin_actor_vertices = skin
    monkeypatch.setattr(geometry, "_grounding_module", lambda: tool)
    idle = SimpleNamespace(semantic_action_id="idle", rotations_xyzw=np.array([[[1., 0., 0., 1.]]]))
    walk = SimpleNamespace(semantic_action_id="walk", rotations_xyzw=np.array([[[2., 0., 0., 1.]], [[3., 0., 0., 1.]]]))
    monkeypatch.setattr(actions, "read_baked_actions_npz", lambda _: SimpleNamespace(
        runtime_joint_order=("joint",), actions=(idle, walk)))
    geometry._BODY_ENVELOPE_CACHE.clear()
    record = {"asset_id": "first", "timeline": {"idle_action_id": "idle", "walking_action_id": "walk"},
              "runtime_backends": {"habitat": {"glb_path": str(visual)}}}
    result = geometry.measure_registered_body_envelope(record)
    assert result["bounds_min_m"] == [-1., 0., -2.]
    assert result["bounds_max_m"] == [1., 3., 2.]
    assert result["action_frame_counts"] == {"idle": 1, "walk": 2}
    assert len(result["vertices_m"]) == 8
    result["vertices_m"][0][0] = 999
    again = geometry.measure_registered_body_envelope(record)
    assert again["vertices_m"][0][0] == -1
    assert len(visited) == 3
    other = geometry.measure_registered_body_envelope({**record, "asset_id": "second"})
    assert other["asset_id"] == "second"


def test_unregistered_body_is_missing_evidence():
    result = geometry.measure_registered_body_envelope({"asset_id": "missing"})
    assert result["measurement"] == "not_run"
