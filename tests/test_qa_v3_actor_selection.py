from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))
from qa_v3_actor_selection import _actor_entry, _mesh_package_for


def test_explicit_mesh_is_independent_of_historical_directory(tmp_path):
    record = {"runtime_backends": {"spear_unreal": {
        "skeletal_mesh_path": "/Game/Characters/Violet/Body.Body",
    }}}
    assert _mesh_package_for(record, tmp_path) == "/Game/Characters/Violet/Body"


def test_sibling_lookup_uses_package_directory_and_rejects_ambiguity(tmp_path):
    directory = tmp_path / "Characters" / "Violet"
    directory.mkdir(parents=True)
    for name in ("Body", "Body_Skeleton", "Idle"):
        (directory / f"{name}.uasset").touch()
    record = {"runtime_backends": {"spear_unreal": {
        "idle_animation": "/Game/Characters/Violet/Idle.Idle",
        "skeletal_mesh_path": None,
    }}}
    assert _mesh_package_for(record, tmp_path) == "/Game/Characters/Violet/Body"
    for name in ("Other", "Other_Skeleton"):
        (directory / f"{name}.uasset").touch()
    with pytest.raises(RuntimeError, match="cannot uniquely identify"):
        _mesh_package_for(record, tmp_path)


def test_static_selection_needs_no_skeleton_animation_or_species(tmp_path):
    path = tmp_path / "Objects" / "Speaker.uasset"
    path.parent.mkdir(parents=True)
    path.touch()
    record = {"asset_id": "speaker", "revision": "v1", "entity_class": "rigid_object",
              "runtime_backends": {"spear_unreal": {
                  "static_mesh_binding": "explicit_path",
                  "static_mesh_object_path": "/Game/Objects/Speaker.Speaker",
              }}}
    selected = _actor_entry("source2", "speaker", {"speaker": record}, tmp_path)
    assert selected["entity_class"] == "rigid_object"
    assert selected["physical_authorized_internal_sources"] == {"static_mesh": str(path)}
    assert selected["ue_binding"]["static_mesh_package"] == "/Game/Objects/Speaker"
    assert "idle_object_path" not in selected["ue_binding"]
    path.unlink()
    with pytest.raises(RuntimeError, match="missing physical source"):
        _actor_entry("source2", "speaker", {"speaker": record}, tmp_path)
