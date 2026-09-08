"""QA production interfaces reject absent or contradictory native evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.capture.neutral_readback import (
    validate_clock, validate_neutral_readback, write_neutral_readback,
)
from avengine.capture.ue_neutral_readback import (
    neutral_from_ue_readbacks, neutral_m_to_ue_cm, ue_cm_to_neutral_m,
    ue_rotator_to_neutral_basis,
)
from avengine.capture.habitat_neutral_readback import neutral_from_habitat_readbacks
from avengine.rooms.room_package import (
    SCHEMA, package_from_catalog_entry, renderer_for_room, room_package_errors,
    validate_room_package,
)
from avengine.rooms.evidence_contract import validate_evidence_contract


@pytest.fixture
def clock():
    return dict(frame_count=2, frame_rate_hz=2, sample_rate_hz=16000, sample_count=16000,
                time_base_hz=48000, ticks_per_frame=24000)


@pytest.fixture
def ue(clock):
    def series(x):
        return [dict(frame_index=i, location_cm=[x + i * 10, 50, 150],
                     rotation_deg=[7, -9, 31]) for i in range(2)]
    return dict(clock=clock, camera=series(0), actors={"source1": series(100)},
                emitters={"source1": series(110)})


@pytest.fixture
def package():
    return dict(schema=SCHEMA, room_id="room", family="apartment", renderer="ue_spear",
                visual_scene=dict(map_path="/Game/Apartment", uproject="stage.uproject"),
                acoustic_package="acoustic/manifest.json",
                walkable_space=dict(kind="route_bank", path="routes.json"),
                floor_reference="measured_floor.json",
                static_geometry=dict(vertices="vertices.npy", triangles="triangles.npy"),
                semantics=dict(path="room.json"),
                coordinate_frame=dict(linear_unit="centimeter", up_axis="+Z",
                                      handedness="left", world_transform="ue_xyz_cm_to_xzy_m_v1"),
                subrooms=[])


def test_package_requires_measured_floor_and_keeps_legacy_draft_missing(package):
    assert validate_room_package(package)["room_id"] == "room"
    del package["floor_reference"]
    with pytest.raises(ValueError, match="floor_reference"):
        validate_room_package(package)
    old = dict(room_id="A", map_path="/Game/A", manifest="layout.json", backend="spear_unreal")
    draft = package_from_catalog_entry(old)
    assert draft["legacy_catalog_entry"] == old
    assert draft["floor_reference"] is None
    assert "missing floor_reference" in room_package_errors(draft)


@pytest.mark.parametrize("family,renderer", [
    ("apartment", "ue_spear"), ("kujiale", "ue_spear"), ("authored", "ue_spear"),
    ("mp3d", "habitat"), ("hm3d", "habitat")])
def test_production_dispatch_uses_family_renderer(family, renderer):
    assert renderer_for_room(dict(family=family, renderer=renderer)) == renderer
    with pytest.raises(ValueError, match="invalid production"):
        renderer_for_room(dict(family=family, renderer="habitat" if renderer == "ue_spear" else "ue_spear"))


def test_clock_never_repairs_drift(clock):
    assert validate_clock(clock) == clock
    bad = {**clock, "sample_count": 15999}
    with pytest.raises(ValueError, match="sample counts"):
        validate_clock(bad)
    with pytest.raises(ValueError, match="time_base_hz"):
        validate_clock({k: v for k, v in clock.items() if k != "time_base_hz"})


def test_ue_roundtrip_and_catalog_basis():
    from avengine.qa.unified_catalog import _ue_rotator_to_m3_basis
    rng = np.random.default_rng(260906)
    for position, rotation in zip(rng.uniform(-100000, 100000, (50, 3)),
                                  rng.uniform(-179, 179, (50, 3)), strict=True):
        error_cm = np.max(np.abs(np.array(neutral_m_to_ue_cm(ue_cm_to_neutral_m(position))) - position))
        assert error_cm < 0.1  # one millimeter
        expected = _ue_rotator_to_m3_basis(rotation.tolist())
        observed = ue_rotator_to_neutral_basis(rotation)
        for key in expected:
            assert np.allclose(observed[key], expected[key], rtol=0, atol=1e-12)


def test_observed_readback_and_no_clobber(clock, ue, tmp_path):
    plan = {"clock": clock}
    data = neutral_from_ue_readbacks(ue, plan, source_readbacks="native.json")
    assert data["entities"]["source1"][0]["root"] == [1, 1.5, 0.5]
    assert data["entities"]["source1"][0]["moving"] is True
    assert validate_neutral_readback(data, plan=plan)["status"] == "pass"
    write_neutral_readback(tmp_path / "neutral.json", data, plan=plan)
    with pytest.raises(FileExistsError):
        write_neutral_readback(tmp_path / "neutral.json", data, plan=plan)


@pytest.mark.parametrize("damage", ["basis", "frame", "emitter", "coordinate", "clock"])
def test_neutral_rejects_missing_or_inconsistent_values(clock, ue, damage):
    data = neutral_from_ue_readbacks(ue, {"clock": clock}, source_readbacks="native.json")
    if damage == "basis":
        data["camera"][0]["basis"]["right"] = [0, 0, 0]
    elif damage == "frame":
        data["camera"][0]["frame_index"] = 1
    elif damage == "emitter":
        del data["entities"]["source1"][0]["emitter"]
    elif damage == "coordinate":
        data["coordinate_frame"]["up_axis"] = "+Z"
    else:
        data["clock"]["ticks_per_frame"] = 20000
    with pytest.raises(ValueError):
        validate_neutral_readback(data)


def test_habitat_uses_actual_arrays_and_sensor_basis(clock):
    roots = np.tile(np.eye(4), (2, 1, 1, 1))
    roots[1, 0, 0, 3] = 0.1
    emitters = roots[:, :, :3, 3].copy()
    emitters[:, :, 1] = 0.4
    frames = []
    for i in range(2):
        frames.append(dict(frame_index=i, pts_ticks=i * clock["ticks_per_frame"],
            actor_readbacks=[dict(actor_id="dog", asset_id="beagle", source_slot_id="source1",
                source_endpoint_id="mouth", world_from_skin_root=roots[i, 0].tolist(),
                emitter_world_position_m=emitters[i, 0].tolist())],
            modalities={"rgb": {"sensor_uuid": "rgb"}},
            camera_readback={"sensors": {"rgb": {"translation_m": [0, 1.5, 2],
                                                "rotation_xyzw": [0, 0, 0, 1]}}}))
    records = {"render": clock, "frames": frames}
    data = neutral_from_habitat_readbacks(records, roots, emitters, {"clock": clock},
                                          source_readbacks=["native.json", "roots.npy", "emitters.npy"])
    assert validate_neutral_readback(data)["status"] == "pass"
    assert data["camera"][0]["basis"]["forward"] == [0, 0, -1]
    # A yaw change moves an offset skin root while the asset origin stays still.
    actor_from_skin = np.eye(4)
    actor_from_skin[0, 3] = 1.0
    observed_actor = np.tile(np.eye(4), (2, 1, 1))
    observed_actor[1, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    rotated_skin = (observed_actor @ actor_from_skin)[:, None]
    rotated_records = deepcopy(records)
    for i in range(2):
        rotated_records["frames"][i]["actor_readbacks"][0]["world_from_skin_root"] = rotated_skin[i, 0].tolist()
    canonical = neutral_from_habitat_readbacks(
        rotated_records, rotated_skin, emitters, {"clock": clock},
        source_readbacks=["native.json"],
        actor_from_skin_root_by_slot={"source1": actor_from_skin})
    assert all(np.allclose(frame["root"], [0, 0, 0]) and not frame["moving"]
               for frame in canonical["entities"]["source1"])
    bad = emitters.copy()
    bad[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="emitter JSON"):
        neutral_from_habitat_readbacks(records, roots, bad, {"clock": clock},
                                       source_readbacks=["native.json"])


@pytest.fixture
def evidence(tmp_path, clock):
    truth = dict(schema="avengine_qa_pixel_visibility_truth_v1", frame_indices=[0, 1],
                 camera_pose_ids=["f0", "f1"], resolution_hw=[2, 3],
                 per_instance={"source1": dict(semantic_id=1, frames=[
                     dict(frame_index=i, state="visible_clear") for i in range(2)])})
    masks = tmp_path / "native_pixel_masks_depth_authority_v1.npz"
    np.savez_compressed(masks, depth_derived_modal_semantic=np.ones((2, 2, 3), dtype="uint32"),
                        target_only_source1=np.ones((2, 2, 3), dtype="uint8"))
    mixture, stem = tmp_path / "mix.wav", tmp_path / "stem.wav"
    mixture.write_bytes(b"format test only")
    stem.write_bytes(b"format test only")
    docs = {"pixel_visibility_truth.json": truth,
            "appearance_review.json": {"actors": {"source1": {"status": "reviewed", "value": "blue", "frame_refs": [0]}}},
            "actor_occluders.json": {"frame_records": [], "masks_path": str(masks)},
            "research_report.json": {"clock": clock, "mixture_path": str(mixture), "events": [
                {"event_id": "e1", "actor_id": "source1", "output_stem": str(stem), "wet_tail_interval": [0, 300]}]}}
    paths = {"native_pixel_masks_depth_authority_v1.npz": str(masks)}
    for name, value in docs.items():
        path = tmp_path / name
        path.write_text(json.dumps(value))
        paths[name] = str(path)
    return paths


def test_evidence_legal_and_missing_file(evidence, clock):
    assert validate_evidence_contract(evidence, clock=clock)["kind"] == "format_and_consistency_only"
    del evidence["appearance_review.json"]
    with pytest.raises(ValueError, match="appearance_review"):
        validate_evidence_contract(evidence, clock=clock)


def test_evidence_rejects_conflicting_modal_alias(evidence, clock):
    np.savez_compressed(evidence["native_pixel_masks_depth_authority_v1.npz"],
                        depth_derived_modal_semantic=np.ones((2, 2, 3), dtype="uint32"),
                        modal=np.zeros((2, 2, 3), dtype="uint32"),
                        target_only_source1=np.ones((2, 2, 3), dtype="uint8"))
    with pytest.raises(ValueError, match="aliases disagree"):
        validate_evidence_contract(evidence, clock=clock)


def test_controller_selects_native_capture_entrypoint(tmp_path, clock, package):
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("p1_qa_controller", root / "tools/studio/run_qa_episode.py")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    native_room = tmp_path / "room_manifest.json"
    native_room.write_text("{}")
    (plan_dir / "case_manifest.json").write_text(json.dumps({"clock": clock}))
    (plan_dir / "m1_capture_request.json").write_text("{}")
    (plan_dir / "episode_plan.json").write_text(json.dumps({
        "clock": clock, "resources": {"room_manifest": str(native_room)}}))
    request = {"runtime": {"uproject": "stage.uproject", "unreal_editor": "UnrealEditor",
                           "spear_ext_dir": "sdk", "graphics_adapter": 2}}
    (plan_dir / "room_package.json").write_text(json.dumps(package))
    ue_command = controller.capture_command(request, tmp_path)
    assert ue_command[1].endswith("tools/rooms/run_spear_residential_episode.py")
    habitat = {**package, "family": "mp3d", "renderer": "habitat"}
    (plan_dir / "room_package.json").write_text(json.dumps(habitat))
    habitat_command = controller.capture_command(request, tmp_path)
    assert habitat_command[1].endswith("tools/capture/capture_mp3d_multi_actor.py")
    assert "--case-manifest" in habitat_command
    bad_clock = {**clock, "frame_count": 4, "sample_count": 32000}
    (plan_dir / "case_manifest.json").write_text(json.dumps({"clock": bad_clock}))
    with pytest.raises(ValueError, match="materialized Habitat clock differs"):
        controller.capture_command(request, tmp_path)


def test_short_canary_clock_preserves_declared_sample_rounding():
    clock = dict(frame_count=5, frame_rate_hz=15, sample_rate_hz=16000,
                 sample_count=5333, time_base_hz=48000, ticks_per_frame=3200)
    assert validate_clock(clock) == clock
    with pytest.raises(ValueError, match="sample counts"):
        validate_clock({**clock, "sample_count": 5334})


def test_room_path_configuration_overrides_environment_and_keeps_template(monkeypatch):
    from avengine.rooms.room_package import resolve_room_package_paths
    monkeypatch.setenv('AVENGINE_TEST_ROOT', '/environment')
    package={'scene': '${AVENGINE_TEST_ROOT}/scene.glb', 'scene_template': '${AVENGINE_TEST_ROOT}/scene.glb'}
    result=resolve_room_package_paths(package,runtime={'path_bindings':{'AVENGINE_TEST_ROOT':'/request'}})
    assert result['scene']=='/request/scene.glb'
    assert result['scene_template']=='${AVENGINE_TEST_ROOT}/scene.glb'
    with pytest.raises(ValueError,match='AVENGINE_MISSING_TEST_ROOT'):
        resolve_room_package_paths({'scene':'${AVENGINE_MISSING_TEST_ROOT}/scene.glb'})
