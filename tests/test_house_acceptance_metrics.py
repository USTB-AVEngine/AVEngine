"""Analytic decay, truncated tails, FOA handedness and CPU pixel projection."""
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("scipy")
pytestmark = pytest.mark.fast_unit
from avengine.acoustics.house_acceptance_metrics import decay_fit, rir_metrics, foa_direction, distance_attenuation


def exponential_noise(rt60=.8, duration=4., sample_rate=16000):
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return np.random.default_rng(175).normal(size=len(t)) * np.exp(-3 * np.log(10) * t / rt60)


def test_t20_and_t30_recover_analytic_exponential_energy_decay():
    # Pressure is down 60 dB at rt60, with a fixed broadband carrier.
    x = exponential_noise()
    t20, _ = decay_fit(x, 16000, -25)
    t30, _ = decay_fit(x, 16000, -35)
    assert t20["status"] == t30["status"] == "valid"
    assert t20["rt60_s"] == pytest.approx(.8, rel=.025)
    assert t30["rt60_s"] == pytest.approx(.8, rel=.025)


def test_short_unresolved_tail_is_not_certified_by_backward_integral():
    x = exponential_noise(rt60=6., duration=.25)
    result, _ = rir_metrics(np.stack([x, x, x, x]), 16000, 1., .25)
    assert result["tail_truncated"]
    assert result["status"] != "valid"
    assert all(b["t20"]["status"] != "valid" for b in result["octaves"])


def test_silent_rir_produces_no_rt_estimate():
    result, _ = rir_metrics(np.zeros((4, 16000)), 16000, 1., 4.)
    assert result["status"] == "no_energy"
    assert result["rt60_mid_s"] is None


@pytest.mark.parametrize("source,forward", [
    ([1., .2, -2.], [0., 0., -1.]),
    ([-2., -.2, -1.], [0., 0., -1.]),
    ([1., .4, 2.], [1., 0., 0.]),
])
def test_foa_world_direction_survives_camera_rotation(source, forward):
    unit = np.asarray(source) / np.linalg.norm(source)
    w = np.zeros(400); w[100:108] = np.arange(1, 9)
    ir = np.stack([w, np.sqrt(3) * unit[1] * w, np.sqrt(3) * unit[2] * w, np.sqrt(3) * unit[0] * w])
    result = foa_direction(ir, [0., 0., 0.], source, forward)
    assert result["check"]["angle_deg"] < 1e-5
    assert result["check"]["foa_world_direction"] == pytest.approx(unit)
    assert result["check"]["foa_azimuth_left_positive_deg"] == pytest.approx(result["check"]["geometry_azimuth_left_positive_deg"])


def test_left_right_axis_swap_is_detected():
    w = np.zeros(400); w[100:108] = 1.
    # Right-side source X+, accidentally encoded into world Y instead.
    ir = np.stack([w, w, np.zeros_like(w), np.zeros_like(w)])
    result = foa_direction(ir, [0., 0., 0.], [2., 0., 0.], [0., 0., -1.])
    assert result["check"]["angle_deg"] == pytest.approx(90.)


def test_direct_energy_follows_inverse_square_not_total_room_energy():
    d = np.array([1., 2., 4.])
    result = distance_attenuation(d, 7. / d ** 2)
    assert result["status"] == "valid"
    assert result["pressure_exponent"] == pytest.approx(1.)
    assert distance_attenuation([1., 1.1, 1.2], [1., .9, .8])["status"] != "valid"


def test_camera_pixels_are_right_positive_and_up_negative():
    pytest.importorskip("trimesh")
    from avengine.acoustics.house_acceptance_visual import camera_rays, project_point
    _, focal, f, right, up = camera_rays([-.6, 0., -.8])
    listener = np.array([4., 1.5, -3.])
    source = listener + 3*f + right + .2*up
    p = project_point(source, listener, f, right, up, focal)
    assert p == pytest.approx([80 + focal/3, 45 - focal*.2/3])
    assert project_point(listener - f, listener, f, right, up, focal) is None


def test_cpu_marker_is_read_back_from_pixels_with_mesh_occlusion(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    from trimesh.ray.ray_triangle import RayMeshIntersector
    from avengine.acoustics.house_acceptance_visual import render_probe
    mesh = trimesh.creation.box([8., 8., 8.])
    colors = np.full((len(mesh.faces), 3), 150)
    source = np.array([1., 0., -2.])
    unit = source / np.linalg.norm(source)
    d = {"camera_forward": [0.,0.,-1.], "listener_m": [0.,0.,0.],
         "source_m": source.tolist(), "probe": "unit_fixture",
         "check": {"foa_world_direction": unit.tolist()}}
    seen = render_probe(mesh, colors, RayMeshIntersector(mesh), d, tmp_path, 1)
    assert seen["status"] == "pass"
    assert seen["marker_visible_pixels"] >= 3
    assert seen["projected_foa_px"] == pytest.approx(seen["projected_source_px"])
    # Put the marker outside a closed wall: screen position alone is insufficient.
    d["source_m"] = [3.,0.,-6.]
    blocked = render_probe(mesh, colors, RayMeshIntersector(mesh), d, tmp_path, 2)
    assert blocked["status"] == "review"
    assert blocked["marker_visible_pixels"] == 0

def test_zero_surface_diagnostic_preserves_small_real_surfaces():
    from avengine.acoustics.house_acceptance_geometry import valid_native_triangles
    v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[2.,0.,0.],
                  [1e-6,0.,0.],[0.,1e-6,0.],
                  [1e8,0.,0.],[1e8+1,0.,0.],[1e8,1.,0.]])
    faces = np.array([[0,1,2],[0,0,1],[0,1,3],[0,4,5],[6,7,8]])
    # The tiny positive surface survives; only no-surface faces are omitted.
    assert valid_native_triangles(v,faces).tolist() == [True,False,False,True,False]


def test_diagnostic_face_omission_keeps_material_assignment_and_source(tmp_path):
    from dataclasses import dataclass
    from avengine.acoustics.house_acceptance_geometry import diagnostic_scene_without_degenerate_faces
    @dataclass(frozen=True)
    class Source:
        objects: tuple
        material_categories: tuple
        triangle_count_by_material: dict
        manifest_path: Path
    raw = {"object_id":"fixture","vertices":np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
           "triangles":np.array([[0,0,1],[0,1,2]]),"triangle_material_ids":np.array([0,1])}
    source = Source((raw,),("a","b"),{"a":1,"b":1},tmp_path/"source_manifest.json")
    source.manifest_path.write_text("original")
    derived,evidence = diagnostic_scene_without_degenerate_faces(source,tmp_path)
    assert derived.objects[0]["triangles"].tolist() == [[0,1,2]]
    assert derived.objects[0]["triangle_material_ids"].tolist() == [1]
    assert derived.triangle_count_by_material == {"b":1}
    assert len(raw["triangles"]) == 2
    assert source.manifest_path.read_text() == "original"
    assert evidence["omitted_faces"] == 1
    assert not evidence["production_qualified"]

def test_boundary_diagnostic_survives_without_native_rir_or_views(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    from avengine.acoustics import house_acceptance_visual as visual
    mesh = trimesh.creation.box([2.,2.,2.])
    # Remove a square cap: four boundary edges remain after seam welding.
    keep = ~np.all(mesh.vertices[mesh.faces][:,:,2] == 1., axis=1)
    mesh = trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces[keep],process=False)
    monkeypatch.setattr(visual,"load_raw_scan",lambda _: (mesh,np.full((len(mesh.faces),3),150),["fixture"]))
    def rejected_native(*args, **kwargs):
        raise AssertionError("Native backend must not be needed for boundary diagnostics")
    monkeypatch.setattr(visual,"NativeRawRayIntersector",rejected_native)
    result = {"house":"fixture","input":{"scene_directory":str(tmp_path),"scan_id":"fixture","acoustic_rooms":[]},
              "rirs":[],"leakage":[],"foa_checks":[]}
    diagnostic = visual.inspect_boundary_geometry(result,tmp_path)
    assert diagnostic["boundary_edge_count"] == 4
    assert not diagnostic["native_rir_or_rgb_claim"]
    assert not diagnostic["ray_origins_available"]
    assert (tmp_path/"hole_locators.png").exists()
    assert (tmp_path/"raw_boundary_edges.npz").exists()


def test_stronger_early_reflection_cannot_replace_direct_arrival_for_distance_fit():
    # Direct pressure follows 1/r. A fixed stronger echo at +1.6875 ms made
    # the former largest-peak search assign the far probe's echo as direct.
    energies = []
    for distance in [1., 2., 4.]:
        geometric = round(distance / 343 * 16000)
        w = np.zeros(64000)
        w[geometric] = 1. / distance
        w[geometric + 27] = .6
        result, _ = rir_metrics(np.stack([w, w, w, w]), 16000, distance, 4.)
        assert result["direct_peak_s"] == pytest.approx(geometric / 16000)
        assert result["direct_energy"] == pytest.approx(1. / distance ** 2)
        energies.append(result["direct_energy"])
    assert distance_attenuation([1., 2., 4.], energies)["pressure_exponent"] == pytest.approx(1.)


def test_late_first_arrival_is_measured_instead_of_forced_to_geometric_time():
    w = np.zeros(64000)
    w[round(1. / 343 * 16000) + 96] = 1.
    result, _ = rir_metrics(np.stack([w, w, w, w]), 16000, 1., 4.)
    assert result["direct_delay_error_ms"] > 3.
    assert result["direct_energy"] == pytest.approx(1.)


def test_distance_energy_and_foa_share_the_fixed_first_arrival_window():
    energies = []
    for distance in [1., 2., 4.]:
        arrival = round(distance / 343 * 16000)
        w = np.zeros(64000)
        w[arrival] = 1. / distance
        # At 0.5 ms the echo is outside the predeclared FOA direct window,
        # but was included by the former larger distance-energy window.
        w[arrival + 8] = .6
        result, _ = rir_metrics(np.stack([w, w, w, w]), 16000, distance, 4.)
        assert result["direct_gate_samples"] == [arrival, arrival + 8]
        energies.append(result["direct_energy"])
    assert distance_attenuation([1., 2., 4.], energies)["pressure_exponent"] == pytest.approx(1.)
