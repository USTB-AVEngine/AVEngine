"""Project actual navmesh triangles with the RGB render camera matrices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.rooms.runtime_config import (
    RUNTIME_PREFIX,
    MAGNUM_SITE,
    RLR_SDK_ROOT,
    MP3D_ROOT,
    TASKS_ROOT,
    MEDIA_ROOT,
    ROOM_PYTHON,
)
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_r8_overhead_rgb import (
    prepare_installed_habitat_runtime,
    look_at,
    SCENE,
    OUT,
)


def main():
    all_nav = "--all-nav" in sys.argv
    rt = prepare_installed_habitat_runtime(
        runtime_prefix=RUNTIME_PREFIX,
        magnum_python_site=MAGNUM_SITE,
        rlr_sdk_root=RLR_SDK_ROOT,
        mp3d_root=MP3D_ROOT,
        allow_mp3d_environment=False,
    )
    hs = rt.habitat_sim
    cfg = hs.SimulatorConfiguration()
    cfg.scene_id = str(SCENE)
    cfg.load_semantic_mesh = False
    spec = hs.CameraSensorSpec()
    spec.uuid = "rgb"
    spec.sensor_type = hs.SensorType.COLOR
    spec.resolution = [900, 900]
    spec.hfov = 75.0
    spec.position = [0.0, 0.0, 0.0]
    ac = hs.agent.AgentConfiguration()
    ac.sensor_specifications = [spec]
    sim = hs.Simulator(hs.Configuration(cfg, [ac]))
    assert sim.pathfinder.load_nav_mesh(str(SCENE.with_suffix(".basis.navmesh")))
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.array([-2.061, 10, -0.2645], dtype=np.float32)
    state.rotation = look_at([0.0, -1.0, 0.0], rt.quaternion.quaternion)
    state.sensor_states = {}
    agent.set_state(state, True)
    rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
    camera = sim._sensors["rgb"]._sensor_object.render_camera
    view = np.array(camera.camera_matrix)
    projection = np.array(camera.projection_matrix)
    verts = np.asarray(sim.pathfinder.build_navmesh_vertices(), dtype=float)
    indices = np.asarray(
        sim.pathfinder.build_navmesh_vertex_indices(), dtype=int
    ).reshape(-1, 3)
    print("matrices", view, projection, "triangles", len(indices), flush=True)

    def project(points):
        clip = (projection @ view @ np.column_stack((points, np.ones(len(points)))).T).T
        ndc = clip[:, :3] / clip[:, 3, None]
        return np.column_stack(((ndc[:, 0] + 1) * 450, (1 - ndc[:, 1]) * 450))

    # Clip triangle polygons against R8 bbox and a narrow same-floor band.
    def clip_poly(poly, axis, bound, greater):
        out = []
        for a, b in zip(poly, poly[1:] + poly[:1]):
            ina = (a[axis] >= bound) if greater else (a[axis] <= bound)
            inb = (b[axis] >= bound) if greater else (b[axis] <= bound)
            if ina:
                out.append(a)
            if ina != inb:
                out.append(a + (b - a) * ((bound - a[axis]) / (b[axis] - a[axis])))
        return out

    layer = Image.new("RGBA", (900, 900))
    draw = ImageDraw.Draw(layer)
    count = 0
    for idx in indices:
        tri = verts[idx]
        if not all_nav and np.any(abs(tri[:, 1] - 2.836) > 0.20):
            continue
        poly = list(tri)
        limits = (
            []
            if all_nav
            else [
                (0, -4.897, True),
                (0, 0.775, False),
                (2, -5.236, True),
                (2, 4.707, False),
            ]
        )
        for axis, bound, greater in limits:
            if poly:
                poly = clip_poly(poly, axis, bound, greater)
        if len(poly) < 3:
            continue
        draw.polygon(
            [tuple(p) for p in project(np.asarray(poly))], fill=(15, 160, 95, 100)
        )
        count += 1
    result = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"), layer)
    d = ImageDraw.Draw(result)
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20
    )
    d.rectangle((0, 0, 900, 65), fill="white")
    d.text(
        (12, 6),
        (
            "整栋原始导航网格：未限制房间、楼层或切分区域"
            if all_nav
            else "R8 导航三角网格叠加：使用同一 RGB 相机的实际投影矩阵"
        ),
        font=font,
        fill="black",
    )
    d.text(
        (12, 34),
        (
            "当前视野内绿色=导航投影（含楼梯/其他楼层，可重叠，不做遮挡剔除）"
            if all_nav
            else "绿色为地面导航区域投影；家具遮挡下的地面也会显示绿色"
        ),
        font=font,
        fill="black",
    )
    stem = "house_original_navmesh_overlay" if all_nav else "R8_aligned_navmesh_overlay"
    dest = OUT.parent / (stem + ".png")
    result.convert("RGB").save(dest)
    old = np.asarray(Image.open(OUT).convert("RGB"))
    evidence = {
        "view_matrix": view.tolist(),
        "projection_matrix": projection.tolist(),
        "triangle_count": count,
        "manual_pixel_offset": 0,
        "floor_band_m": [2.636, 3.036],
        "rgb_mean_abs_difference_from_reference": float(
            np.abs(old.astype(float) - rgb.astype(float)).mean()
        ),
        "note": "Ground projection, not furniture segmentation; no depth occlusion applied.",
    }
    evidence["all_nav"] = all_nav
    evidence["source_navmesh"] = str(SCENE.with_suffix(".basis.navmesh"))
    evidence["floor_band_m"] = None if all_nav else [2.636, 3.036]
    (OUT.parent / (stem + ".json")).write_text(json.dumps(evidence, indent=2))
    if all_nav:
        # Whole-house orthographic overview includes areas beyond the RGB camera view.
        low = verts[:, [0, 2]].min(axis=0)
        high = verts[:, [0, 2]].max(axis=0)
        scale = 900 / max(high - low)
        overview = Image.new(
            "RGB",
            (
                int((high[0] - low[0]) * scale) + 80,
                int((high[1] - low[1]) * scale) + 120,
            ),
            "white",
        )
        od = ImageDraw.Draw(overview)
        for idx in indices:
            pts = [
                (40 + (v[0] - low[0]) * scale, 80 + (v[2] - low[1]) * scale)
                for v in verts[idx]
            ]
            od.polygon(pts, fill=(70, 175, 120), outline=(35, 120, 75))
        od.text(
            (12, 12),
            "整栋导航网格全部三角形（各楼层投影重叠）",
            font=font,
            fill="black",
        )
        overview.save(OUT.parent / "house_original_navmesh_full.png")
    print(dest, evidence["rgb_mean_abs_difference_from_reference"])
    sim.close()


if __name__ == "__main__":
    main()
