"""Single-region audit: exact orthographic rendering, nav clipping, video evidence."""

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
import json, sys, math, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_r8_overhead_rgb import prepare_installed_habitat_runtime, look_at, SCENE

OUT = Path("/data/smy/room_review_overheads/R8_coordinate_check")
OLD = Path("/data/smy/room_split_experiments/00006_20260907_v9")


def clip(poly, axis, bound, greater):
    result = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        ina = a[axis] >= bound if greater else a[axis] <= bound
        inb = b[axis] >= bound if greater else b[axis] <= bound
        if ina:
            result.append(a)
        if ina != inb:
            result.append(a + (b - a) * (bound - a[axis]) / (b[axis] - a[axis]))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    shutil.copy2(OLD / "R8_overhead_rgb.png", OUT / "01_original_perspective.png")
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
    spec.sensor_subtype = hs.SensorSubType.ORTHOGRAPHIC
    spec.resolution = [1024, 1024]
    spec.position = [0.0, 0.0, 0.0]
    spec.ortho_scale = 1.0 / 12
    spec.near = 28.2
    spec.far = 31.0
    ac = hs.agent.AgentConfiguration()
    ac.sensor_specifications = [spec]
    sim = hs.Simulator(hs.Configuration(cfg, [ac]))
    agent = sim.get_agent(0)
    floor = 2.836
    lo = np.array([-4.897, -5.236])
    hi = np.array([0.775, 4.707])
    centre = (lo + hi) / 2
    st = agent.get_state()
    st.position = np.array([centre[0], floor + 30, centre[1]], dtype=np.float32)
    st.rotation = look_at([0.0, -1.0, 0.0], rt.quaternion.quaternion)
    st.sensor_states = {}
    agent.set_state(st, True)
    rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
    Image.fromarray(rgb).save(OUT / "02_strict_orthographic.png")
    cam = sim._sensors["rgb"]._sensor_object.render_camera
    V = np.array(cam.camera_matrix, dtype=float)
    P = np.array(cam.projection_matrix, dtype=float)
    T = np.linalg.inv(V)

    def project(points):
        c = (P @ V @ np.column_stack((points, np.ones(len(points)))).T).T
        return np.column_stack(
            ((c[:, 0] / c[:, 3] + 1) * 512, (1 - c[:, 1] / c[:, 3]) * 512)
        )

    pf = sim.pathfinder
    assert pf.load_nav_mesh(str(SCENE.with_suffix(".basis.navmesh")))
    vertices = np.asarray(pf.build_navmesh_vertices(), dtype=float)
    indices = np.asarray(pf.build_navmesh_vertex_indices(), dtype=int).reshape(-1, 3)

    def domain(box, band=0.2):
        polys = []
        for ids in indices:
            poly = list(vertices[ids])
            for axis, bound, greater in [
                (0, box[0][0], True),
                (0, box[1][0], False),
                (2, box[0][1], True),
                (2, box[1][1], False),
                (1, floor - band, True),
                (1, floor + band, False),
            ]:
                if poly:
                    poly = clip(poly, axis, bound, greater)
            if len(poly) >= 3:
                polys.append(np.array(poly))
        a3 = 0.0
        a2 = 0.0
        for poly in polys:
            for i in range(1, len(poly) - 1):
                cross = np.cross(poly[i] - poly[0], poly[i + 1] - poly[0])
                a3 += np.linalg.norm(cross) / 2
                a2 += abs(cross[1]) / 2
        return polys, {
            "nav_surface_area_m2": float(a3),
            "nav_xz_projected_triangle_sum_m2": float(a2),
            "height_band_m": [floor - band, floor + band],
            "clipped_polygon_count": len(polys),
        }

    records = {}
    metas = list(OLD.glob("videos/**/*R8*.camera.json"))
    for file in metas:
        e = json.loads(file.read_text())
        box = np.array(e["room_bbox_xz_m"])
        _, area = domain(box)
        records[e["room_label"]] = {
            "metadata_source": str(file),
            "bbox_xz_m": box.tolist(),
            "length_xz_m": (box[1] - box[0]).tolist(),
            "bbox_area_m2": float(np.prod(box[1] - box[0])),
            **area,
            "camera_eye_xyz_m": e["camera_eye_xyz_m"],
            "trajectory_eye_xyz_m": [e["camera_eye_xyz_m"]] * e["frames"],
            "trajectory_note": "原地环视，位置固定；逐帧朝向由渲染程序生成，元数据未保存每帧旋转。",
            "bbox_is_visible_footprint": False,
        }
    polys, area = domain([lo, hi])
    records["R8_original"] = {
        "bbox_xz_m": [lo.tolist(), hi.tolist()],
        "length_xz_m": (hi - lo).tolist(),
        "bbox_area_m2": float(np.prod(hi - lo)),
        **area,
        "camera_eye_xyz_m": None,
        "trajectory_eye_xyz_m": None,
        "missing_reason": "正式 R8.mp4 无 camera.json；现有批处理日志只保留条目编号、耗时与输出路径，不能还原精确机位。",
        "semantic_floor_area_m2": 28.55,
        "bbox_is_visible_footprint": False,
    }
    layer = Image.new("RGBA", (1024, 1024))
    d = ImageDraw.Draw(layer)
    for poly in polys:
        d.polygon([tuple(p) for p in project(poly)], fill=(20, 180, 115, 100))
    image = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"), layer)
    d = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 19
    )
    colours = {
        "R8_original": (255, 200, 20),
        "apartment_R8_01": (20, 225, 255),
        "apartment_R8_02": (235, 90, 240),
        "R8_green_center": (20, 225, 255),
    }
    for name, r in records.items():
        a, b = np.array(r["bbox_xz_m"])
        corners = np.array(
            [
                [a[0], floor, a[1]],
                [b[0], floor, a[1]],
                [b[0], floor, b[1]],
                [a[0], floor, b[1]],
                [a[0], floor, a[1]],
            ]
        )
        px = project(corners)
        d.line([tuple(v) for v in px], fill=colours[name], width=3)
        if r["camera_eye_xyz_m"]:
            # Ground footprint of camera, explicitly not projecting its elevated eye.
            point = np.array(r["camera_eye_xyz_m"])
            point[1] = floor
            x, y = project([point])[0]
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colours[name])
            d.text((x + 8, y), name.replace("apartment_", ""), font=font, fill="white")
    # 2 m horizontal scale from projected world points, independent of image resizing.
    pp = project([[centre[0] - 1, floor, centre[1]], [centre[0] + 1, floor, centre[1]]])
    scale = float(np.linalg.norm(pp[1] - pp[0]))
    d.rectangle((30, 950, 30 + scale, 956), fill="white")
    d.text((30, 920), "2 m", font=font, fill="white")
    d.rectangle((0, 0, 1024, 83), fill="white")
    d.text(
        (12, 5),
        "00006 / R8：真实正交图 + 世界坐标验证（同旧示例朝向）",
        font=font,
        fill="black",
    )
    d.text(
        (12, 31),
        "绿：同层导航；黄：原 R8 包围盒；青/紫：历史视频候选框（不作为新切分）",
        font=font,
        fill="black",
    )
    d.text(
        (12, 57),
        "地面投影可能覆盖上方家具；图像本身没有移位、拉伸或镜像校准。",
        font=font,
        fill="black",
    )
    image.convert("RGB").save(OUT / "03_navigation_bounds_scale.png")
    # Validate orthographic equal scale at different heights and camera-axis alignment.
    scales = []
    for y in [floor, floor + 1.0, floor + 1.8]:
        px = project(
            [
                [centre[0], y, centre[1]],
                [centre[0] + 1, y, centre[1]],
                [centre[0], y, centre[1] + 1],
            ]
        )
        scales.append(
            [float(np.linalg.norm(px[1] - px[0])), float(np.linalg.norm(px[2] - px[0]))]
        )
    forward = T[:3, :3] @ np.array([0, 0, -1])
    assert np.linalg.norm(forward - np.array([0, -1, 0])) < 1e-5
    assert np.max(np.abs(np.array(scales) - 1024 / 12)) < 0.001
    evidence = {
        "source_glb": str(SCENE),
        "navmesh_source": str(SCENE.with_suffix(".basis.navmesh")),
        "navmesh_provenance": "此前从原始 train GLB 自产的导航文件；不是下载的官方 navmesh。",
        "camera_position_xyz_m": T[:3, 3].tolist(),
        "rotation_camera_to_world": T[:3, :3].tolist(),
        "forward_world": forward.tolist(),
        "view_matrix": V.tolist(),
        "projection_matrix": P.tolist(),
        "camera_height_above_floor_m": 30,
        "projection_type": "orthographic",
        "hfov": "不适用：正交投影以 12 m 视宽控制范围",
        "resolution_px": [1024, 1024],
        "image_world_bounds_xz_m": [(centre - 6).tolist(), (centre + 6).tolist()],
        "pixels_per_m_at_three_heights": scales,
        "near_far_m": [28.2, 31.0],
        "visible_height_band_m": [floor - 1, floor + 1.8],
        "ranges": records,
        "transform": "clip=P*V*[x,y,z,1]; pixel u=(clip.x/clip.w+1)*512; v=(1-clip.y/clip.w)*512",
        "nav_area_definition": "按世界坐标裁剪原导航三角形至 bbox 与地面±0.2m，累加3D三角面面积及XZ投影面积；未使用像素计数，未宣称是语义房间或可见范围。",
    }
    (OUT / "coordinates.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "camera": evidence["camera_position_xyz_m"],
                "scales": scales,
                "ranges": {
                    k: {
                        kk: vv
                        for kk, vv in v.items()
                        if kk
                        in [
                            "length_xz_m",
                            "bbox_area_m2",
                            "nav_surface_area_m2",
                            "nav_xz_projected_triangle_sum_m2",
                        ]
                    }
                    for k, v in records.items()
                },
            },
            ensure_ascii=False,
        )
    )
    sim.close()


if __name__ == "__main__":
    main()
