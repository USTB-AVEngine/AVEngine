"""Render R1 and show the two different annotation measurements."""

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
import sys, json, csv, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_r8_overhead_rgb import prepare_installed_habitat_runtime, look_at
from avengine.acoustics.gltf import (
    extract_triangle_scene_document,
    load_glb_bytes,
    triangle_vertex_colours,
)
from avengine.acoustics.semantic import _linear_to_srgb_bytes


def main():
    root = Path(
        "/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/train/00250-U3oQjwTuMX8"
    )
    out = Path("/data/smy/room_split_experiments/00250_R1_explanation")
    out.mkdir(exist_ok=True)
    rooms_path = Path(
        "/data/avengine_external/studio/tasks/20260831T133631Z-hm3d_end_to_end/output/render/rooms/hm3d_train_00250_U3oQjwTuMX8/rooms.json"
    )
    room = next(
        r for r in json.loads(rooms_path.read_text())["rooms"] if r["region_id"] == 1
    )
    mapping = {}
    for line in (root / "U3oQjwTuMX8.semantic.txt").read_text().splitlines()[1:]:
        row = line.split(",")
        if len(row) >= 4 and row[0].strip().isdigit():
            mapping[tuple(bytes.fromhex(row[1].strip()))] = (
                row[2].strip().strip('"').lower(),
                int(row[3]),
            )
    path = root / "U3oQjwTuMX8.semantic.glb"
    doc = load_glb_bytes(path.read_bytes(), source_path=str(path))
    mesh = extract_triangle_scene_document(doc)
    colors = _linear_to_srgb_bytes(triangle_vertex_colours(doc, mesh)[0])
    floors = []
    counts = {}
    for i, color in enumerate(colors):
        cat, region = mapping.get(tuple(color), ("", -1))
        if region == 1 and cat in ("floor", "carpet", "rug", "flooring"):
            face = mesh.vertices[mesh.triangles[i].astype(int)].astype(float)
            face = np.stack((face[:, 0], face[:, 2], -face[:, 1]), axis=-1)
            area = float(
                np.linalg.norm(np.cross(face[1] - face[0], face[2] - face[0])) / 2
            )
            counts[cat] = counts.get(cat, 0) + area
            floors.append(face)
    rt = prepare_installed_habitat_runtime(
        runtime_prefix=RUNTIME_PREFIX,
        magnum_python_site=MAGNUM_SITE,
        rlr_sdk_root=RLR_SDK_ROOT,
        mp3d_root=MP3D_ROOT,
        allow_mp3d_environment=False,
    )
    hs = rt.habitat_sim
    cfg = hs.SimulatorConfiguration()
    cfg.scene_id = str(root / "U3oQjwTuMX8.glb")
    cfg.load_semantic_mesh = False
    sp = hs.CameraSensorSpec()
    sp.uuid = "rgb"
    sp.sensor_type = hs.SensorType.COLOR
    sp.resolution = [1200, 1200]
    sp.hfov = 75.0
    sp.position = [0.0, 0.0, 0.0]
    ac = hs.agent.AgentConfiguration()
    ac.sensor_specifications = [sp]
    sim = hs.Simulator(hs.Configuration(cfg, [ac]))
    lo, hi = np.array(room["bbox_xz_m"])
    centre = (lo + hi) / 2
    height = float(
        room["floor_y_m"] + max(hi - lo) * 0.65 / math.tan(math.radians(37.5))
    )
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.array([centre[0], height, centre[1]], dtype=np.float32)
    state.rotation = look_at([0.0, -1.0, 0.0], rt.quaternion.quaternion)
    state.sensor_states = {}
    agent.set_state(state, True)
    rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
    Image.fromarray(rgb).save(out / "R1_actual_overhead.png")
    cam = sim._sensors["rgb"]._sensor_object.render_camera
    v = np.array(cam.camera_matrix)
    p = np.array(cam.projection_matrix)

    def project(points):
        c = (p @ v @ np.column_stack((points, np.ones(len(points)))).T).T
        n = c[:, :2] / c[:, 3, None]
        return np.column_stack(((n[:, 0] + 1) * 600, (1 - n[:, 1]) * 600))

    layer = Image.new("RGBA", (1200, 1200))
    d = ImageDraw.Draw(layer)
    for face in floors:
        d.polygon([tuple(x) for x in project(face)], fill=(255, 160, 15, 160))
    image = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"), layer)
    d = ImageDraw.Draw(image)
    y = room["floor_y_m"]
    corners = np.array(
        [[lo[0], y, lo[1]], [hi[0], y, lo[1]], [hi[0], y, hi[1]], [lo[0], y, hi[1]]]
    )
    q = project(corners)
    d.line([tuple(x) for x in np.vstack((q, q[0]))], fill=(0, 210, 255), width=6)
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 25
    )
    for a, b, t in [(q[0], q[1], "10.16 m"), (q[1], q[2], "10.36 m")]:
        pos = (a + b) / 2
        d.rectangle((pos[0] - 65, pos[1] - 18, pos[0] + 70, pos[1] + 20), fill="white")
        d.text((pos[0] - 60, pos[1] - 16), t, font=font, fill="black")
    d.rectangle((0, 0, 1200, 112), fill="white")
    for i, t in enumerate(
        [
            "00250 / R1：真实俯视图 + 两种标注范围",
            "青框：全部 R1 标注面的外接矩形 10.16 × 10.36 m ≈ 105.26 m²",
            "橙色：计入面积的地板/地毯三角面，共 %.2f m²（不是导航面积）"
            % sum(counts.values()),
        ]
    ):
        d.text((12, 5 + i * 35), t, font=font, fill="black")
    image.convert("RGB").save(out / "R1_measurements_explained.png")
    evidence = {
        "room": room,
        "recomputed_floor_categories_m2": counts,
        "floor_triangle_count": len(floors),
        "rooms_source": str(rooms_path),
        "view": v.tolist(),
        "projection": p.tolist(),
    }
    (out / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2)
    )
    print(out, counts)
    sim.close()


if __name__ == "__main__":
    main()
