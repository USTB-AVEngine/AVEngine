#!/usr/bin/env python3
"""Render a temporary overhead RGB reference for R8."""

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
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

SCENE = Path(
    "/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.glb"
)
OUT = Path("/data/smy/room_split_experiments/00006_20260907_v9/R8_overhead_rgb.png")
GEOM = OUT.parent / "R8_geometry_overlay.png"
COMPOSITE = OUT.parent / "R8_geometry_and_real_rgb.png"


def look_at(direction, np_quaternion):
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    import math

    yaw = math.atan2(-d[0], -d[2])
    pitch = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    qy = np_quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    qx = np_quaternion(math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0)
    return qy * qx


def main():
    rt = prepare_installed_habitat_runtime(
        runtime_prefix=RUNTIME_PREFIX,
        magnum_python_site=MAGNUM_SITE,
        rlr_sdk_root=RLR_SDK_ROOT,
        mp3d_root=MP3D_ROOT,
        allow_mp3d_environment=False,
    )
    hs = rt.habitat_sim
    config = hs.SimulatorConfiguration()
    config.scene_id = str(SCENE)
    config.load_semantic_mesh = False
    config.enable_physics = True
    if rt.physics_config_path:
        config.physics_config_file = str(rt.physics_config_path)
    sensor = hs.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = hs.SensorType.COLOR
    sensor.resolution = [900, 900]
    sensor.hfov = 75.0
    sensor.position = [0.0, 0.0, 0.0]
    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    sim = hs.Simulator(hs.Configuration(config, [agent_cfg]))
    agent = sim.get_agent(0)
    state = agent.get_state()
    # Center of the original R8 bbox, viewed from above.
    state.position = np.asarray(
        [(-4.897 + 0.775) / 2, 10.0, (-5.236 + 4.707) / 2], dtype=np.float32
    )
    state.rotation = look_at([0.0, -1.0, 0.0], rt.quaternion.quaternion)
    state.sensor_states = {}
    agent.set_state(state, True)
    rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
    real = Image.fromarray(rgb)
    real.save(OUT)
    if GEOM.is_file():
        geom = Image.open(GEOM).convert("RGB")
        target_w = geom.width
        real_small = real.resize(
            (target_w, int(real.height * target_w / real.width)),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new(
            "RGB", (target_w, geom.height + real_small.height + 55), "white"
        )
        canvas.paste(geom, (0, 0))
        canvas.paste(real_small, (0, geom.height + 55))
        ImageDraw.Draw(canvas).text(
            (12, geom.height + 15),
            "下面：同一场景的实际 Habitat 俯视 RGB 渲染",
            fill="black",
        )
        canvas.save(COMPOSITE)
    sim.close()
    print(OUT)


if __name__ == "__main__":
    main()
