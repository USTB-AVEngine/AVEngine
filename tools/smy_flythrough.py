#!/usr/bin/env python
"""最小纯视觉 demo:相机在 blender_custom 房间内 360° 环视,渲染 RGB 逐帧并合成 mp4。"""
import os, sys, math, numpy as np, quaternion, imageio.v2 as imageio
import habitat_sim

ROOM = "examples/m1/rooms/blender_custom/visual"
DATASET = os.path.join(ROOM, "m1_custom_room.scene_dataset_config.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/datasets/avengine_workspaces/users/smy/outputs/flythrough"
os.makedirs(OUT, exist_ok=True)
N = 90                      # 帧数
CAM = np.array([-1.5, 1.5, 0.0])   # 室内机位
PITCH = math.radians(-8)   # 略微俯视

bk = habitat_sim.SimulatorConfiguration()
bk.scene_dataset_config_file = DATASET
bk.scene_id = "m1_custom_room"
bk.enable_physics = False

cam = habitat_sim.CameraSensorSpec()
cam.uuid = "rgb"; cam.sensor_type = habitat_sim.SensorType.COLOR
cam.resolution = [480, 640]; cam.hfov = 90; cam.position = [0.0, 0.0, 0.0]

ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [cam]
sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))
agent = sim.get_agent(0)

frames = []
q_pitch = quaternion.from_rotation_vector([PITCH, 0, 0])
for i in range(N):
    theta = 2 * math.pi * i / N
    st = habitat_sim.AgentState()
    st.position = CAM
    st.rotation = quaternion.from_rotation_vector([0, theta, 0]) * q_pitch
    agent.set_state(st)
    rgb = sim.get_sensor_observations()["rgb"][:, :, :3]   # RGBA->RGB
    frames.append(np.ascontiguousarray(rgb))

mp4 = os.path.join(OUT, "flythrough.mp4")
imageio.mimwrite(mp4, frames, fps=15, quality=8, macro_block_size=1)
# 也存几张关键帧 png 便于快速看
for k in (0, N // 4, N // 2, 3 * N // 4):
    imageio.imwrite(os.path.join(OUT, f"frame_{k:03d}.png"), frames[k])
b = np.mean([f.mean() for f in frames])
print(f"OK frames={len(frames)} 平均亮度={b:.1f} mp4={mp4} 大小={os.path.getsize(mp4)}B")
sim.close()
