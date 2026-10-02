#!/usr/bin/env python
"""直接用 habitat 加载 apartment 的 scene.glb 渲染(绕开 AVEngine 溯源)。"""
import os, sys, math, numpy as np, quaternion, imageio.v2 as imageio
import habitat_sim

PKG = "/data/datasets/avengine_workspaces/users/smy/apartment/package"
DATASET = os.path.join(PKG, "visual", "legacy_apartment_0000.scene_dataset_config.json")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

print("[1] 配置模拟器: 加载 apartment 场景数据集", flush=True)
bk = habitat_sim.SimulatorConfiguration()
bk.scene_dataset_config_file = DATASET
bk.scene_id = "legacy_apartment_0000"
bk.enable_physics = False

cam = habitat_sim.CameraSensorSpec()
cam.uuid = "rgb"; cam.sensor_type = habitat_sim.SensorType.COLOR
cam.resolution = [512, 768]; cam.hfov = 90; cam.position = [0.0, 0.0, 0.0]
ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [cam]

print("[2] 启动模拟器(首次加载 65MB mesh,稍等)...", flush=True)
sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))
agent = sim.get_agent(0)
print("    场景包围盒:", sim.get_active_scene_graph().get_root_node().cumulative_bb, flush=True)

CAM = np.array([-0.7, 1.471, 0.65])   # jzy 选定的机位

print("[3] 渲染 jzy 选定视角(view0)", flush=True)
st = habitat_sim.AgentState(); st.position = CAM
st.rotation = quaternion.quaternion(0.7071067811865476, 0.0, 0.7071067811865475, 0.0)  # w,x,y,z
agent.set_state(st)
img0 = sim.get_sensor_observations()["rgb"][:, :, :3]
imageio.imwrite(os.path.join(OUT, "view0_jzy_pose.png"), np.ascontiguousarray(img0))
print("    view0 亮度=%.1f"%img0.mean(), flush=True)

print("[4] 从该机位做 360° 环视,渲染 90 帧 + 合成 mp4", flush=True)
frames=[]; q_pitch = quaternion.from_rotation_vector([math.radians(-6),0,0])
for i in range(90):
    th = 2*math.pi*i/90
    st = habitat_sim.AgentState(); st.position = CAM
    st.rotation = quaternion.from_rotation_vector([0,th,0]) * q_pitch
    agent.set_state(st)
    frames.append(np.ascontiguousarray(sim.get_sensor_observations()["rgb"][:, :, :3]))
    if i % 20 == 0: print("    帧 %d/90"%i, flush=True)
mp4 = os.path.join(OUT, "apartment_orbit.mp4")
imageio.mimwrite(mp4, frames, fps=15, quality=8, macro_block_size=1)
for k in (0,22,45,67): imageio.imwrite(os.path.join(OUT,f"frame_{k:03d}.png"), frames[k])
print("[5] 完成: mp4=%s (%dB), 平均亮度=%.1f"%(mp4, os.path.getsize(mp4), np.mean([f.mean() for f in frames])), flush=True)
sim.close()
