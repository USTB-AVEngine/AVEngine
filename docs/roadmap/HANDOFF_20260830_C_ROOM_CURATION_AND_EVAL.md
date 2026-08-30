# 交接 · 同学 C:房间初筛(现在)+ 评测基线(排队)(20260830)

> 任务一现在就能干:对每栋房子的候选房间做人工初筛。owner 指定这是你的练手任务:
> **自己看俯视图,自己写一个 360° 环绕渲染工具**去看拿不准的房间,判定盖章落盘。
> 你的章会直接改变量产系统挑房的行为。

## 访问

```
ssh -L 8765:127.0.0.1:8765 <你的服务器别名>
```

浏览器 http://localhost:8765(首页是全量跑批的成片墙)→ 页脚"房间初筛台"(/studio/rooms)。
仓库 `/data/jzy/code/AVEngine-lead-a`(main)。

## 任务一:房间初筛

### 背景一句话

车队在自动跑全部 181 栋 HM3D 房子(约 30 机器小时),每栋登记出十几个房间;机器已按
**面积 [6, 50] m²、短边 ≥2.2 m** 初筛(上限锚定 kujiale 参照房 49.5 m²,下限是 1.5 m
路线带的物理需要),剩约 1,400 间等你逐间判:**✓ 用 / ✗ 不用 / ? 存疑**。
"不用"的典型:大厅、走廊、楼梯间、扫描破洞严重的怪空间。

### 数据在哪(全部路径)

- **聚合入口(建议从这看)**:`GET http://localhost:8765/api/room-curation`
  ——所有入围房间:面积/尺寸/推断房型/里面有什么/俯视图引用/已有的章。
  页面 /studio/rooms 就是它的可视化,盖章按钮就在每行。
- **每栋的登记文件**:`/data/avengine_external/studio/tasks/<任务id>/output/render/rooms/<house>/`
  - `rooms.json`:每间房 `region_id`、`bbox_xz_m`(habitat 坐标,直接用)、
    `floor_area_m2`、`extent_m`、`floor_y_m`、`top_categories`
  - `connectivity_topdown_y*.png`:分楼层俯视图,房间框上标着 R 编号
- **场景本体**:`/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/<split>/<编号>-<id>/`
  - `<id>.glb` —— **渲染用这个**
  - `<id>.basis.navmesh`、`<id>.semantic.glb`、`<id>.semantic.txt`(初筛用不到)
- **章落盘**:`/data/avengine_external/studio/room_curation/<house>__R<编号>.json`。
  页面点按钮即写;脚本批量盖章用:

```bash
curl -X POST http://localhost:8765/api/room-curation/verdict \
  -H 'Content-Type: application/json' \
  -d '{"house":"hm3d_val_00800_TEEsavR23oF","room_label":"R3","verdict":"use","note":"标准卧室"}'
```

`verdict` 取 `use` / `skip` / `unsure`。章齐一栋,量产挑房就优先信你的章。

### 自写 360° 环绕渲染(owner 指定的练手件)

目的:俯视图定不了的房间(破洞?家具悬空?其实是露台?),渲一段绕房一周的视频亲眼看。

**起步配方**:

1. 环境激活**照抄现成工具**:看 `tools/visual/render_moving_source_video.py` 文件开头
   的 runtime 激活段(它接 `--runtime-prefix / --magnum-site / --rlr-sdk-root` 三个参数;
   三个参数的值在 `tools/studio/studio_config_48g.json` 的 `hm3d_episode` 模板默认值里,
   解释器用 `/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`)。
2. 相机轨道:圆心 = `bbox_xz_m` 中心、高度 = `floor_y_m + 1.5`;半径 = 短边 × 0.4;
   绕一圈 36–72 帧,每帧相机看向圆心。
3. 帧转视频:`ffmpeg -framerate 12 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4`。

**坑清单(每条都有人流过血)**:

1. 渲染加载 `<id>.glb`,**绝对不要碰 `<id>.basis.glb`**(压缩纹理版,部分加载路径直接段错误)。
2. 相机朝向一律传 **aim 向量**(看向点 − 相机位),不要传 yaw 角——本仓库历史上两处 yaw
   定义相差 60°,向量没有歧义。
3. 原始 glb 文件是 Z-up,habitat 加载后是 Y-up:**所有坐标都在 habitat 里取**,
   不要自己解析文件算;`rooms.json` 里的数已经是 habitat 坐标。
4. 长渲染挂 `nohup`/tmux,别在登录 shell 裸跑。
5. 显存/CPU 和车队共享:一次渲一间,别开并行。

**建议工作流**:俯视图能定的直接盖章(多数);拿不准的才渲 360°(1,400 间全渲不划算)。
工具写好后发我 review——好用的话晋升成 studio 正式模板,算你在仓库的第一个署名贡献。

## 任务二(排队):评测基线

等题型全案定稿 + 出题基建(P2–P5)通了以后接手。预读两份文档即可,不用现在动手:

1. **双声道管线修复**:现在 Qwen2.5-Omni 评测管线把双耳声折成单声道
   (`preprocessed_audio_shapes=[[76800]]`),左右耳线索进模型前就被抹掉——修这个是
   论文核心主张的前置。背景见 `docs/roadmap/DATA_EVAL_20260823_UNIQUE1000_QA.md` §4。
2. **基线复现配方**:`docs/roadmap/SO7B_TRAIN_ABLATION_20260824.md`
   (现最好成绩 SO-7B legacy-e3 70.9% vs Qwen 零样本 54.5%,pilot48 189 题)。

## 遇到问题

段错误/黑图/坐标不对,先对照上面坑清单;仍不解就把命令和报错贴群里喊 owner 或 Claude。
