# 交接 · 同学 A:声源 3D 资产岗(20260830)

> 你的产出:场景里"会发声的东西"的带贴图 3D 模型(音响、家电、动物…)。
> 每个资产合不合格由机器量,页面直接显示结论;你不需要读懂整个仓库。

## 访问

```
ssh -L 8765:127.0.0.1:8765 <你的服务器别名>
```

浏览器开 http://localhost:8765 → 页脚"声源资产台"(/studio/assets)。
仓库在服务器 `/data/jzy/code/AVEngine-lead-a`(main 分支)。

## 现状

- **44 个资产在库**,姿态整改已收口;当前无紧急活,按需求单补新类目。
- 资产树:`/data/avengine_external/assets/sound_source_assets_v1/<类目>/<物种或型号>/<变体>/`
  每个目录一个 `asset.json`(该资产的**唯一验收档案**)+ `finalized.glb` + `emitter_marker.glb`。
- 注册进正式数据集只有一张表(registry),`formal_dataset_registration_authorized`
  没批准前一律 research_candidate,不影响你生产。

## 怎么生产(经过实测的配方,20260829 用它出过 14 件厨房设备)

三级调用,工具都在仓库里,只喂你自己的 profile:

1. **FLUX 出图**:`avengine-imagegen` env,模型
   `/data/models/hub/models--black-forest-labs--FLUX.2-klein-4B`。
   distilled 模型:`guidance_scale=1`、steps 28、1024²。静物纯 t2i,不用 clay guide。
2. **抠图**:`tools/assets/segment_canonical_2d.py`,跑 `hunyuan3d` env(CPU 秒级)。
3. **网格**:`tools/assets/run_pixal3d_mesh.py`,`avengine-3dgen` env。

**必设环境,少一个就废**(全部有血泪案底):

- `HF_HOME=/data/models`(不设会静默重下 23G)
- `ATTN_BACKEND=sdpa`(env 里没有 flash_attn)
- `--pixal3d-root /data/jzy/code/Pixal3D-lead-b`
- `--resolution 1024`(默认 1536 会在 CuMesh 阶段 OOM;1024 还把单件从 13 分钟压到 6 分钟)
- **一件一进程**,不要在同一进程里缓存 Pixal3D+MoGe 跑批量(常驻 20G 显存,必 OOM)

## 后处理三个坑(都吃过亏)

1. glTF 根是名叫 `world` 的 EMPTY,网格在它子级——"只变换没有父级的网格"等于什么都没做。
2. 出来的模型普遍**前俯 5–27.5°**:纠姿用 ±35° 有界搜索(找水平投影面积最大的 pitch/roll),
   别用法向聚类,也别用凸包稳定面(家电最大平面是背板,会把它放倒)。
3. 包围盒对正用**凸包最小面积外接矩形**,不能用 PCA(PCA 跟顶点密度走,曾把烤盘车转了 45°)。
   深度是最弱的轴:profile 里写真实产品 W/D 做校正(±50% 夹限);救不回来的直接弃件,不硬压。

## 验收(机器判,页面看)

`asset.json` 里的 `acceptance` 字段:`resting_pose_verdict`(要 level)、
`base_normal_tilt_deg`、`mounting_plane_normal_tilt_deg`。挂墙件没有安装面就如实写
`no_mounting_plane_found`,不造假(有先例:窗机)。屏幕类倾角有设计豁免先例(电视 29.61°,
按屏幕法线仰角另判)。发声点用 `emitter` 锚点标注,`emitter_marker.glb` 可视化核对。

## 遇到问题

页面上红字/量出来的数看不懂,群里贴 asset 路径喊 owner 或 Claude;不要自己改验收脚本。
