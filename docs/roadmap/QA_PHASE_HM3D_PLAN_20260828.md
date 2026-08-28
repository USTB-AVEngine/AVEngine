# HM3D 接入认证 QA 链:阶段方案(2026-08-28)

> 状态:方案,未动手。按审计规矩写:每一步先说"用什么验收",
> 不确定的事标成"待验证",不写进承诺。

## 0. 链的真实形状(读代码得到,非猜测)

认证 QA 链(pilot48 / batch2d 909 题走过的那条)是:

```
规划产物                     声学渲染                 聚合            出题/认证
trajectory_bank.json   →   资产绑定双耳批       →   fact table  →  miner → certify
rir_job_plan.json          (render_asset_bound_     (compile_*_     (mine_simple /
                            binaural_batch)          fact_tables)    mine_temporal /
                                                                     certify_axis1)
```

fact table 编译器(`tools/qa/compile_apartment_fact_tables.py`)的输入合同:
`--plan-dir`(trajectory_bank.json + rir_job_plan.json)、`--audio-batch-dir`、
声源运行时注册表、**锚点库**、**room capsule**、**m1 相机标定请求**
(带"必须与 RIR 听者一致"的闸)、schema。它只聚合、绝不重渲;新增值只有
解析推导(听者系 DoA、速度、朝向、锚点距离)。

**关键结论:我为试听/演示搭的 ss2 episode 链(run_hm3d_episode)不是这条链
的输入格式。** 它继续服务人工验收;进认证链要走原生格式。

## 1. HM3D 手里已经有什么

| 认证链要的 | HM3D 现状 |
|---|---|
| habitat_native 房间清单 | ✅ stage① 批量产出,带实测连通性 |
| M3 声学包 | ✅ 原生编译 + **帧视差准入**(12/12) |
| 房间划分 | ✅ rooms.json(00800 十三间,床/沙发/冰箱可辨) |
| trajectory_bank + rir_job_plan | ❌ 我的 routes 银行是另一种格式 |
| 锚点库 / room capsule | ❌ 无,需从 rooms.json 派生 |
| m1 相机标定请求 | ❌ 无,MP3D 链的 route author 会生成 |
| 受控实例属性(歧义压力) | ⚠️ FLUX 动物 / Rocketbox 换色机制在 apartment 侧验证过,HM3D 未接 |

## 2. 两条候选路线,以及先验证哪个

**路线 A(优先验证):HM3D 房间直接搭 MP3D 原生链的车。**
`run_mp3d_end_to_end` 吃 habitat_native 房间清单 + M3 包 + m2 声源资产,
产整条 m5/m6x 链;我已把 room_manifest / package_manifest / mp3d_root 开成
模板可覆盖参数。HM3D 房间在契约上就是 habitat_native。
**待验证的三件**(每件都可能否决路线 A):
1. mp3d route author 的规划产物是否与 fact-table 的 plan-dir 同 schema;
2. 相机标定与 RIR 听者一致性闸在 HM3D 房间上如何满足;
3. hm3d 场景配置(scene_dataset_config)喂给 m1 采集时,视觉端是否踩
   *.basis.glb(必须指向非压缩 glb)。

**路线 B(备胎):写适配器**把 routes 银行 + 听者位姿转成
trajectory_bank/rir_job_plan 格式。工作量更大,但每一环都是我们自己的。

## 3. 阶段划分(每阶段一个验收器,过了才进下一阶段)

| 阶段 | 做什么 | 验收 |
|---|---|---|
| P0 参照跑 | 用默认参数把 mp3d_end_to_end 完整跑一遍留档 | 链自己的产物 + fact table 能编译该输出(**这一步同时回答 §2 的待验证 1**) |
| P1 换房间 | 同链,房间覆盖为 HM3D 00800(清单+正身包) | 与 P0 相同产物集,视差 12/12,fact table 编译通过 |
| P2 房间语义 | 从 rooms.json 派生 HM3D 锚点库与 room capsule | fact table 的锚点距离字段非空且抽查与俯视图一致 |
| P3 小批出题 | miner(simple+temporal)+ certify 在 P1 产物上 | certify 通过率与 batch2d 基线同量级;逐题证书完整 |
| P4 歧义压力 | 接受控实例(≥2 同类实例/场景)与答案均衡工单 | v2 公理 1/2 的闸门指标 |
| P5 模板化 | P0–P3 各成 studio 模板,看板加 fact/qa 两列 | 同学 C 能点按钮出题 |

风险清单:P1 的相机-听者一致性闸(最可能卡);HM3D 扫描房的 m2 动物
资产落地(接触/穿模policy 沿用 owner 决定);certify 对遮挡帧的语义
(遮挡是保留样本,不是失败)。

## 4. 与两位同学的接口

- 同学 B 的干声库(/studio/sounds)按资产档案词表补类;audio program
  出题侧从注册表选声,**库→注册表的晋升脚本**在 P3 前补上。
- 同学 C 的量产在 P5 后自然扩展:看板从五列变七列。
