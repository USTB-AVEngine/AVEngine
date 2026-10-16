# 人类与动物 Mesh 效率审计

> 统计方法只读取 glTF accessor 元数据或逐行扫描 OBJ，不加载几何邻接；因此可安全审计百万面模型。

## 人类面数

| 集合 | 数量 | 三角面 min / median / p95 / max | GLB MB min / median / p95 / max | 减面结论 |
|---|---:|---:|---:|---|
| Rocketbox 原生运行时 | 115 | 4,344 / 8,146 / 13,026 / 15,517 | 13.36 / 29.69 / 43.79 / 76.11 | keep_runtime_mesh; no mandatory decimation |

| 代表人类资产 | 顶点 | 三角面 | GLB MB |
|---|---:|---:|---:|
| rocketbox_adults_female_adult_01_original_ue_v1 | 5,563 | 8,732 | 32.44 |
| rocketbox_adults_male_adult_01_original_ue_v1 | 4,803 | 7,440 | 29.60 |
| rocketbox_children_female_child_01_original_ue_v1 | 4,224 | 7,102 | 25.38 |
| rocketbox_children_male_child_01_original_ue_v1 | 3,936 | 6,660 | 24.33 |
| rocketbox_professions_medical_female_01_original_ue_v1 | 5,274 | 8,328 | 31.28 |
| rocketbox_male_adult_01_shirt_blue_ue_v3 | 4,803 | 7,440 | 29.37 |
| pixal_route2_male_raw | 699,120 | 976,970 | 36.43 |
| pixal_route2_female_raw | 711,181 | 996,836 | 36.82 |
| pixal_route2_male_rigged | 712,704 | 976,951 | 51.24 |

## 动物逐资产减面判断

| 资产 | 后端/状态 | 原始三角面 | 运行时三角面 | 减少 | GLB/OBJ MB（原→运行时） | UE资产 MB | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| cat_british_shorthair_v2 | Hunyuan3D legacy; technical_spike_only | 588,108 | 39,998 | 93.2% | 10.59 → 4.76 | 28.81 | keep_runtime_mesh; no mandatory decimation |
| cat_persian | Hunyuan3D legacy; technical_spike_only | 879,606 | 40,000 | 95.5% | 15.83 → 3.14 | 30.48 | keep_runtime_mesh; no mandatory decimation |
| cat_siamese_v1 | Hunyuan3D legacy; technical_spike_only | 480,920 | 40,000 | 91.7% | 8.66 → 4.71 | 28.51 | keep_runtime_mesh; no mandatory decimation |
| cat_tabby | Hunyuan3D legacy; technical_spike_only | 660,760 | 39,998 | 93.9% | 11.89 → 2.94 | 30.26 | keep_runtime_mesh; no mandatory decimation |
| chipmunk | Hunyuan3D legacy; technical_spike_only | 391,940 | 40,000 | 89.8% | 7.05 → 3.08 | 30.86 | keep_runtime_mesh; no mandatory decimation |
| dog_beagle_v2 | Hunyuan3D legacy; technical_spike_only | 537,436 | 40,000 | 92.6% | 9.67 → 4.68 | 28.75 | keep_runtime_mesh; no mandatory decimation |
| dog_golden | Hunyuan3D legacy; technical_spike_only | 618,214 | 39,996 | 93.5% | 11.13 → 4.78 | 30.10 | keep_runtime_mesh; no mandatory decimation |
| dog_pug_v1 | Hunyuan3D legacy; technical_spike_only | 662,564 | 39,998 | 94.0% | 11.93 → 4.67 | 28.67 | keep_runtime_mesh; no mandatory decimation |
| cattle_bovinae | Hunyuan3D legacy; technical_spike_only | 488,292 | 40,000 | 91.8% | 8.79 → 1.34 | 5.34 | keep_runtime_mesh; no mandatory decimation |
| donkey_ass | Hunyuan3D legacy; technical_spike_only | 418,216 | 40,000 | 90.4% | 7.53 → 1.44 | 4.88 | keep_runtime_mesh; no mandatory decimation |
| goat | Hunyuan3D legacy; technical_spike_only | 195,036 | 40,000 | 79.5% | 3.51 → 1.29 | 3.47 | keep_runtime_mesh; no mandatory decimation |
| horse | Hunyuan3D legacy; technical_spike_only | 440,514 | 40,000 | 90.9% | 7.93 → 1.34 | 4.87 | keep_runtime_mesh; no mandatory decimation |
| pig | Hunyuan3D legacy; technical_spike_only | 245,470 | 40,000 | 83.7% | 4.42 → 1.42 | 5.46 | keep_runtime_mesh; no mandatory decimation |
| sheep | Hunyuan3D legacy; technical_spike_only | 1,032,536 | 40,000 | 96.1% | 18.59 → 1.36 | 5.10 | keep_runtime_mesh; no mandatory decimation |
| yak | Hunyuan3D legacy; technical_spike_only | 43,752 | 40,000 | 8.6% | 0.79 → 1.34 | 3.37 | keep_runtime_mesh; no mandatory decimation |
| dog_pug_pixal_canary_v2_100k | Pixal3D; research_candidate | 931,167 | 100,000 | 89.3% | 34.92 → 33.43 | 81.77 | keep_runtime_mesh; no mandatory decimation |

## Pixal 狗 40k / 100k UE 实测

同一台 RTX 4090 D、640×480、72 帧、每张 PNG 回读，均含相同地板、灯光、动画和相机绕视。单角色时主要瓶颈是 UE 冷启动、固定步进和 PNG 回读，因此面数差异未形成可测的吞吐收益。

| LOD | 质量 | UE启动 s | 场景准备 s | 72帧 s | 捕获 fps | 单帧 p95 ms | GPU0峰值 MiB | GPU峰值利用率 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | passed_no_holes_double_sided | 4.17 | 8.73 | 15.74 | 4.57 | 241.9 | 2358 | 48% |
| 40,000 | rejected_holes_single_sided | 3.91 | 8.58 | 16.00 | 4.50 | 246.9 | 2340 | 44% |

40k 与 100k 的 72 帧耗时分别约 16.00s 和 15.74s，差异仅 1.6% 且方向相反，属于冷启动/回读噪声；100k 只多约 18 MiB 峰值显存，却消除了可见空洞并显著改善轮廓，所以近景固定采用 100k。40k 只保留为远景候选，且必须重新生成双面版本后再批量压力测试。

## 当前统一策略

- 运行时不超过 100k 三角面：默认不减面；只做纹理、法线、脚部和 UE 回读 QA。
- 100k–150k：保留近景 LOD，可选生成 40k 远景 LOD。
- 超过 150k：生成约 100k 近景和约 40k 远景两级 LOD；Pixal 局部绕序不一致时保留双面材质。
- 结论不能只看面数：任何减面代理都必须经过 Front/Back/Side/Top、Walk/Idle、UE PAK 回读和地面阴影检查。
- Hunyuan 行仅是历史性能证据，保持 technical_spike_only；正式动物必须用 Pixal3D 重新生成并逐级过门禁。
