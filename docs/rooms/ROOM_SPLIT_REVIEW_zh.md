# 切房结果审核页

给人逐间审最终房间名单（HM3D / MP3D / 酷家乐）用的网页。2026-10-10 第一次用于 1329 间（HM3D 1014、MP3D 244、酷家乐 71），审核人 smy。

## 生成

```bash
python tools/rooms/room_split_review/build_review_site.py --config CONFIG.json --out 新目录 --workers 16
```

- 输出目录必须是新的，已存在就拒绝。
- 每个来源区域一张卡：真实俯视图按区域裁出来，名单里每间一种颜色、正中大号编号和面积；切分时丢掉的块画斜线，没判定或不在名单里的块画网格；同一层别的房间画白色细线。
- 只用 CPU，不改任何输入。1160 张图 16 进程约 30 秒。
- `build_receipt.json` 记下输入文件的 sha256、各家数量、漏声分档、缺底图的卡。

配置格式见 `build_review_site.py` 开头的说明。每一家要给：房间名单（带 `rooms` 数组的 JSON）、区域文件目录（块的 retain/discard/unresolved）、整层俯视图目录（`{house}__Y*.json`，带 view/projection/size_px/floor_y_m/span_m/path）、可选的单区域俯视图目录、漏声结果 CSV（后面的覆盖前面的）、可选的旧审核结论目录（`{house}__{region}.json`）。

## 审核和保存

- 推荐：`python3 serve_review.py --port 8790`（生成时会拷进输出目录，只用标准库）。只绑 127.0.0.1，用 ssh -L 或 VS Code 转发端口。每次点击都存到 `feedback/review_feedback.json`（原子写），另外逐条追加到 `feedback/review_feedback.log.jsonl`。
- 也可以直接打开 `index.html`：结果存在浏览器里，看完点「导出 JSON」。换浏览器可以「导入 JSON」接着审，旧记录不会覆盖新记录。
- 每条记录：`assessment`（ok / bad / unsure）、`reason_codes`、`note`、`reviewer`、`updated_at`（UTC）。房间用房间 ID，丢掉部分用 `dropped:<卡片 ID>`。

## 核查

`tests/unit/test_room_split_review.py` 覆盖生成、拒绝覆盖、保存服务的校验和新旧合并。2026-10-10 在 48g 用无头 Chromium 138 实点过：1156 张卡全部渲染，点击后实时存盘，刷新后能恢复，筛选计数正确，首屏只加载 2 张图。

## 可用性规则（2026-10-10，按审核人 smy 的标准）

smy 第一轮审了 HM3D 里 9 月标过「不用 / 存疑」的 269 间：177 间对、92 间不对。切出来的 27 间只否了 1 间，否掉的多是「这类房间要不要」：宽过道和门厅、没家具的空房间、卫生间、太小的房间。smy 的理由：

- 宽过道、空房间：没有东西可以问距离，也没有遮挡，只能问纯距离题，不要。
- 卫生间：一般比较窄，放不下人和机器，不要。
- 小房间：放得下机器、人有活动范围就留；smy 审存疑房间时大约 7.5 m² 以上就留。

写成规则（`tools/rooms/room_usability/rules.py`）。审过的房间以审核人的结论为准，没审过的房间只要有一条成立就去掉：

- 卫生间：标成卫生间；或者面积不到 10 m² 且里面有马桶、浴缸、淋浴（更大的房间带这些，多半是带独卫的卧室）。
- 面积小于 7.5 m²。
- 大件家具不到 2 件：桌椅、沙发、床、柜子、电器、健身器材这类，占地至少 0.25 m²；枕头、窗帘、挂画、电视不算。

家具来自数据集自带的物体标注（`tools/rooms/room_usability/objects.py`）：HM3D 和 MP3D 按物体标注所在的区域归到房间，切开的区域归给最近的那间；HM3D 的地面多边形在家具底下有洞，不能用「中心点落在多边形里」来数。MP3D 只有 mpcat40 粗类，「furniture」「gym equipment」这类也算家具。酷家乐用切分时整理好的贴地家具占地，按最小外接矩形算面积（贴地切片里桌子只剩四条腿）。

```bash
python tools/rooms/room_usability/objects.py --family hm3d --rooms HM3D名单.json --out 新目录/hm3d
python tools/rooms/room_usability/objects.py --family mp3d --rooms MP3D名单.json --mp3d-adapter-root MP3D适配器检出 --out 新目录/mp3d
python tools/rooms/room_usability/objects.py --family kujiale --rooms 酷家乐名单.json --kujiale-adapters 场景适配目录 --out 新目录/kujiale
python tools/rooms/room_usability/rules.py --site-data 审核页/site_data.json --config 审核配置.json \
  --feedback 审核结果.json --objects 新目录/*/objects.csv --out 新目录/lists
```

和 smy 的结论对照（HM3D 审过的 269 间）：她否掉的 92 间规则抓到 68 间，抓不到的主要是摆了几件家具的过道；她说对的 177 间规则会误去 16 间（11 间是 7–7.5 m² 的小房间），这些以她的结论为准留下。2026-10-10 套完：三家 1329 → 1062 间。HM3D 807 间（测试 732 / 只训练 62 / 不用 13），MP3D 185 间（127 / 54 / 4），酷家乐 70 间（全测试）。owner 当天确认这份名单作数。

## 第二轮

规则筛完后，没有人看过的房间（HM3D 新切出来的、第一轮跳过的、MP3D、酷家乐）先逐张看图筛一遍，拿不准的卡写进一个轮次文件（`docs/rooms/room_split_review_round2_20261010.json`：标题、说明、卡片 ID 和类别），再只给审核人看这些卡：

```bash
python tools/rooms/room_split_review/subset_config.py --config 审核配置.json --lists 新目录/lists \
  --round docs/rooms/room_split_review_round2_20261010.json --site-data 第一轮审核页/site_data.json --out 新目录/round2
python tools/rooms/room_split_review/build_review_site.py --config 新目录/round2/review_config.json --out 新目录/round2_site
```

2026-10-10 第二轮：没人看过的 332 张卡（452 间）里挑出 65 张卡、106 间。其中家具少、看着空的 21 张；过道、门厅或长条形的 24 张；切得形状怪的 16 张；房间里有楼梯，或者像泳池、大卫生间的 4 张。选中卡里留在名单上的房间整张显示。
