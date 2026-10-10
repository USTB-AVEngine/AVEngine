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
