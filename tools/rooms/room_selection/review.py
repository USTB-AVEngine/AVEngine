"""Produce the shared smy manifest; there is no second UI or HTTP server."""

from pathlib import Path
import json
from tools.rooms.room_screening.build_review_manifest import build_manifest
from .analysis import REASONS
from .runtime import MEDIA_ROOT
from .run import write_json


def metric_cards(m):
    if not m:
        return []
    return [
        dict(label="语义地面面积", value=m.get("floor_area_m2")),
        dict(label="已知家具占地", value=m.get("furniture_footprint_m2")),
        dict(label="navmesh 主连续区域", value=m.get("nav_main_area_m2")),
        dict(label="短边", display=f"{m.get('short_side_m',0):.2f} m"),
        dict(
            label="黑像素比例",
            display=(
                "未评估"
                if m.get("black_fraction") is None
                else f"{m['black_fraction']*100:.2f}%"
            ),
        ),
        dict(
            label="摆放证据",
            display="找到" if (m.get("placement") or {}).get("found") else "未找到",
        ),
    ]


def build_review(out, rows, sample):
    second = {(r["house"], r["room_label"]) for r in sample["samples"]}
    geometry_dir = out / "review_geometry"
    geometry_dir.mkdir(exist_ok=True)
    items = []
    for row in rows:
        base_id = row["house"] + "__" + row["room_label"]
        units = [(base_id, "original_region", row.get("metrics"), row["stage1"], None)]
        for floor in row.get("floors", []):
            for part in floor.get("split_parts", []):
                units.append(
                    (
                        base_id + "__" + part["part_id"],
                        "split_part",
                        dict(part["metrics"], floor_y_m=floor["metrics"]["floor_y_m"]),
                        part["stage1"],
                        base_id,
                    )
                )
        overlays = [
            str(out / f["stage2"]["overlay_path"])
            for f in row.get("floors", [])
            if f["stage2"].get("overlay_path")
        ]
        overhead = row.get("overhead")
        for uid, kind, m, decision, parent in units:
            item = dict(
                id=uid,
                house=row["house"],
                label=row["room_label"] if parent is None else uid.split("__")[-1],
                unit_kind=kind,
                automatic=decision,
                gate=row["stage4"],
                split=row["stage2"],
                split_overlay_paths=overlays,
                floor_selection=row.get("floor_selection"),
                metrics=metric_cards(m),
                second_review=parent is None
                and (row["house"], row["room_label"]) in second,
                human_review_status="not_reviewed_in_merged_protocol",
                scope_note="语义扫描范围是测量代理；子块必须独立判断，不能继承原区域 use。",
                review_note="自动原因：" + ", ".join(decision.get("reason_codes", [])),
            )
            if parent:
                item["parent_id"] = parent
            image_path = Path(overhead["image_cache"]) if overhead else None
            if image_path and image_path.is_file():
                item["image_path"] = str(image_path)
                meta = json.loads(Path(overhead["metadata_cache"]).read_text())
                record = next(
                    i for i in meta["images"] if i["path"].endswith("_overview.png")
                )
                from PIL import Image

                item.update(
                    coordinate_space="world_xz",
                    view=meta["view"],
                    projection=record["projection"],
                    floor_y_m=meta["floor_y_m"],
                    image_size=list(Image.open(image_path).size),
                )
                if (
                    m
                    and m.get("floor_polygon")
                    and abs(m.get("floor_y_m", meta["floor_y_m"]) - meta["floor_y_m"])
                    <= 0.3
                ):
                    g = dict(
                        ground=m["floor_polygon"],
                        raw=m.get("nav_polygon"),
                        corrected=m["floor_polygon"],
                        blockers=m.get("furniture_polygon"),
                    )
                    gp = geometry_dir / (uid + ".json")
                    write_json(gp, g)
                    item["geometry_path"] = str(gp)
            else:
                item["image_missing_reason"] = (
                    "原始俯视图缺失；先补素材，不用合成图替代扫描。"
                )
            if row.get("video_exists"):
                item["video_path"] = row["video_path"]
            items.append(item)
    source = dict(
        title="HM3D 合并流程人工审核",
        purpose="固定规则测量 + 人工审核；几何代理不是建筑范围真值",
        source_note="room_screening 测量层 / room_selection 判断层；原始数据与第一次审核章只读",
        choices=["use", "skip", "unsure"],
        reason_catalog=REASONS,
        labels_status="not_judged",
        second_reviewer_sample_source=str(out / "second_reviewer_sample.json"),
        items=items,
    )
    manifest = build_manifest(source, out.resolve(), MEDIA_ROOT.resolve())
    write_json(out / "review_manifest.json", manifest)
    return manifest
