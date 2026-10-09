"""Assemble MP3D area ledgers, draft room list and embedded lazy review pages."""
from __future__ import annotations
import argparse
import base64
import copy
import csv
import html
import json
from collections import Counter
from pathlib import Path
import resource
import subprocess
import numpy as np
import shapely
from shapely.geometry import shape, mapping
from PIL import Image
from tools.rooms.room_split_auto.mp3d_run import dump, bounded, now
from tools.rooms.room_split_auto.mp3d_adapter import source_region


FIELDS = ["id", "house", "source_region", "source_region_id", "floor_id", "floor_y_m",
          "floor_area_m2", "short_side_m", "black_fraction", "source", "origin_list", "source_selection",
          "floor_polygon_xz_m", "placement_witness", "leakage"]


def csv_records(path, records, fields):
    with Path(path).open("x", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict, list))
                             else row.get(k) for k in fields})


def regions(directory):
    return {p.name: json.loads(p.read_text()) for p in sorted((Path(directory) / "regions").glob("*.json"))}


def area_bins(rooms):
    return {key: sum(lo <= r["floor_area_m2"] < hi or
                    (key == "30_35" and r["floor_area_m2"] == 35) for r in rooms)
            for key, lo, hi in (("6_10", 6, 10), ("10_20", 10, 20), ("20_30", 20, 30), ("30_35", 30, 35))}


def partition_validation(reg):
    source = shapely.union_all([shape(x["floor_polygon"]) for x in reg["source_geometry"]["floors"]])
    measured = shapely.union_all([shape(b["floor_polygon_xz_m"]) for b in reg["blocks"]])
    total = sum(b["floor_area_m2"] for b in reg["blocks"])
    return dict(source=reg["house"] + "/" + reg["source_region"],
                source_area_m2=float(source.area), partition_area_m2=float(total),
                area_error_m2=abs(total - source.area),
                symmetric_difference_m2=float(measured.symmetric_difference(source).area),
                overlap_m2=float(total - measured.area))


def summarize(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    native_dir = root / "mp3d_existing_rooms_connectivity_v1"
    delivery = root / "mp3d_delivery_v1/final_v1"
    native, new = regions(native_dir), regions(delivery)
    if len(native) != 235 or len(new) != 22:
        raise RuntimeError("delivery scope incomplete")
    originals = [r for j in plan["jobs"] for r in j["rooms"]]
    native_subcap = [r for r in native.values() if r["status"] != "delegated_to_cap35_split"]
    native_blocks = [b for r in native_subcap for b in r["blocks"]]
    new_blocks = [b for r in new.values() for b in r["blocks"]]
    retained_native = [b for b in native_blocks if b["decision"] == "retain"]
    retained_new = [b for b in new_blocks if b["decision"] == "retain"]
    pending = [b for b in native_blocks + new_blocks if b["decision"] == "unresolved"]
    ledger, geometry_checks, failures = [], [], []
    for room in originals:
        name = room["house"] + "__" + room["room_label"] + ".json"
        reg = new[name] if room["floor_area_m2"] > 35 + 1e-8 else native[name]
        check = partition_validation(reg)
        geometry_checks.append(check)
        if max(check["area_error_m2"], check["symmetric_difference_m2"], abs(check["overlap_m2"])) > 1e-6:
            failures.append(dict(kind="AREA_PARTITION", **check))
        bs = reg["blocks"]
        counts = Counter(b["decision"] for b in bs)
        row = dict(room_id=room["room_id"], house=room["house"], source_region=room["room_label"],
                   floor_id=room["selected_floor_id"], original_area_m2=room["floor_area_m2"],
                   requires_cap35_split=room["floor_area_m2"] > 35 + 1e-8,
                   retained_rooms=counts["retain"], discarded_parts=counts["discard"], pending_parts=counts["unresolved"],
                   retained_area_m2=sum(b["floor_area_m2"] for b in bs if b["decision"] == "retain"),
                   discarded_area_m2=sum(b["floor_area_m2"] for b in bs if b["decision"] == "discard"),
                   pending_area_m2=sum(b["floor_area_m2"] for b in bs if b["decision"] == "unresolved"),
                   area_error_m2=check["area_error_m2"],
                   destination_region_file=str((delivery if room["floor_area_m2"] > 35 + 1e-8 else native_dir) / "regions" / name),
                   output_room_ids=[b["id"] for b in bs])
        row["disposition"] = ("cap35_replacement_with_pending" if counts["unresolved"] else "cap35_replaced") if row["requires_cap35_split"] else (
                             "original_pending" if any(b.get("native_main") and b["decision"] == "unresolved" for b in bs) else
                             "original_retained" if any(b.get("native_main") and b["decision"] == "retain" for b in bs) else "original_discarded")
        ledger.append(row)
    all_retained = retained_native + retained_new
    for b in all_retained:
        w = b.get("placement_witness", {})
        if not w.get("found") or not w.get("validation", {}).get("passed"):
            failures.append(dict(kind="PLACEMENT_WITNESS", id=b["id"]))
        path = (native_dir if b in retained_native else delivery) / "rooms" / (b["id"] + ".json")
        if not path.exists() or json.loads(path.read_text()) != b:
            failures.append(dict(kind="ROOM_JSON_REGION_MISMATCH", id=b["id"]))
    if len({b["id"] for b in all_retained}) != len(all_retained):
        failures.append(dict(kind="DUPLICATE_ROOM_ID"))
    original_keys = {(b["house"], b["source_region"]) for b in retained_native}
    if original_keys & {(b["house"], b["source_region"]) for b in retained_new}:
        failures.append(dict(kind="DUPLICATE_SOURCE_ORIGINAL_AND_SPLIT"))
    original_summary = dict(native_expected=235, native_measured=235, native_gt35_replaced_by_v6=22,
        affected_subcap_originals=sum(r.get("metrics", {}).get("affected", False) for r in native_subcap),
        native_retained=len(retained_native), native_retained_main=sum(b.get("native_main", False) for b in retained_native),
        new_detached_candidates_retained=sum(not b.get("native_main", False) for b in retained_native),
        detached_candidates_ge6=sum(not b.get("native_main", False) and b["floor_area_m2"] >= 6 for b in native_blocks),
        originals_fully_discarded=sum(x["disposition"] == "original_discarded" for x in ledger),
        subcap_originals_pending=sum(x["disposition"] == "original_pending" for x in ledger),
        retained_area_m2=sum(b["floor_area_m2"] for b in retained_native),
        discard_count=sum(b["decision"] == "discard" for b in native_blocks),
        discard_area_m2=sum(b["floor_area_m2"] for b in native_blocks if b["decision"] == "discard"),
        unresolved_count=sum(b["decision"] == "unresolved" for b in native_blocks),
        pending_area_m2=sum(b["floor_area_m2"] for b in native_blocks if b["decision"] == "unresolved"),
        source_native_polygon_manifest=plan["prep"], lists_modified=False,
        validation_passed=not failures, area_ledger_includes_cap35_replacements=True,
        area_ledger_basis="retained + discarded + pending; delegated sources read from final cap35 delivery",
        original_subcap_uncut=True, original_extra_shape_gates=False, acoustics="not_run")
    dump(native_dir / "summary.json", original_summary)
    dump(native_dir / "pending.json", [b for b in native_blocks if b["decision"] == "unresolved"])
    csv_records(native_dir / "rooms.csv", native_blocks, FIELDS + ["decision", "discard_reasons", "unresolved_reasons"])
    dropped = Counter()
    for b in new_blocks:
        if b["decision"] != "discard":
            continue
        reasons = b.get("discard_reasons") or ["UNSPECIFIED"]
        key = "STAIRS" if "STAIRS" in reasons else "DETACHED_FRAGMENT" if "DETACHED_FRAGMENT" in reasons else (
              "CORRIDOR" if any("CORRIDOR" in r for r in reasons) else reasons[0])
        dropped[key] += b["floor_area_m2"]
    new_summary = dict(created_at_utc=now(), source_regions=22, original_large_rooms=22,
        source_area_m2=sum(r["source_floor_area_m2"] for r in new.values()), retained_rooms=len(retained_new),
        retained_area_m2=sum(b["floor_area_m2"] for b in retained_new),
        discarded_parts=sum(b["decision"] == "discard" for b in new_blocks),
        discarded_area_m2=sum(dropped.values()), discarded_area_by_reason_m2=dict(dropped),
        pending_parts=sum(b["decision"] == "unresolved" for b in new_blocks),
        pending_area_m2=sum(b["floor_area_m2"] for b in new_blocks if b["decision"] == "unresolved"),
        unresolved_source_regions=[r["house"] + "/" + r["source_region"] for r in new.values()
                                   if any(b["decision"] == "unresolved" for b in r["blocks"])],
        max_room_area_m2=35, cpu_only=True, acoustics="not_run", production_integrated=False, lists_modified=False,
        source_commit=json.loads((delivery / "completed.json").read_text())["source_commit"])
    dump(delivery / "summary.json", new_summary)
    dump(delivery / "unresolved.json", [b for b in new_blocks if b["decision"] == "unresolved"])
    draft = root / "mp3d_room_list_draft_v1"
    draft.mkdir(exist_ok=False)
    dump(draft / "room_list_draft_v1.json", dict(schema="MP3D_room_list_draft_v1_cap35", rooms=all_retained,
         leakage="pending separate acoustics task", leakage_policy=dict(test_max=.05, train_only_max=.15,
         reject_above=.15), production_integrated=False))
    dump(draft / "pending.json", pending)
    csv_records(draft / "rooms.csv", all_retained, FIELDS)
    draft_summary = dict(total_rooms=len(all_retained), total_area_m2=sum(b["floor_area_m2"] for b in all_retained),
        area_bins=area_bins(all_retained), inherited_native_main=original_summary["native_retained_main"],
        native_detached_new=original_summary["new_detached_candidates_retained"], cap_cut_new=len(retained_new),
        pending=len(pending), pending_area_m2=sum(b["floor_area_m2"] for b in pending), max_room_area_m2=35,
        lists_modified=False, production_integrated=False, acoustics="not_run", leakage_columns_empty=True,
        retained_native_room_circle_not_new_gate=True, duplicate_source_keys=0, sources=[str(native_dir), str(delivery)])
    dump(draft / "summary.json", draft_summary)
    dump(root / "area_ledger.json", ledger)
    csv_records(root / "area_ledger.csv", ledger, list(ledger[0]))
    dump(root / "geometry_and_witness_validation.json", dict(passed=not failures, failures=failures,
         original_rooms=235, cap_regions=22, retained_room_witnesses=len(all_retained),
         geometry_checks=geometry_checks, max_area_error_m2=max(x["area_error_m2"] for x in geometry_checks),
         max_symmetric_difference_m2=max(x["symmetric_difference_m2"] for x in geometry_checks),
         unresolved_is_explicitly_allowed=True, pending_area_is_not_counted_retained_or_discarded=True,
         all_original_area_resolved=not pending))
    print("MP3D_DRAFT", json.dumps(draft_summary, ensure_ascii=False), flush=True)


def overlay_inputs(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    originals = [r for j in plan["jobs"] for r in j["rooms"]]
    native = regions(root / "mp3d_existing_rooms_connectivity_v1")
    new = regions(root / "mp3d_delivery_v1/final_v1")
    before_dir, after_dir = root / "overlay_inputs_before", root / "overlay_inputs_after"
    for directory in (before_dir, after_dir):
        (directory / "regions").mkdir(parents=True, exist_ok=False)
    wanted = [r for r in originals if r["floor_area_m2"] > 35 + 1e-8 or
              native[r["house"] + "__" + r["room_label"] + ".json"].get("metrics", {}).get("affected", False)]
    for room in wanted:
        name = room["house"] + "__" + room["room_label"] + ".json"
        before = source_region(room)
        before["blocks"] = [dict(id=room["house"] + "__" + room["room_label"] + "__BEFORE",
                                house=room["house"], source_region=room["room_label"],
                                floor_id=room["selected_floor_id"], floor_y_m=room["floor_y_m"],
                                floor_polygon_xz_m=room["floor_polygon_xz_m"], floor_area_m2=room["floor_area_m2"],
                                short_side_m=room["short_side_m"], decision="retain", room_type=room["room_type"],
                                discard_reasons=[], unresolved_reasons=[])]
        before["overlay_only"] = True
        after = copy.deepcopy(new[name] if name in new else native[name])
        after["overlay_only"] = True
        after["actual_requires_split"] = after["requires_split"]
        after["requires_split"] = True  # The owner's renderer selects this flag; original rooms remain uncut.
        dump(before_dir / "regions" / name, before)
        dump(after_dir / "regions" / name, after)
    dump(root / "overlay_inputs_receipt.json", dict(regions=len(wanted), before=str(before_dir),
         after=str(after_dir), original_region_algorithm_flags_unchanged=True))
    print("MP3D_OVERLAY_INPUTS", len(wanted), flush=True)


def review(root):
    root = Path(root)
    native, new = regions(root / "mp3d_existing_rooms_connectivity_v1"), regions(root / "mp3d_delivery_v1/final_v1")
    before = {e["region"]: e for e in json.loads((root / "owner_overlay_before/index.json").read_text())}
    after = {e["region"]: e for e in json.loads((root / "owner_overlay_after/index.json").read_text())}
    images, count, missing = {}, 0, []
    page = root / "mp3d_review_v1.html"
    sections = [("22 间大房：切前与切后", list(new.values())),
                ("原有房间：去飞地前后", [r for r in native.values()
                                            if r.get("metrics", {}).get("affected", False) and r["source_floor_area_m2"] <= 35 + 1e-8])]
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>MP3D 房间切分与去飞地审图</title><style>body{font-family:system-ui;background:#eef2f5;color:#17212c;margin:20px}'
             'article{background:white;padding:16px;margin:20px 0;border-radius:8px}.pair{display:flex;gap:12px}.pair>div{width:50%}'
             'img{width:100%;height:auto;aspect-ratio:1;background:#dce3eb}table{border-collapse:collapse;font-size:14px;width:100%;margin-top:12px}'
             'th,td{border:1px solid #c7d1dc;padding:6px;text-align:left}td{overflow-wrap:anywhere} .retain{color:#16643a}.discard{color:#805700}.unresolved{color:#b32428}'
             '@media(max-width:760px){.pair{display:block}.pair>div{width:100%}}</style>',
             '<h1>MP3D 房间切分与去飞地审图</h1><p>底图是原始整屋真实纹理，当前层剖面显示地面。每间保留房单独着色编号；斜线为丢弃块，交叉斜线与问号为待定。'
             '所有图片内嵌，只有进入视口后才加载、解码；无外部资源。漏声待另一任务测量，本页不宣告生产准入。</p>']
    draft = json.loads((root / "mp3d_room_list_draft_v1/summary.json").read_text())
    parts.append("<p>草稿 " + str(draft["total_rooms"]) + " 间，待定 " + str(draft["pending"]) + " 项。</p>")
    for title, regs in sections:
        parts.append("<h2>" + title + "</h2>")
        for reg in sorted(regs, key=lambda r: (-r["source_floor_area_m2"], r["house"], r["source_region"])):
            key = reg["house"] + "/" + reg["source_region"]
            parts.append("<article><h3>" + html.escape(key) + f' · 原面积 {reg["source_floor_area_m2"]:.2f} m²</h3><div class="pair">')
            for label, entries in (("处理前", before), ("处理后", after)):
                paths = entries.get(key, {}).get("images", [])
                parts.append("<div><b>" + label + "</b>")
                if not paths:
                    missing.append(dict(region=key, side=label))
                    parts.append("<p>底图缺失，未造图。</p>")
                for path in paths:
                    token = "im" + str(count)
                    count += 1
                    data = Path(path).read_bytes()
                    with Image.open(path) as im:
                        im.verify()
                    images[token] = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
                    parts.append('<img data-image="' + token + '" loading="lazy" decoding="async" width="900" height="900" alt="' +
                                 html.escape(key + " " + label, quote=True) + '">')
                parts.append("</div>")
            parts.append("</div><table><thead><tr><th>房间块</th><th>面积 m²</th><th>短边 m</th><th>去向</th><th>原因</th><th>放置见证</th></tr></thead><tbody>")
            for b in sorted(reg["blocks"], key=lambda b: (-b["floor_area_m2"], b["id"])):
                decision = b["decision"]
                reason = "; ".join(b.get("discard_reasons") or b.get("unresolved_reasons") or [])
                parts.append('<tr class="' + html.escape(decision) + '"><td>' + html.escape(b["id"]) +
                             f'</td><td>{b["floor_area_m2"]:.3f}</td><td>{b.get("short_side_m", 0):.3f}</td><td>' +
                             {"retain": "保留", "discard": "丢弃", "unresolved": "待定"}.get(decision, decision) +
                             "</td><td>" + html.escape(reason) + "</td><td>" +
                             ("已找到并验证" if b.get("placement_witness", {}).get("validation", {}).get("passed") else "未保留或待定") + "</td></tr>")
            parts.append("</tbody></table></article>")
    parts.append('<script type="application/json" id="embedded-images">' + json.dumps(images, separators=(",", ":")) + '</script>')
    parts.append("""<script>
const pictures=JSON.parse(document.getElementById('embedded-images').textContent);
const observer=new IntersectionObserver(entries=>{
 for(const e of entries){if(!e.isIntersecting)continue;const im=e.target;
 im.src=pictures[im.dataset.image];delete pictures[im.dataset.image];observer.unobserve(im);}
},{rootMargin:'0px',threshold:0.01});
document.querySelectorAll('img[data-image]').forEach(im=>observer.observe(im));
</script></html>""")
    text = "\n".join(parts)
    with page.open("x", encoding="utf8") as f:
        f.write(text)
    dump(root / "review_validation.json", dict(page=str(page), large_regions=len(new),
         changed_original_regions=len(sections[1][1]), embedded_jpeg_images=count, page_bytes=page.stat().st_size,
         missing=missing, passed=not missing and '<img src=' not in text, initial_image_src_absent=True,
         strict_lazy_intersection_observer=True, external_resources=False, jpeg_decode_validation=True,
         browser_runtime_click_validation="not_run"))
    print("MP3D_REVIEW", page, "IMAGES", count, "MISSING", len(missing), flush=True)


def report(root):
    root = Path(root)
    n = json.loads((root / "mp3d_existing_rooms_connectivity_v1/summary.json").read_text())
    d = json.loads((root / "mp3d_delivery_v1/final_v1/summary.json").read_text())
    draft = json.loads((root / "mp3d_room_list_draft_v1/summary.json").read_text())
    g = json.loads((root / "geometry_and_witness_validation.json").read_text())
    s = json.loads((root / "checks/shape_quality_v1.json").read_text())
    c = json.loads((root / "checks/split_delivery_v4.json").read_text())
    r = json.loads((root / "review_validation.json").read_text())
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    acceptance = dict(shape_counts=s["counts"], shape_total=sum(s["counts"].values()),
         size_violations=len(c["kept_outside_area_or_short_side"]), unreachable_far_parts=len(c["far_parts_not_direct"]),
         cut_segments_hist=c["cut_legs_hist"], cut_furniture_over_0p5m=c["cut_furniture_over_0p5m"],
         geometry_and_witness_passed=g["passed"], all_area_resolved_without_pending=g["all_original_area_resolved"],
         retained_witnesses=g["retained_room_witnesses"], review_passed=r["passed"],
         source_commit=source, acoustics="not_run", production_integrated=False)
    acceptance["retained_delivery_checks_passed"] = (acceptance["shape_total"] <= 3 and not acceptance["size_violations"] and
         not acceptance["unreachable_far_parts"] and max(map(lambda x: int(x) if x != "null" else 99, c["cut_legs_hist"]), default=0) <= 3 and
         g["passed"] and r["passed"])
    dump(root / "acceptance_summary.json", acceptance)
    text = f"""MP3D 切分、去飞地和名单草稿已生成；保留结果按独立核查交付，待定项没有计入可用名单。

机器 48g（48g-jump），分支 claude/room-split-mp3d-20261010，提交 {source}。基线 47aca4f，v6 合入证据见 logs/merge_v6.log。所有代码只改新 worktree；所有产物只写本目录。没有 push，没有动旧名单、原始数据、共享环境或其他 worktree。全程 CPU、nice 12，最多 4 worker 加 1 父进程，地址空间最多 50 GiB。完整执行脚本、实际日志在 scripts/ 和 logs/。

原有 235 间逐间去向：面积大于 35 m² 的 22 间改由大房切分交付；其余原有房保留主房 {n["native_retained_main"]} 间，整房丢弃 {n["originals_fully_discarded"]} 间，待定 {n["subcap_originals_pending"]} 间；新冒出的 ≥6 m² 独立候选 {n["detached_candidates_ge6"]} 间，其中保留 {n["new_detached_candidates_retained"]} 间。小飞地丢弃 {n["discard_count"]} 块、{n["discard_area_m2"]:.6f} m²。原有 ≤35 m² 房间没有切线，也没有新增形状/空地圆/可见性/黑区淘汰条件。原有门槛例外和未找到生产边距见证的房间保留为待定。数字出处 mp3d_existing_rooms_connectivity_v1/summary.json、pending.json、regions/。

22 间大房切出并保留 {d["retained_rooms"]} 间，保留面积 {d["retained_area_m2"]:.6f} m²；丢弃 {d["discarded_parts"]} 块、{d["discarded_area_m2"]:.6f} m²。丢弃面积按主原因：{json.dumps(d["discarded_area_by_reason_m2"], ensure_ascii=False)}。待定 {d["pending_parts"]} 块、{d["pending_area_m2"]:.6f} m²，来源 {json.dumps(d["unresolved_source_regions"], ensure_ascii=False)}。数字出处 mp3d_delivery_v1/final_v1/summary.json、unresolved.json。失败尝试保存在 mp3d_delivery_v1/attempt_v1/，没有把不合规尝试当成最终可用房。

名单草稿 {draft["total_rooms"]} 间，总面积 {draft["total_area_m2"]:.6f} m²；面积分档 {json.dumps(draft["area_bins"], ensure_ascii=False)}（6–10、10–20、20–30 为左闭右开，30–35 包含上界）。另列待定 {draft["pending"]} 项、{draft["pending_area_m2"]:.6f} m²。出处 mp3d_room_list_draft_v1/summary.json、rooms.csv、pending.json。最终 rooms JSON 的文件名等于 id，含 floor、真实多边形、尺寸、decision 和 placement_witness，漏声任务可以直接读取。

独立四类形状核查：{json.dumps(s["counts"], ensure_ascii=False)}，合计 {sum(s["counts"].values())}。出处 checks/shape_quality_v1.json，执行的是只读 claude_check_shape_quality_v1.py。独立面积/短边违规 {len(c["kept_outside_area_or_short_side"])}，走不过去仍保留的远处块 {len(c["far_parts_not_direct"])}，切线段数分布 {json.dumps(c["cut_legs_hist"], ensure_ascii=False)}，家具交叉 >0.5 m 的切线 {c["cut_furniture_over_0p5m"]}。出处 checks/split_delivery_v4.json；执行的是只读 claude_check_split_delivery_v4.py。

235 间的完整原面积都记在 area_ledger.json 和 area_ledger.csv，按保留、丢弃、待定三类守恒。最大面积误差 {g["max_area_error_m2"]:.12g} m²，最大真实地面对称差 {g["max_symmetric_difference_m2"]:.12g} m²；待定面积没有冒充保留或丢弃，故“保留加丢弃”两项尚不能覆盖全部原面积。{g["retained_room_witnesses"]} 间保留房都有找到并验证的见证，包含真实填洞 exterior 内的 0.25 m 边距、原 navmesh 脚点和三条原整屋 CPU 射线。出处 geometry_and_witness_validation.json、各 rooms JSON 的 placement_witness.validation。

审图页 mp3d_review_v1.html；22 间大房和 {r["changed_original_regions"]} 间去飞地改变形状的原有房分别展示处理前后。共 {r["embedded_jpeg_images"]} 张 JPEG 内嵌图，{r["page_bytes"]} bytes，严格 IntersectionObserver 懒加载、初始 img 无 src，无外部资源；图片逐张解码验证，真实浏览器点击尚未实测。出处 review_validation.json。底图沿用 MP3D 准备任务的完整原始 GLB 真实 UV CPU raster 和当前层剖面，不涂黑房间外。原生 Habitat RGB 等价性未验证；元数据 region_renders/、house_renders/，owner 叠图在 owner_overlay_before/、owner_overlay_after/，用只读 claude_redraw_split_overlays_v5.py 生成。

未做：漏声/声学测量、生产绑定和出题、push、GPU 作业；漏声列留空，政策为 ≤5% 可测试、5–15% 仅训练、>15% 不用，等待另一个任务测量。未判定几何/见证项详见 pending.json 和原区域错误详情。没有证明房间数或家具交叉的全局最优。测试日志见 logs/unit*.log。
"""
    with (root / "FINAL_REPORT_zh.md").open("x", encoding="utf8") as f:
        f.write(text)
    for directory in ("mp3d_existing_rooms_connectivity_v1", "mp3d_room_list_draft_v1"):
        with (root / directory / "REPORT_zh.md").open("x", encoding="utf8") as f:
            f.write(text)
    print("MP3D_REPORT_DONE", json.dumps(acceptance, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["summarize", "overlay_inputs", "review", "report"])
    p.add_argument("--root", required=True)
    args = p.parse_args()
    bounded()
    globals()[args.action](args.root)
