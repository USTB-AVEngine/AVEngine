import csv
import json

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, box, mapping

from tools.rooms.room_split_review import subset_config
from tools.rooms.room_usability import objects, rules

pytestmark = pytest.mark.fast_unit


def room(rid, region, geom, floor_y=0.0):
    return dict(id=rid, house="h1", source_region_id=region, floor_y_m=floor_y, floor_polygon_xz_m=mapping(geom))


def obj(region, x, z, ylo=0.0, yhi=0.9):
    return dict(cat="chair", role="blocker", region=region, cx=x, cz=z, ylo=ylo, yhi=yhi, foot=0.3)


def test_region_objects_go_to_the_nearest_room_and_count_inside_floor_holes():
    with_hole = box(0, 0, 4, 4).difference(box(1, 1, 2, 2))
    rooms = [room("A", 3, with_hole), room("B", 3, box(4, 0, 8, 4)), room("C", 5, box(0, 5, 4, 9))]
    items = [obj(3, 1.5, 1.5), obj(3, 6, 2), obj(3, 4.5, 4.6), obj(3, 10, 2), obj(5, 2, 7), obj(3, 1.5, 3, ylo=3.0, yhi=3.8),
             obj(5, 6, 2)]
    got = objects.assign_by_region(rooms, items)
    assert [(o["cx"], o["cz"]) for o in got["A"]] == [(1.5, 1.5)]  # standing in the floor hole still counts
    assert [(o["cx"], o["cz"]) for o in got["B"]] == [(6, 2), (4.5, 4.6)]  # 0.6 m outside B, 0.78 m from A
    assert [(o["cx"], o["cz"]) for o in got["C"]] == [(2, 7)]  # region 5 never lands in B; upper storey and 2 m away drop


def test_kujiale_footprint_is_the_rotated_rectangle_of_the_slice():
    legs = MultiPolygon([box(x, z, x + 0.05, z + 0.05) for x in (1.0, 2.15) for z in (1.0, 1.75)])
    rooms = [dict(room("K", 0, box(0, 0, 4, 4)), floor_y_m=0.0)]
    furniture = {"0.0": [dict(category="dining_table", geometry=mapping(legs)),
                         dict(category="sofa", geometry=mapping(box(4.1, 1, 4.3, 2))),
                         dict(category="sofa", geometry=mapping(box(4.5, 1, 4.9, 2)))]}
    got = objects.assign_kujiale(rooms, furniture)["K"]
    assert [o["cat"] for o in got] == ["dining table", "sofa"]  # 0.1 m outside joins, 0.5 m outside does not
    assert got[0]["foot"] == pytest.approx(1.2 * 0.8) and got[0]["role"] == "blocker"
    assert objects.table(rooms, {"K": got})[0] == dict(room="K", cat="dining table", role="blocker", foot=0.96, top="")


def write_inputs(tmp_path):
    specs = [  # id, area, room_type, objects (cat, role, foot), smy verdict
        ("r1", 12.0, "bedroom", [("bed", "blocker", 3.0), ("wardrobe", "blocker", 1.0)], ""),
        ("r2", 9.0, "bathroom", [("sink", "blocker", 0.5), ("cabinet", "blocker", 0.5)], ""),
        ("r3", 15.0, "bedroom", [("toilet", "blocker", 0.4), ("bed", "blocker", 3.0)], ""),
        ("r4", 8.0, "", [("toilet", "blocker", 0.4), ("cabinet", "blocker", 0.5), ("table", "blocker", 0.5)], ""),
        ("r5", 7.0, "bedroom", [("bed", "blocker", 3.0), ("desk", "blocker", 0.6)], ""),
        ("r6", 20.0, "living", [("sofa", "blocker", 2.0)] + [("chair", "blocker", 0.1)] * 5, ""),
        ("r7", 20.0, "unknown", [("furniture", "review", 1.0), ("gym equipment", "review", 1.5)], ""),
        ("r8", 6.5, "bedroom", [], "ok"),
        ("r9", 20.0, "bedroom", [("bed", "blocker", 3.0), ("wardrobe", "blocker", 1.0)], "bad"),
    ]
    records = [dict(id=i, floor_area_m2=a, room_type=t) for i, a, t, _, _ in specs]
    (tmp_path / "list.json").write_text(json.dumps(dict(schema="room_list_v1", rooms=records)))
    site = dict(cards=[dict(card=f"mp3d/h1/R{n}", family="MP3D", split=False,
                            rooms=[dict(id=i, no=1, area=round(a, 2), type="", band="测试")])
                       for n, (i, a, _, _, _) in enumerate(specs)])
    (tmp_path / "site_data.json").write_text(json.dumps(site))
    (tmp_path / "config.json").write_text(json.dumps(dict(families=[dict(name="MP3D", rooms=str(tmp_path / "list.json"))])))
    (tmp_path / "feedback.json").write_text(json.dumps(dict(feedback={i: dict(assessment=v) for i, _, _, _, v in specs if v})))
    with (tmp_path / "objects.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=objects.COLUMNS)
        w.writeheader()
        for i, _, _, objs, _ in specs:
            w.writerows(dict(room=i, cat=c, role=r, foot=f, top=1.0) for c, r, f in objs)
    return tmp_path


def test_rules_drop_bathrooms_small_and_bare_rooms_but_keep_the_reviewer_verdict(tmp_path):
    src = write_inputs(tmp_path)
    out = tmp_path / "out"
    assert rules.main(["--site-data", str(src / "site_data.json"), "--config", str(src / "config.json"),
                       "--feedback", str(src / "feedback.json"), "--objects", str(src / "objects.csv"), "--out", str(out)]) == 0
    rows = {r["id"]: r for r in csv.DictReader((out / "room_decisions.csv").open(encoding="utf-8-sig"))}
    assert {k: (r["decision"], r["rules_fired"]) for k, r in rows.items()} == {
        "r1": ("keep", ""), "r2": ("drop", "bathroom"), "r3": ("keep", ""), "r4": ("drop", "bathroom"),
        "r5": ("drop", "small"), "r6": ("drop", "bare"), "r7": ("keep", ""), "r8": ("keep", "small|bare"),
        "r9": ("drop", "")}
    assert rows["r8"]["basis"] == rows["r9"]["basis"] == "smy"
    new = json.loads((out / "mp3d_room_list_usability.json").read_text())
    assert [r["id"] for r in new["rooms"]] == ["r1", "r3", "r7", "r8"]
    dropped = {d["id"]: d for d in new["usability_filter"]["dropped"]}
    assert dropped["r6"]["rules"] == ["bare"] and dropped["r9"]["smy"] == "bad"
    s = json.loads((out / "summary.json").read_text())["families"]["MP3D"]
    assert (s["kept"], s["dropped_by_smy"], s["dropped_by_rules"]) == (4, 1, 4)
    assert s["rules_vs_smy"] == dict(bad_caught=0, bad_total=1, ok_hit=1, ok_total=1,
                                     ok_hit_by_rule=dict(bathroom=0, small=1, bare=1))
    with pytest.raises(SystemExit):
        rules.main(["--site-data", str(src / "site_data.json"), "--config", str(src / "config.json"),
                    "--feedback", str(src / "feedback.json"), "--objects", str(src / "objects.csv"), "--out", str(out)])


def test_subset_config_keeps_whole_chosen_cards(tmp_path):
    src = write_inputs(tmp_path)
    rules.main(["--site-data", str(src / "site_data.json"), "--config", str(src / "config.json"),
                "--feedback", str(src / "feedback.json"), "--objects", str(src / "objects.csv"), "--out", str(tmp_path / "lists")])
    spec = dict(title="第二轮", build_id="r2", lead_html="<p>x</p>", cards={"mp3d/h1/R0": "bare", "mp3d/h1/R2": "odd_shape"})
    (tmp_path / "round.json").write_text(json.dumps(spec, ensure_ascii=False))
    out = tmp_path / "round"
    assert subset_config.main(["--config", str(src / "config.json"), "--lists", str(tmp_path / "lists"), "--round",
                               str(tmp_path / "round.json"), "--site-data", str(src / "site_data.json"), "--out", str(out)]) == 0
    cfg = json.loads((out / "review_config.json").read_text())
    assert (cfg["title"], cfg["build_id"], cfg["lead_html"]) == ("第二轮", "r2", "<p>x</p>")
    rooms = json.loads(open(cfg["families"][0]["rooms"]).read())
    assert [r["id"] for r in rooms["rooms"]] == ["r1", "r3"]
    assert rooms["round_selection"]["cards"] == spec["cards"]
    spec["cards"]["mp3d/h9/R0"] = "bare"
    (tmp_path / "round.json").write_text(json.dumps(spec))
    with pytest.raises(SystemExit):
        subset_config.main(["--config", str(src / "config.json"), "--lists", str(tmp_path / "lists"), "--round",
                            str(tmp_path / "round.json"), "--site-data", str(src / "site_data.json"), "--out", str(tmp_path / "r2")])


def test_box_object_measures_the_axis_aligned_footprint():
    tri = np.array([[[0.0, 0.0, 0.0], [2.0, 1.5, 0.0], [0.0, 0.2, 0.5]]])
    o = objects.box_object("table", "blocker", 4, tri)
    assert (o["cx"], o["cz"], o["foot"], o["ylo"], o["yhi"], o["region"]) == (1.0, 0.25, 1.0, 0.0, 1.5, 4)
