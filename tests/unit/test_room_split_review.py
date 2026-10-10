import csv
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image
from shapely.geometry import box, mapping

from tools.rooms.room_split_review import build_review_site
from tools.rooms.room_split_review.serve_review import ReviewStore, make_handler
from http.server import ThreadingHTTPServer

pytestmark = pytest.mark.fast_unit

P = [[0.2, 0, 0, 0], [0, 0, -0.2, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
I4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def room(rid, house, region, geom, source, **extra):
    return dict(id=rid, house=house, source_region=region, floor_id="F0", floor_y_m=0.0, floor_polygon_xz_m=mapping(geom),
                floor_area_m2=geom.area, short_side_m=min(geom.bounds[2] - geom.bounds[0], geom.bounds[3] - geom.bounds[1]),
                source=source, **extra)


@pytest.fixture()
def inputs(tmp_path):
    renders, regions, prior = tmp_path / "renders", tmp_path / "regions", tmp_path / "prior"
    for d in (renders, regions, prior):
        d.mkdir()
    Image.new("RGB", (200, 200), (180, 170, 150)).save(renders / "h1__Y000.png")
    (renders / "h1__Y000.json").write_text(json.dumps(dict(view=I4, projection=json.dumps(P), size_px=[200, 200], floor_y_m=0.0,
                                                            span_m=10.0, path=str(renders / "h1__Y000.png"))))
    kept = room("h1__R1__F0__S000", "h1", "R1", box(-4, -3, 0, 3), "new_cut")
    blocks = [dict(kept, decision="retain", room_type="bedroom"),
              dict(room("h1__R1__F0__S001", "h1", "R1", box(0, -3, 1, 3), "new_cut"), decision="discard", discard_reasons=["CORRIDOR"]),
              dict(room("h1__R1__F0__S002", "h1", "R1", box(-4, 3, 0, 4.5), "new_cut"), decision="retain")]
    (regions / "h1__R1.json").write_text(json.dumps(dict(house="h1", source_region="R1", requires_split=True,
                                                          source_floor_area_m2=36.0, blocks=blocks)))
    native = room("h1__R2__F0__E000", "h1", "R2", box(1.5, -3, 4.5, 0), "original")
    rooms_path = tmp_path / "rooms.json"
    rooms_path.write_text(json.dumps(dict(rooms=[kept, native])))
    leak = tmp_path / "leak.csv"
    with leak.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["room", "max_escape_fraction", "band"])
        w.writerow(["h1__R1__F0__S000", "0.01", "测试"])
        w.writerow(["h1__R2__F0__E000", "0.09", "只训练"])
    (prior / "h1__R2.json").write_text(json.dumps(dict(verdict="skip", note="太小", author="smy")))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(dict(title="测试页", build_id="t1", families=[dict(
        name="HM3D", rooms=str(rooms_path), region_dirs=[str(regions)], house_render_dirs=[str(renders)],
        leakage_csvs=[str(leak)], prior_verdict_dir=str(prior))])))
    return tmp_path, cfg


def test_build_site_cards_images_and_refuses_overwrite(inputs):
    tmp, cfg = inputs
    out = tmp / "site"
    assert build_review_site.main(["--config", str(cfg), "--out", str(out), "--workers", "1"]) == 0
    data = json.loads((out / "site_data.json").read_text())
    cards = {c["card"]: c for c in data["cards"]}
    split, native = cards["hm3d/h1/R1"], cards["hm3d/h1/R2"]
    assert split["split"] and not native["split"]
    assert [r["id"] for r in split["rooms"]] == ["h1__R1__F0__S000"] and split["rooms"][0]["type"] == "卧室"
    assert {(d["id"], d["kind"]) for d in split["dropped"]} == {("h1__R1__F0__S001", "discard"), ("h1__R1__F0__S002", "unresolved")}
    assert native["prior"]["verdict"] == "skip" and native["rooms"][0]["band"] == "只训练"
    for c in data["cards"]:
        assert len(c["images"]) == 1
        size = Image.open(out / c["images"][0]).size
        assert size == ((720, 720) if c is split else (560, 560))
    page = (out / "index.html").read_text()
    assert "__DATA__" not in page and '"build_id": "t1"' in page
    receipt = json.loads((out / "build_receipt.json").read_text())
    assert receipt["room_total"] == 2 and receipt["families"]["HM3D"]["rooms_without_region_file"] == ["h1__R2__F0__E000"]
    with pytest.raises(SystemExit):
        build_review_site.main(["--config", str(cfg), "--out", str(out), "--workers", "1"])


def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_saves_validates_and_keeps_newer(inputs):
    tmp, cfg = inputs
    out = tmp / "site"
    build_review_site.main(["--config", str(cfg), "--out", str(out), "--workers", "1"])
    store = ReviewStore(out, out / "feedback" / "review_feedback.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert call(base + "/")[0] == 200 and call(base + "/api/feedback") == (200, b"{}")
        rec = dict(assessment="bad", reason_codes=["bad_cut"], note="切线穿沙发", reviewer="smy", updated_at="2026-10-10T02:00:00Z")
        assert call(base + "/api/feedback", dict(id="h1__R1__F0__S000", feedback=rec))[0] == 200
        assert call(base + "/api/feedback", dict(id="nope", feedback=rec))[0] == 400
        assert call(base + "/api/feedback", dict(id="h1__R1__F0__S000", feedback=dict(rec, reason_codes=["x"])))[0] == 400
        old = dict(rec, assessment="ok", updated_at="2026-10-10T01:00:00Z")
        assert call(base + "/api/feedback/bulk", dict(items={"h1__R1__F0__S000": old, "dropped:hm3d/h1/R1": dict(old, reason_codes=[])}))[0] == 200
        saved = json.loads(call(base + "/api/feedback")[1])
        assert saved["h1__R1__F0__S000"]["assessment"] == "bad" and saved["dropped:hm3d/h1/R1"]["assessment"] == "ok"
        assert call(base + "/feedback/review_feedback.json")[0] == 404
        assert call(base + "/../cfg.json")[0] == 404
        assert len((out / "feedback" / "review_feedback.log.jsonl").read_text().splitlines()) == 3
    finally:
        server.shutdown()
        server.server_close()
