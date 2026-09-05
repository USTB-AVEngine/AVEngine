"""Build a standalone HTML visual-review page for generated Pixal3D assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "avengine_generated_asset_visual_review_page_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace {output}")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    fail_path = root / "failed" / "failed_assets.json"
    failed = []
    if fail_path.is_file():
        failed = json.loads(fail_path.read_text(encoding="utf-8"))
    assets = []
    review_root = root / "review"
    names = []
    if review_root.is_dir():
        names = sorted(path.name for path in review_root.iterdir() if path.is_dir())
    for fallback in ("siamese", "jack_russell", "human_male"):
        if fallback not in names:
            names.append(fallback)
    for name in names:
        review = root / "review" / name
        record = {
            "name": name,
            "mesh": (root / "mesh" / f"{name}.glb").is_file(),
            "rig": (root / "rig" / f"{name}_rig.glb").is_file(),
            "front": (review / "front.png").is_file(),
            "side": (review / "side.png").is_file(),
            "back": (review / "back.png").is_file(),
            "turntable": sorted(p.name for p in (review / "turntable").glob("*.png")) if (review / "turntable").is_dir() else [],
            "action": sorted(p.name for p in (review / "action").glob("*.png")) if (review / "action").is_dir() else [],
        }
        assets.append(record)
    html = ["<!doctype html><meta charset=utf-8><title>Generated asset visual review</title>"]
    html.append("<style>body{font:16px/1.4 sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}img{max-width:32%;height:auto}section{margin:24px 0;padding:12px 0;border-top:1px solid #ddd}</style>")
    html.append("<h1>Generated asset visual review</h1>")
    html.append("<p>Formal dataset admission: <b>false</b>. Human acceptance is recorded separately.</p>")
    html.append(f"<p>Run: <code>{root}</code></p>")
    if failed:
        html.append("<h2>Failed assets</h2><ul>")
        items = failed if isinstance(failed, list) else failed.get("assets", [])
        for item in items:
            html.append(f"<li><pre>{json.dumps(item, ensure_ascii=False)}</pre></li>")
        html.append("</ul>")
    for asset in assets:
        html.append(f"<section><h2>{asset['name']}</h2>")
        html.append(f"<p>mesh={asset['mesh']} rig={asset['rig']}</p>")
        for view in ("front", "side", "back"):
            rel = Path("review") / asset["name"] / f"{view}.png"
            if (root / rel).is_file():
                html.append(f"<figure><img src='{rel}'><figcaption>{view}</figcaption></figure>")
        html.append("</section>")
    html.append("<h2>Run manifest</h2><pre>")
    html.append(json.dumps({k: manifest.get(k) for k in ("status", "formal_dataset_registration_authorized", "human_acceptance", "gpu")}, indent=2))
    html.append("</pre>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(html) + "\n", encoding="utf-8")
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "schema": SCHEMA,
        "formal_dataset_registration_authorized": False,
        "page": str(output),
        "assets": assets,
        "failed": failed,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"REVIEW_PAGE_OK {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
