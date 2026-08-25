#!/usr/bin/env python3
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw

WS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
CELL = int(sys.argv[3]) if len(sys.argv) > 3 else 420
LABEL = 26

batch = json.loads((WS / "flux/flux2_batch_manifest.json").read_text())
rows = []
for c in batch["candidates"]:
    t = c["profile_schema_id"].replace("audio_playback_", "").replace("_product_view_v1", "")
    v = list(c["sampled_attributes"].values())[0]
    rows.append((t, v, WS / "flux" / c["candidate"]["path"], c["execution_job_id"]))
rows.sort()

cols = 3
n = len(rows)
nrows = (n + cols - 1) // cols
sheet = Image.new("RGB", (cols * CELL, nrows * (CELL + LABEL)), (24, 24, 26))
draw = ImageDraw.Draw(sheet)
for i, (t, v, path, jid) in enumerate(rows):
    img = Image.open(path).convert("RGB")
    img.thumbnail((CELL, CELL))
    cx = (i % cols) * CELL + (CELL - img.width) // 2
    cy = (i // cols) * (CELL + LABEL) + LABEL + (CELL - img.height) // 2
    sheet.paste(img, (cx, cy))
    draw.text(((i % cols) * CELL + 6, (i // cols) * (CELL + LABEL) + 6),
              f"{t}  {v}  [{jid[-6:]}]", fill=(235, 235, 235))
sheet.save(OUT)
print(OUT, sheet.size)
