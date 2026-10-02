"""Copy authorized read-only manual JSON references; never alter smy's files."""

import argparse
import json
import subprocess
from pathlib import Path

PENDING_IDS = {
    "00758-R15-1",
    "00758-R15-2",
    "00802-R16-1",
    "00802-R16-2",
    "00808-R13-1",
    "00848-R0-1",
    "00848-R0-2",
    "00880-R1-2",
    "00891-R8-1",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    paths = subprocess.check_output(
        [
            "sudo",
            "-n",
            "find",
            str(a.source),
            "-maxdepth",
            "2",
            "-type",
            "f",
            "-name",
            "*.json",
        ],
        text=True,
    ).splitlines()
    records = []
    for name in sorted(paths):
        data = json.loads(
            subprocess.check_output(["sudo", "-n", "cat", name], text=True)
        )
        sid = Path(name).stem
        records.append(
            dict(
                source=name,
                coordinate_status=(
                    "pending_original_bbox"
                    if sid in PENDING_IDS
                    else "recorded_bbox_proxy"
                ),
                data=data,
            )
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            dict(
                records=records,
                source_readme=str(a.source / "README.md"),
                limitation="No mask/polygon ground truth in the handed-off 43 directories; 9 bboxes remain the original room frame.",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                total=len(records),
                pending=sum(
                    r["coordinate_status"] == "pending_original_bbox" for r in records
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
