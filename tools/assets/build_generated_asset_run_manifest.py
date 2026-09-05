"""Write an AVEngine-readable generated-asset run index.

Formal dataset admission stays false. Human acceptance is a separate field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "avengine_generated_asset_run_index_v1"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _nonempty_png_dir(path: Path) -> str | None:
    if path.is_dir() and any(path.glob("*.png")):
        return str(path)
    return None


def visual_review_artifacts(review: Path) -> dict[str, Any]:
    """Describe stills, turntable and action clips actually written under review/."""

    review = Path(review)
    manifest_path = review / "review_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read review manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SystemExit(f"review manifest must be an object: {manifest_path}")
        action_dir = manifest.get("action_dir")
        turntable_dir = manifest.get("turntable_dir")
        return {
            "front": str(review / "front.png") if (review / "front.png").is_file() else None,
            "side": str(review / "side.png") if (review / "side.png").is_file() else None,
            "back": str(review / "back.png") if (review / "back.png").is_file() else None,
            "turntable_dir": (
                str(Path(turntable_dir))
                if isinstance(turntable_dir, str) and Path(turntable_dir).is_dir()
                else _nonempty_png_dir(review / "turntable")
            ),
            "action_dir": (
                str(Path(action_dir))
                if isinstance(action_dir, str) and Path(action_dir).is_dir() and any(Path(action_dir).glob("*.png"))
                else _nonempty_png_dir(review / "action")
            ),
        }
    return {
        "front": str(review / "front.png") if (review / "front.png").is_file() else None,
        "side": str(review / "side.png") if (review / "side.png").is_file() else None,
        "back": str(review / "back.png") if (review / "back.png").is_file() else None,
        "turntable_dir": _nonempty_png_dir(review / "turntable"),
        "action_dir": _nonempty_png_dir(review / "action"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace {output}")
    fail_path = root / "failed" / "failed_assets.json"
    failed = json.loads(fail_path.read_text(encoding="utf-8")) if fail_path.is_file() else []
    assets = []
    catalog = (
        ("siamese", "cat", "quadruped", "articulated_animal"),
        ("jack_russell", "dog", "quadruped", "articulated_animal"),
        ("human_male", "human", "biped", "articulated_human"),
    )
    failed_names = {item.get("asset") for item in failed} if isinstance(failed, list) else set()
    for name, category, body, entity in catalog:
        mesh = _file(root / "mesh" / f"{name}.glb")
        rig = _file(root / "rig" / f"{name}_rig.glb")
        oriented = _file(root / "rig" / f"{name}_oriented.glb")
        animated = _file(root / "anim" / f"{name}_animated.glb") or _file(
            root / "anim" / f"{name}_probe.glb"
        )
        review = root / "review" / name
        assets.append({
            "asset_id": f"generated_{name}_visual_review_20260905_v1",
            "category": category,
            "body_class": body,
            "entity_class": entity,
            "admission_state": "research",
            "formal_dataset_registration_authorized": False,
            "human_acceptance": None,
            "failed": name in failed_names,
            "mesh": mesh,
            "rig": rig,
            "oriented_rig": oriented,
            "animated": animated,
            "emitter_anchors": _file(root / "anchors" / f"{name}_emitter.json"),
            "visual_review": visual_review_artifacts(review),
        })
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkout": str(args.checkout.resolve()),
        "run_root": str(root),
        "transform_profile": str(
            args.checkout.resolve() / "examples/assets/pixal3d_transform_profile_v1.json"
        ),
        "formal_dataset_registration_authorized": False,
        "human_acceptance": None,
        "failed_assets": failed,
        "assets": assets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"RUN_INDEX_OK {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
