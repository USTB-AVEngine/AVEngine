"""Per-room floor reference: the measured UE z of the walkable floor.

Why
---
On 2026-09-03 the Apartment scene config declared ``ground_z_ue_cm: 0.0`` while
the cooked map's floor sits about 27 cm higher (depth-derived camera height
1.201 m at camera z 147.1 cm; navmesh points at z 28-32 cm).  Every render
consumer inherits that constant: camera z, actor z, the absolute camera
heights of the clearance table and the height semantics of its target band.
Dogs were rendered sunk into the floor and the camera stood 1.20 m above it
instead of 1.47 m.  The number was a hand-written constant that nothing
checked.

What this module fixes
----------------------
A room's floor height is measured once in the engine (downward line traces
at the solver's own camera points and at random walkable cells, see
``measure_qa_v3_floor_z.py``) and stored as a room product next to the
clearance table and the walkable grid.  Scene configs point at it, and the
scene loader refuses to hand out render facts whose ``ground_z_ue_cm`` does
not equal the measured value.  Nothing here changes a question type or a
generator; it only makes the floor a measured fact instead of a guess.

Boundary
--------
Research placeholder thresholds; the reference is a room product with status
``measured`` or ``inconsistent`` and never a dataset admission statement.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "qa_v3_floor_reference_v1"
INDEX_NAME = "floor_reference.json"
ROWS_NAME = "floor_trace_rows.json"
STATUS_MEASURED = "measured"
STATUS_INCONSISTENT = "inconsistent"
# 配置里写的 ground_z_ue_cm 与实测中位数允许的差:半厘米,超过就拒绝载入。
MATCH_TOLERANCE_CM = 0.5


class FloorReferenceError(ValueError):
    """The floor reference is missing, malformed, tampered or unusable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_floor_hits(hit_z_cm: Sequence[float], *, total_traces: int,
                         within_cm: float = 2.0) -> dict[str, Any]:
    """Robust statistics of the traced floor heights (all in UE cm).

    The median is the floor height; the spread numbers say whether the room
    has one floor level under the sampled points.  Outliers (a rug, a table
    top over a navmesh cell) are counted, never averaged in.
    """
    values = sorted(float(v) for v in hit_z_cm if math.isfinite(float(v)))
    if total_traces <= 0:
        raise FloorReferenceError("no traces were attempted")
    if not values:
        return {"hit_count": 0, "trace_count": int(total_traces), "hit_fraction": 0.0}
    n = len(values)
    median = values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])

    def quantile(q: float) -> float:
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    deviations = sorted(abs(v - median) for v in values)
    mad = deviations[n // 2] if n % 2 else 0.5 * (deviations[n // 2 - 1] + deviations[n // 2])
    within = sum(1 for v in values if abs(v - median) <= within_cm)
    return {
        "trace_count": int(total_traces),
        "hit_count": n,
        "hit_fraction": n / float(total_traces),
        "median_cm": median,
        "mean_cm": sum(values) / n,
        "mad_cm": mad,
        "min_cm": values[0],
        "p05_cm": quantile(0.05),
        "p95_cm": quantile(0.95),
        "max_cm": values[-1],
        "within_cm": float(within_cm),
        "within_fraction": within / float(n),
    }


def write_floor_reference(output_dir: str | Path, *, scene_id: str, native_map: str,
                          method: dict[str, Any], summary: Mapping[str, Any],
                          rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float],
                          code: Mapping[str, Any] | None = None,
                          extra: Mapping[str, Any] | None = None) -> Path:
    """Write ``floor_reference.json`` + rows.  Refuses to overwrite anything."""
    root = Path(output_dir)
    if root.exists():
        raise FloorReferenceError(f"refusing to overwrite existing floor reference: {root}")
    if not str(native_map).startswith("/Game/"):
        raise FloorReferenceError("native_map must be a /Game package path")
    hit_fraction = float(summary.get("hit_fraction", 0.0))
    within_fraction = float(summary.get("within_fraction", 0.0))
    consistent = (summary.get("hit_count", 0) >= int(thresholds["min_hits"])
                  and hit_fraction >= float(thresholds["min_hit_fraction"])
                  and within_fraction >= float(thresholds["min_within_fraction"]))
    status = STATUS_MEASURED if consistent else STATUS_INCONSISTENT
    root.mkdir(parents=True)
    rows_path = root / ROWS_NAME
    rows_path.write_text(json.dumps({"schema": SCHEMA, "scene_id": scene_id,
                                     "rows": list(rows)}, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    index = {
        "schema": SCHEMA,
        "status": status,
        "research_status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": ("measured floor height of one cooked map under the solver's "
                           "camera points and walkable cells; a render input, not "
                           "question admission"),
        "scene_id": str(scene_id),
        "native_map": str(native_map),
        "ground_z_ue_cm": (float(summary["median_cm"]) if status == STATUS_MEASURED else None),
        "method": dict(method),
        "summary": dict(summary),
        "thresholds": dict(thresholds),
        "rows": {"path": ROWS_NAME, "sha256": _sha256_file(rows_path), "count": len(rows)},
        "code": dict(code or {}),
    }
    if extra:
        index.update(dict(extra))
    (root / INDEX_NAME).write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    return root


@dataclass(frozen=True)
class FloorReference:
    root: Path
    index: dict

    @classmethod
    def load(cls, path: str | Path) -> "FloorReference":
        root = Path(path)
        if root.is_file():
            root = root.parent
        index_path = root / INDEX_NAME
        if not index_path.is_file():
            raise FloorReferenceError(f"floor reference index missing: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("schema") != SCHEMA:
            raise FloorReferenceError(
                f"{index_path}: schema {index.get('schema')!r} is not {SCHEMA}")
        rows = index.get("rows") or {}
        rows_path = root / str(rows.get("path", ROWS_NAME))
        if not rows_path.is_file():
            raise FloorReferenceError(f"floor trace rows missing: {rows_path}")
        if _sha256_file(rows_path) != rows.get("sha256"):
            raise FloorReferenceError(f"floor trace rows do not match their sha256: {rows_path}")
        if index.get("status") != STATUS_MEASURED:
            raise FloorReferenceError(
                f"{index_path}: floor reference status is {index.get('status')!r}; only a "
                f"{STATUS_MEASURED!r} reference may feed render facts")
        value = index.get("ground_z_ue_cm")
        if value is None or not math.isfinite(float(value)):
            raise FloorReferenceError(f"{index_path}: ground_z_ue_cm is missing or not finite")
        return cls(root=root, index=index)

    @property
    def scene_id(self) -> str:
        return str(self.index["scene_id"])

    @property
    def native_map(self) -> str:
        return str(self.index["native_map"])

    @property
    def ground_z_ue_cm(self) -> float:
        return float(self.index["ground_z_ue_cm"])

    @property
    def status(self) -> str:
        return str(self.index["status"])

    def matches(self, declared_ground_z_cm: float, tolerance_cm: float = MATCH_TOLERANCE_CM) -> bool:
        value = float(declared_ground_z_cm)
        return math.isfinite(value) and abs(value - self.ground_z_ue_cm) <= tolerance_cm

    @property
    def identity(self) -> dict[str, Any]:
        summary = self.index.get("summary") or {}
        return {"path": str(self.root), "schema": SCHEMA, "status": self.status,
                "scene_id": self.scene_id, "native_map": self.native_map,
                "ground_z_ue_cm": self.ground_z_ue_cm,
                "method": (self.index.get("method") or {}).get("kind"),
                "trace_count": summary.get("trace_count"),
                "hit_count": summary.get("hit_count"),
                "p05_cm": summary.get("p05_cm"), "p95_cm": summary.get("p95_cm"),
                "rows_sha256": (self.index.get("rows") or {}).get("sha256"),
                "code_revision": (self.index.get("code") or {}).get("revision")}


def floor_reference_from_config(value: Any) -> FloorReference:
    """``floor_reference`` in a scene config is the path of the room product."""
    if isinstance(value, Mapping):
        path = value.get("path")
    else:
        path = value
    if not isinstance(path, str) or not path:
        raise FloorReferenceError("floor_reference must be the path of a measured floor reference")
    return FloorReference.load(path)
