"""Read-only catalogs the Studio serves: rooms, registries, review captures.

Base captures offered for re-rendering must come from main-branch commits
(owner rule after the natural-series animation regression). Capture receipts
do not all carry an explicit commit field, so the scan resolves the commit
token embedded in the review directory name against ``git rev-list`` of the
main branch; captures whose commit is known to be off main are excluded from
the default listing.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from avengine.studio.config import RoomPolicy

# Hex runs that could be an abbreviated commit sha. Pure-digit runs of 8+
# characters are dropped: those are dates (20260821), not commit prefixes.
_COMMIT_TOKEN_PATTERN = re.compile(r"[0-9a-f]{7,40}")


def load_registry_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _studio_room_status(record: dict, policy: RoomPolicy) -> tuple[str, str | None]:
    provider_id = str(record.get("provider_id", ""))
    room_id = str(record.get("room_id", "")).lower()
    if provider_id in policy.banned_provider_ids:
        return "banned", f"provider_id {provider_id!r} is banned by owner rule"
    for substring in sorted(policy.excluded_room_id_substrings):
        if substring and substring in room_id:
            return "excluded", f"room_id matches excluded substring {substring!r}"
    return "available", None


def annotate_room_registry(registry: dict, policy: RoomPolicy) -> dict:
    annotated = json.loads(json.dumps(registry))
    for record in annotated.get("records", []):
        status, reason = _studio_room_status(record, policy)
        record["studio_status"] = status
        if reason is not None:
            record["studio_status_reason"] = reason
    return annotated


def main_branch_commit_shas(
    repository_root: str | Path, *, branch: str = "main"
) -> frozenset[str] | None:
    """Full shas reachable from ``branch``, or None when git cannot answer."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-list", branch],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    shas = frozenset(line.strip() for line in completed.stdout.splitlines() if line.strip())
    return shas or None


def extract_commit_candidates(name: str) -> list[str]:
    tokens = _COMMIT_TOKEN_PATTERN.findall(name.lower())
    return [token for token in tokens if not (token.isdigit() and len(token) >= 8)]


def resolve_commit(
    name: str, main_shas: frozenset[str] | None
) -> tuple[str | None, bool | None]:
    """Return (commit_token, on_main) for a review directory name.

    on_main is None when git is unavailable or the name carries no
    letter-bearing token that could only be a commit.
    """

    candidates = extract_commit_candidates(name)
    if main_shas is not None:
        for token in candidates:
            if any(sha.startswith(token) for sha in main_shas):
                return token, True
    for token in candidates:
        if not token.isdigit():
            return token, (False if main_shas is not None else None)
    return None, None


def list_review_captures(
    review_root: str | Path, main_shas: frozenset[str] | None
) -> list[dict]:
    """Scan the external review root for visual base captures.

    A capture directory carries both frame_records.json (the per-frame
    position authority the dynamic-audio chain consumes) and a research
    receipt.
    """

    root = Path(review_root)
    entries: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        frame_records = child / "frame_records.json"
        receipt_path = child / "research_receipt.json"
        if not (frame_records.is_file() and receipt_path.is_file()):
            continue
        commit, on_main = resolve_commit(child.name, main_shas)
        entry: dict = {
            "name": child.name,
            "path": str(child),
            "commit": commit,
            "commit_on_main": on_main,
            "trusted": on_main is True,
        }
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            entry["receipt_error"] = f"{type(exc).__name__}: {exc}"
            entries.append(entry)
            continue
        capture = receipt.get("capture", {})
        entry.update(
            {
                "schema": receipt.get("schema"),
                "status": receipt.get("status"),
                "research_only": receipt.get("research_only"),
                "frame_count": capture.get("frame_count"),
                "modalities": capture.get("modalities"),
                "room_id": receipt.get("selected_room", {}).get("room_id"),
            }
        )
        entries.append(entry)
    return entries
