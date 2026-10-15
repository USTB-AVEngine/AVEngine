"""Preserve failed native attempts and record why they were retained.

This helper never deletes media. Retention/deletion is an explicit operator policy,
separate from rendering and from the original failure that must reach the caller.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


def mark_new_attempt(directory: str | Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = {
        'version': 'v7-native-attempt-retention-v1',
        'attempt_id': uuid.uuid4().hex,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'policy': 'preserve_all_files',
    }
    (directory / 'native_attempt_retention.json').write_text(
        json.dumps(marker, indent=2) + '\n', encoding='utf-8')
    return marker


def retire_failed_media(directory: str | Path, marker: dict, error: Exception) -> dict:
    receipt = {
        **marker,
        'status': 'failed',
        'failure_type': type(error).__name__,
        'failure': str(error),
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'deleted_files': [],
    }
    try:
        (Path(directory) / 'native_attempt_failure.json').write_text(
            json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    except OSError as receipt_error:
        # Never replace the actual render failure with a retention I/O failure.
        receipt['receipt_write_error'] = str(receipt_error)
    return receipt
