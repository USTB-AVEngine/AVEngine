#!/usr/bin/env python3
"""Exact, crash-safe raw spool used by the shared-room full75 capture adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np

CONTRACT_PATH = Path(__file__).with_name("run_strict_two_human_full75_room_batch.py")
SPEC = importlib.util.spec_from_file_location(
    "strict2h_room_batch_contract", CONTRACT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import batch contract: {CONTRACT_PATH}")
CONTRACT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACT
SPEC.loader.exec_module(CONTRACT)

RAW_MEMMAP_CONTRACT = CONTRACT.RAW_MEMMAP_CONTRACT
RAW_MEMMAP_TOTAL_BYTES = CONTRACT.RAW_MEMMAP_TOTAL_BYTES
RAW_RECEIPT_SCHEMA = CONTRACT.RAW_RECEIPT_SCHEMA
FRAME_COUNT = CONTRACT.FRAME_COUNT
HEIGHT = CONTRACT.HEIGHT
WIDTH = CONTRACT.WIDTH

PASS_FILE = {
    "normal_depth": "normal_depth_m.f16le",
    "target_only_source1_depth": "target_only_source1_depth_m.f16le",
    "target_only_source2_depth": "target_only_source2_depth_m.f16le",
    "normal_object_ids": "normal_object_ids.u32le",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RawSpoolWriter:
    """Preallocate and sequentially fill the four exact raw memmaps.

    ``RAW_READY.json`` is the only readiness authority.  It is published after
    all data files and metadata have been flushed and fsynced.  A crash before
    that marker leaves a preserved, non-resumable partial attempt.
    """

    def __init__(self, attempt_root: Path) -> None:
        _require(attempt_root.is_dir(), "attempt root must already exist")
        self.root = attempt_root / "raw_spool"
        self.root.mkdir(exist_ok=False)
        self.rgb_root = self.root / "rgb_frames"
        self.rgb_root.mkdir()
        self._streams: dict[str, Any] = {}
        self._maps: dict[str, np.memmap] = {}
        self._hashers: dict[str, Any] = {}
        self._next_frame = {key: 0 for key in PASS_FILE}
        self._closed = False
        for pass_id, name in PASS_FILE.items():
            contract = RAW_MEMMAP_CONTRACT[name]
            path = self.root / name
            stream = path.open("x+b")
            stream.truncate(int(contract["size_bytes"]))
            stream.flush()
            os.fsync(stream.fileno())
            array = np.memmap(
                stream,
                mode="r+",
                dtype=np.dtype(contract["dtype"]),
                shape=tuple(contract["shape"]),
                order="C",
            )
            self._streams[pass_id] = stream
            self._maps[pass_id] = array
            self._hashers[pass_id] = hashlib.sha256()
        _fsync_directory(self.root)

    def write_frame(self, pass_id: str, frame_index: int, value: Any) -> None:
        _require(not self._closed, "raw spool is already closed")
        _require(pass_id in PASS_FILE, f"unknown raw pass: {pass_id}")
        _require(
            frame_index == self._next_frame[pass_id],
            f"{pass_id}: frame order drift at {frame_index}",
        )
        contract = RAW_MEMMAP_CONTRACT[PASS_FILE[pass_id]]
        frame = np.asarray(value, dtype=np.dtype(contract["dtype"]), order="C")
        _require(frame.shape == (HEIGHT, WIDTH), f"{pass_id}: frame shape drift")
        _require(frame.flags.c_contiguous, f"{pass_id}: frame must be contiguous")
        self._maps[pass_id][frame_index] = frame
        self._hashers[pass_id].update(frame.tobytes(order="C"))
        self._next_frame[pass_id] += 1

    def rgb_path(self, frame_index: int) -> Path:
        _require(0 <= frame_index < FRAME_COUNT, "RGB frame index drift")
        return self.rgb_root / f"frame_{frame_index:06d}.png"

    def write_metadata(self, name: str, value: Mapping[str, Any]) -> Path:
        _require(not self._closed, "raw spool is already closed")
        _require(
            name
            in {
                "runtime_readbacks.json",
                "runtime_asset_readbacks.json",
                "normal_object_id_descriptors.json",
                "capture_context.json",
            },
            f"unapproved raw metadata name: {name}",
        )
        path = self.root / name
        _require(not path.exists(), f"raw metadata already exists: {name}")
        _atomic_json(path, value)
        return path

    def _flush_data(self) -> None:
        _require(
            self._next_frame == {key: FRAME_COUNT for key in PASS_FILE},
            f"raw frame closure failed: {self._next_frame}",
        )
        for pass_id in PASS_FILE:
            self._maps[pass_id].flush()
            self._streams[pass_id].flush()
            os.fsync(self._streams[pass_id].fileno())
        _fsync_directory(self.root)

    def publish_ready(
        self,
        *,
        batch_request_sha256: str,
        episode_id: str,
        input_binding_sha256: str,
        teardown: Mapping[str, Any],
        motion_realism_release_qualified: bool,
    ) -> Path:
        _require(not self._closed, "raw spool is already closed")
        self._flush_data()
        expected_rgb_names = [f"frame_{index:06d}.png" for index in range(FRAME_COUNT)]
        observed_rgb = sorted(self.rgb_root.glob("frame_*.png"))
        _require(
            [path.name for path in observed_rgb] == expected_rgb_names,
            "raw RGB sequence is incomplete",
        )
        for path in observed_rgb:
            _require(path.is_file() and path.stat().st_size > 0, f"empty RGB: {path}")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(self.rgb_root)

        required_metadata = {
            "runtime_readbacks.json",
            "runtime_asset_readbacks.json",
            "normal_object_id_descriptors.json",
            "capture_context.json",
        }
        _require(
            required_metadata == {path.name for path in self.root.glob("*.json")},
            "raw metadata closure failed before READY",
        )
        for name in required_metadata:
            with (self.root / name).open("rb") as stream:
                os.fsync(stream.fileno())
        metadata_records = {
            name: {
                "path": str((self.root / name).resolve()),
                "size_bytes": (self.root / name).stat().st_size,
                "sha256": _sha256(self.root / name),
            }
            for name in sorted(required_metadata)
        }

        required_teardown = {
            "actors_destroyed",
            "segmentation_terminated",
            "prior_stable_names_absent",
            "prior_actor_handles_absent",
            "prior_stable_actor_names_absent",
            "prior_proxy_descriptors_absent",
            "proxy_filters_cleared",
            "show_only_list_cleared",
        }
        _require(required_teardown.issubset(teardown), "teardown gates are missing")
        _require(
            all(teardown[key] is True for key in required_teardown), "teardown failed"
        )
        _require(
            teardown.get("remaining_controlled_actor_handle_count") == 0
            and teardown.get("remaining_controlled_stable_name_count") == 0
            and teardown.get("remaining_controlled_proxy_descriptor_count") == 0,
            "teardown residue counters are nonzero",
        )
        memmaps = {}
        for pass_id, name in PASS_FILE.items():
            contract = RAW_MEMMAP_CONTRACT[name]
            path = self.root / name
            _require(
                path.stat().st_size == contract["size_bytes"], f"{name}: size drift"
            )
            memmaps[name] = {
                **contract,
                "path": str(path.resolve()),
                "sha256": self._hashers[pass_id].hexdigest(),
                "sha256_mode": "rolling_exact_sequential_frame_bytes",
            }
        receipt = {
            "schema": RAW_RECEIPT_SCHEMA,
            "status": "pass_raw_ready",
            "episode_id": episode_id,
            "batch_request_sha256": batch_request_sha256,
            "input_binding_sha256": input_binding_sha256,
            "frame_count": FRAME_COUNT,
            "rgb_frame_count": FRAME_COUNT,
            "runtime_readback_counts": {
                "normal": 75,
                "target_only_source1": 75,
                "target_only_source2": 75,
            },
            "raw_memmap_total_bytes": RAW_MEMMAP_TOTAL_BYTES,
            "raw_memmaps": memmaps,
            "rgb_frames": {
                "path": str(self.rgb_root.resolve()),
                "file_count": FRAME_COUNT,
                "names": expected_rgb_names,
                "inventory_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            {
                                "name": path.name,
                                "size_bytes": path.stat().st_size,
                                "sha256": _sha256(path),
                            }
                            for path in observed_rgb
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            },
            "metadata": metadata_records,
            "episode_teardown": dict(teardown),
            "persistence": {
                "data_files_flushed": True,
                "data_files_fsynced": True,
                "metadata_files_fsynced": True,
                "raw_directory_fsynced_before_ready": True,
                "ready_publication": "same_directory_atomic_rename_then_directory_fsync",
            },
            "qualification_claim": False,
            "formal_episode_count": 0,
            "ground_contact_release_qualified": False,
            "motion_realism_release_qualified": motion_realism_release_qualified,
        }
        ready = self.root / "RAW_READY.json"
        _atomic_json(ready, receipt)
        self.close()
        return ready

    def close(self) -> None:
        if self._closed:
            return
        for array in self._maps.values():
            array.flush()
        self._maps.clear()
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
