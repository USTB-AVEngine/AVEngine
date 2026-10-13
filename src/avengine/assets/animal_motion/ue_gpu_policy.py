"""Optional task-local UE card reservations, applied before any native launch.

A policy lives in the caller's declared output root, never in shared config.
Without that file, other room campaigns retain their existing behavior.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

POLICY_FILENAME = "kujiale_native_gpu_policy.json"


def check_gpu_policy(output_root, gpu_index):
    path = Path(output_root).resolve() / POLICY_FILENAME
    if not path.is_file():
        return None
    policy = json.loads(path.read_text())
    if policy.get("schema") != "kujiale_native_gpu_policy_v1":
        raise ValueError("unsupported task-local native GPU policy")
    allowed = policy.get("allowed_gpu_indices")
    if not isinstance(allowed, list) or not allowed or any(type(i) is not int or i < 0 for i in allowed):
        raise ValueError("native GPU policy needs explicit card indices")
    peak = float(policy["measured_peak_mib"])
    headroom = float(policy.get("headroom_mib", 4096))
    if not math.isfinite(peak) or peak < 0 or not math.isfinite(headroom) or headroom < 4096:
        raise ValueError("native GPU policy requires measured peak plus at least 4GiB")
    floor = int(policy["minimum_disk_free_bytes"])
    if floor < 0:
        raise ValueError("invalid native GPU policy disk floor")
    if int(gpu_index) not in allowed:
        raise ValueError(f"GPU {gpu_index} is reserved; this task only allows GPU {allowed}")
    return {**policy, "policy_path": str(path), "required_free_mib": peak + headroom}


def native_gpu_preflight(output_root, gpu_index):
    """Read live graphics-capable NVML data immediately before UE construction."""
    policy = check_gpu_policy(output_root, gpu_index)
    if policy is None:
        return None
    raw = subprocess.run(["nvidia-smi", "-q", "-x"], check=True,
                         text=True, capture_output=True, timeout=15).stdout
    cards = ET.fromstring(raw).findall("gpu")
    declared_uuid = policy.get("gpu_uuids", {}).get(str(int(gpu_index)))
    if declared_uuid:
        matches = [c for c in cards if c.findtext("uuid") == declared_uuid]
        if len(matches) != 1:
            raise RuntimeError("reserved physical GPU UUID is unavailable")
        card = matches[0]
    else:
        if not 0 <= int(gpu_index) < len(cards):
            raise RuntimeError("reserved native GPU is unavailable")
        card = cards[int(gpu_index)]
    text = card.findtext("./fb_memory_usage/free")
    if not text or not text.endswith(" MiB"):
        raise RuntimeError("reserved native GPU free memory is unavailable")
    free = float(text.split()[0])
    if not math.isfinite(free) or free < policy["required_free_mib"]:
        raise RuntimeError("reserved native GPU has less than measured peak plus 4GiB free")
    disk_free = shutil.disk_usage(Path(output_root).resolve()).free
    if disk_free < policy["minimum_disk_free_bytes"]:
        raise RuntimeError("shared disk free-space floor is insufficient before native launch")
    return {"gpu_index": int(gpu_index), "gpu_uuid": card.findtext("uuid"),
            "free_mib": free, "required_free_mib": policy["required_free_mib"],
            "shared_disk_free_bytes": disk_free, "policy_path": policy["policy_path"],
            "measurement_command": ["nvidia-smi", "-q", "-x"]}
