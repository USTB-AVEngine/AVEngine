#!/usr/bin/env python3
"""Bind legacy static-camera SPEAR plans to audited QA capture requests."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.sensor_rig_trajectory import (  # noqa: E402
    POSE_HASH_ALGORITHM,
    compute_sensor_rig_pose_hash,
)


SCHEMA = "avengine_static_spear_suite_camera_upgrade_v1"
REQUEST_SCHEMA = "avengine_jaeger_full_av_native_capture_requests_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _world_from_static_listener(listener: Mapping[str, Any]) -> dict[str, Any]:
    _require(listener.get("static") is True, "Facts listener must be static")
    position = [float(value) for value in listener["position_m"]]
    yaw_rad = math.radians(float(listener["yaw_deg"]))
    world_from_rig = {
        "translation_m": position,
        "rotation_xyzw": [0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0)],
    }
    fact_wxyz = [float(value) for value in listener["orientation_wxyz"]]
    observed_wxyz = [
        world_from_rig["rotation_xyzw"][3],
        *world_from_rig["rotation_xyzw"][:3],
    ]
    _require(
        max(abs(left - right) for left, right in zip(fact_wxyz, observed_wxyz))
        <= 1.0e-12,
        "Facts listener orientation differs from its declared yaw",
    )
    return world_from_rig


def upgrade(
    *, suite_path: Path, requests_path: Path, output_path: Path
) -> Mapping[str, Any]:
    suite = deepcopy(_load(suite_path))
    requests_document = _load(requests_path)
    _require(
        requests_document.get("schema") == REQUEST_SCHEMA
        and requests_document.get("status") == "ready_for_native_capture",
        "capture requests did not pass audit",
    )
    requests = {
        request["episode_id"]: request
        for request in requests_document.get("requests", [])
    }
    _require(
        len(requests) == len(requests_document.get("requests", [])),
        "capture request Episode IDs are not unique",
    )
    scenarios = {item["scenario_id"]: item for item in suite["scenarios"]}
    _require(set(scenarios) == set(requests), "suite/request Episode sets differ")

    for episode_id, request in requests.items():
        fact_path = Path(request["fact_path"]).resolve()
        _require(
            fact_path.is_file() and _sha256(fact_path) == request["fact_sha256"],
            f"{episode_id} Fact file identity drift",
        )
        facts = _load(fact_path)
        _require(
            facts.get("episode_id") == episode_id and facts.get("status") == "pass",
            f"{episode_id} Facts did not pass",
        )
        scenario = scenarios[episode_id]
        plan = scenario["plan"]
        frames = plan["frames"]
        _require(len(frames) == 75, f"{episode_id} is not a 75-frame plan")
        declarations = {
            declaration["actor_id"].removesuffix("_actor"): declaration
            for declaration in plan["actors"]
        }
        _require(
            set(declarations) == set(facts["tracks"]["instances"]),
            f"{episode_id} source slots differ between plan and Facts",
        )
        fact_instances = {
            instance["instance_id"]: instance for instance in facts["instances"]
        }
        for slot_id, declaration in declarations.items():
            _require(
                declaration["asset_id"] == fact_instances[slot_id]["asset_id"],
                f"{episode_id}/{slot_id} visual asset identity drift",
            )
            roots = facts["tracks"]["instances"][slot_id]["root_position_m"]
            for frame_index, frame in enumerate(frames):
                state = next(
                    item
                    for item in frame["actor_states"]
                    if item["actor_id"] == f"{slot_id}_actor"
                )
                _require(
                    max(
                        abs(float(left) - float(right))
                        for left, right in zip(state["translation_m"], roots[frame_index])
                    )
                    <= 1.0e-9,
                    f"{episode_id}/{slot_id} root drift at frame {frame_index}",
                )

        world_from_rig = _world_from_static_listener(facts["listener"])
        camera = plan["camera"]
        _require(
            camera["habitat_position_m"] == facts["listener"]["position_m"]
            and float(camera["habitat_yaw_deg"])
            == float(facts["listener"]["yaw_deg"]),
            f"{episode_id} plan camera differs from Facts listener",
        )
        pose_hash = compute_sensor_rig_pose_hash(world_from_rig)
        for frame_index, frame in enumerate(frames):
            _require(
                frame["frame_index"] == frame_index
                and frame["pts_ticks"] == frame_index * 3200,
                f"{episode_id} formal frame clock drift",
            )
            frame["camera_state"] = {
                "frame_index": frame_index,
                "pts_ticks": frame["pts_ticks"],
                "habitat_position_m": list(camera["habitat_position_m"]),
                "habitat_yaw_deg": float(camera["habitat_yaw_deg"]),
                "ue_position_cm": list(camera["ue_position_cm"]),
                "ue_yaw_deg": float(camera["ue_yaw_deg"]),
                "world_from_rig": deepcopy(world_from_rig),
                "pose_hash": pose_hash,
            }
        scenario["authoritative_capture_request"] = deepcopy(request)
        scenario["static_camera_upgrade"] = {
            "schema": SCHEMA,
            "fact_sha256": request["fact_sha256"],
            "pose_hash_algorithm": POSE_HASH_ALGORITHM,
            "constant_pose_hash": pose_hash,
            "checked_actor_frame_count": 75 * len(declarations),
        }

    suite["camera_upgrade"] = {
        "schema": SCHEMA,
        "source_suite": str(suite_path.resolve()),
        "source_suite_sha256": _sha256(suite_path),
        "capture_requests": str(requests_path.resolve()),
        "capture_requests_sha256": _sha256(requests_path),
        "episode_count": len(requests),
    }
    _write(output_path, suite)
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    suite = upgrade(
        suite_path=args.suite.resolve(),
        requests_path=args.requests.resolve(),
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {"status": "pass", "episode_count": len(suite["scenarios"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
