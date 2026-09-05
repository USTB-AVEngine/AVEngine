#!/usr/bin/env python3
"""One HM3D house, start to finish, in a single task.

The owner's interface is one click: pick a house, press go. This chain
runs the four stages the submit page used to offer as separate buttons -
room inventory, acoustic package with frame parity, a room-scoped route
bank, and a rendered episode - and ends on the machine audition, so the
task's colour is the verdict on the finished deliverable.

The one decision a person used to make between the buttons, which room,
is made here by a deliberate and recorded policy: rooms are ranked
furnished-first (a couch or a bed makes a room recognizably a room),
then by floor area; rooms too small to host the shortest route band are
excluded; and up to three candidates are tried in order, because a room
may honestly refuse routes (doors too narrow, islands of navmesh) and
the next-best room is a better answer than a red task. Every attempt
and the reason for the final choice go into the receipt.

The acoustic package is not consumed by the episode render (the ss2
audio chain reads the raw scene plus material config); it is compiled
here because it is the certified-QA chain's admission gate for this
house, and its frame-parity check is the one defence a sideways mesh
cannot pass. A scan-mesh geometry report may legitimately say ``fail`` for
its production watertight check; that reported QA status is retained and is
not the package stage exit condition. This chain fails the package stage when
the compiler or frame-parity command exits nonzero, including a frame-parity
mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.episode_clock import (  # noqa: E402
    EpisodeClock,
    EpisodeClockError,
    LEGACY_FRAME_COUNT,
    LEGACY_FRAME_RATE_HZ,
    LEGACY_SAMPLE_RATE_HZ,
)


def _child_environment() -> dict[str, str]:
    """Run every stage with this worktree's source ahead of inherited paths."""

    environment = dict(os.environ)
    source = str((REPOSITORY / "src").resolve())
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source + (os.pathsep + existing if existing else "")
    )
    return environment


def _source_provenance() -> dict[str, object]:
    """Record the source and interpreter without making provenance a gate."""

    record: dict[str, object] = {
        "repository": str(REPOSITORY.resolve()),
        "git_commit": None,
        "git_branch": None,
        "entrypoint": str(Path(__file__).resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        # These are the cwd and source used for every AVEngine child stage.
        "cwd": str(REPOSITORY.resolve()),
        "avengine_source": str((REPOSITORY / "src").resolve()),
    }
    errors: list[str] = []

    def git(*args: str, optional: bool = False) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(REPOSITORY), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except OSError as exc:
            if not optional:
                errors.append(f"{args}: {exc}")
            return None
        if completed.returncode != 0:
            if not optional:
                errors.append(
                    f"{args}: {completed.stderr.strip() or completed.returncode}"
                )
            return None
        return completed.stdout.strip() or None

    repository = git("rev-parse", "--show-toplevel")
    if repository:
        try:
            record["repository"] = str(Path(repository).resolve())
        except OSError:
            record["repository"] = repository
    record["git_commit"] = git("rev-parse", "HEAD")
    record["git_branch"] = git(
        "symbolic-ref", "--quiet", "--short", "HEAD", optional=True
    )
    if errors:
        record["git_error"] = "; ".join(errors)
    return record


_FURNISHED = {
    "couch", "sofa", "bed", "chair", "table", "tv", "cabinet", "desk",
    "refrigerator", "shelf", "bookshelf", "armchair", "nightstand",
}
# The owner's sizing rule: a room must be neither too small nor too large,
# anchored on the rooms the chain already proved out - the kujiale living
# room that hosted 200 routes measures 49.5 m2 (6 x 12), so 50 keeps it in,
# while the 81 m2 stairwell hall in 00803 stays out. Both bounds are argv
# knobs; these are the defaults.
_MINIMUM_AREA_M2 = 6.0
_MAXIMUM_AREA_M2 = 50.0
_MINIMUM_SHORTER_M = 2.2
_ROOM_ATTEMPTS = 3
_FLOOR_PATTERN = re.compile(r"_y([+-]?\d+(?:\.\d+)?)")


def run(step: str, argv: list[str], log_dir: Path) -> None:
    log_path = log_dir / f"{step}.log"
    print(f"=== {step}: {' '.join(argv)}", flush=True)
    completed = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        cwd=str(REPOSITORY),
        env=_child_environment(),
    )
    log_path.write_bytes(completed.stdout)
    sys.stdout.buffer.write(completed.stdout)
    sys.stdout.flush()
    if completed.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {completed.returncode}")


def attempt(step: str, argv: list[str], log_dir: Path) -> int:
    """Like run(), but a nonzero exit is an answer, not a crash."""

    log_path = log_dir / f"{step}.log"
    print(f"=== {step}: {' '.join(argv)}", flush=True)
    completed = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        cwd=str(REPOSITORY),
        env=_child_environment(),
    )
    log_path.write_bytes(completed.stdout)
    sys.stdout.buffer.write(completed.stdout)
    sys.stdout.flush()
    return completed.returncode


def the_only(pattern: str, directory: Path) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {pattern} in {directory}, found "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def rank_rooms(
    rooms: list[dict],
    *,
    minimum_area_m2: float = _MINIMUM_AREA_M2,
    maximum_area_m2: float = _MAXIMUM_AREA_M2,
) -> list[dict]:
    """Furnished rooms first, larger first, inside the owner's size range.

    Too small is a physics bound - the shortest route band is 1.5 m, so a
    room whose shorter side cannot contain it plus wall clearance would
    only burn a failed attempt. Too large is a data-quality bound - an
    81 m2 stairwell hall is not the residential room this benchmark is
    about, however navigable it is.
    """

    candidates = []
    for room in rooms:
        extent = room.get("extent_m") or [0.0, 0.0]
        shorter = min(float(extent[0]), float(extent[1]))
        area = float(room.get("floor_area_m2") or 0.0)
        if not (minimum_area_m2 <= area <= maximum_area_m2):
            continue
        if shorter < _MINIMUM_SHORTER_M:
            continue
        furnished = sorted(
            set(str(c) for c in room.get("top_categories") or []) & _FURNISHED
        )
        candidates.append({**room, "_furnished": furnished, "_shorter": shorter})
    candidates.sort(
        key=lambda r: (0 if r["_furnished"] else 1, -float(r["floor_area_m2"]))
    )
    return candidates


def band_for(shorter: float) -> tuple[float, float]:
    if shorter >= 6.0:
        return 3.5, 5.5
    return 1.5, max(2.0, round(shorter - 0.5, 1))


def _build_receipt(
    *,
    scene_dir: Path,
    scene_id: str,
    chosen: dict,
    attempts: list[dict],
    bank: Path,
    rooms_dir: Path,
    routes_dir: Path,
    output: Path,
    clock: EpisodeClock | None = None,
    runtime_ray_report: Path | None = None,
) -> dict:
    receipt = {
        "schema": "avengine_hm3d_end_to_end_receipt_v1",
        "scene_dir": str(scene_dir),
        "scene_id": scene_id,
        "source": _source_provenance(),
        "room_chosen": {
            "label": f"R{chosen['region_id']}",
            "floor_area_m2": chosen["floor_area_m2"],
            "furnished_with": chosen["_furnished"],
            "reason": (
                "有家具且面积最大" if chosen["_furnished"] else "无家具房间中面积最大"
            ),
        },
        "rooms_tried": attempts,
        "bank": str(bank),
        "stages": {
            "rooms": str(rooms_dir),
            "package": str(output / "package"),
            "routes": str(routes_dir),
            "episode": str(output / "episode"),
        },
        "episode_receipt": str(output / "episode" / "receipt.json"),
        "machine_audition": str(output / "episode" / "machine_audition.json"),
    }
    if runtime_ray_report is not None:
        ray_value = json.loads(runtime_ray_report.read_text(encoding="utf-8"))
        receipt["runtime_ray_leakage"] = {
            "path": str(runtime_ray_report.resolve()),
            "status": ray_value.get("rlr_runtime_ray_check_status"),
            "check_count": ray_value.get("rlr_runtime_ray_check_count"),
            "backend": ray_value.get("rlr_runtime_backend"),
        }
    if clock is not None:
        receipt["clock"] = clock.to_dict()
        receipt.update(
            {
                "frame_count": clock.frame_count,
                "frame_rate_hz": clock.frame_rate_float,
                "sample_rate_hz": clock.sample_rate_hz,
                "sample_count": clock.sample_count,
                "clip_seconds": clock.clip_seconds_float,
            }
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--hm3d-root", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--material-rules", required=True, type=Path)
    parser.add_argument("--audio-python", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--materials-json", required=True, type=Path)
    parser.add_argument("--hrtf", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--frame-rate-hz", type=float)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--clip-seconds", type=float)
    parser.add_argument("--connectivity-samples", type=int, default=64)
    parser.add_argument("--episodes-per-motion-case", type=int, default=8)
    parser.add_argument("--minimum-room-area-m2", type=float,
                        default=_MINIMUM_AREA_M2)
    parser.add_argument("--maximum-room-area-m2", type=float,
                        default=_MAXIMUM_AREA_M2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        clock = EpisodeClock.from_values(
            frame_count=(
                args.frame_count
                if args.frame_count is not None
                else LEGACY_FRAME_COUNT
            ),
            frame_rate_hz=(
                args.frame_rate_hz
                if args.frame_rate_hz is not None
                else LEGACY_FRAME_RATE_HZ
            ),
            sample_rate_hz=(
                args.sample_rate
                if args.sample_rate is not None
                else LEGACY_SAMPLE_RATE_HZ
            ),
            clip_seconds=args.clip_seconds,
            compatibility=(
                "legacy_default"
                if args.frame_count is None
                and args.frame_rate_hz is None
                and args.sample_rate is None
                and args.clip_seconds is None
                else "configured"
            ),
        )
    except EpisodeClockError as error:
        raise SystemExit(f"invalid episode clock: {error}") from error
    if clock.frame_rate_hz.denominator != 1:
        raise SystemExit(
            "HM3D route banks require an integer frame_rate_hz; "
            f"got {clock.frame_rate_float:g}"
        )

    scene_dir = args.scene_dir.resolve()
    scene_id = scene_dir.name.split("-", 1)[1] if "-" in scene_dir.name else None
    if not scene_id:
        raise SystemExit(
            f"scene dir name carries no id after the dash: {scene_dir.name}"
        )
    scene_glb = scene_dir / f"{scene_id}.glb"
    navmesh = scene_dir / f"{scene_id}.basis.navmesh"
    for path in (scene_glb, navmesh):
        if not path.is_file():
            raise SystemExit(f"scene input missing: {path}")

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists (fresh/no-clobber): {output}")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    python = sys.executable
    runtime = [
        "--runtime-prefix", args.runtime_prefix,
        "--magnum-site", args.magnum_site,
        "--rlr-sdk-root", args.rlr_sdk_root,
    ]

    # --- stage 1: rooms -----------------------------------------------------
    rooms_dir = output / "rooms"
    run(
        "rooms",
        [
            python, str(REPOSITORY / "tools/rooms/emit_hm3d_room_manifest.py"),
            "--scene-dir", str(scene_dir),
            "--hm3d-root", str(args.hm3d_root),
            *runtime,
            "--split", args.split,
            "--connectivity-samples", str(args.connectivity_samples),
            "--connectivity-seed", str(args.seed),
            "--output-dir", str(rooms_dir),
        ],
        logs,
    )
    manifest = the_only("*/room_manifest.json", rooms_dir)
    inventory = json.loads(
        (manifest.parent / "rooms.json").read_text(encoding="utf-8")
    )

    # --- stage 2: acoustic package with frame parity ------------------------
    run(
        "package",
        [
            python,
            str(REPOSITORY / "tools/acoustics/compile_semantic_research_package.py"),
            "--room-manifest", str(manifest),
            "--material-rules", str(args.material_rules),
            "--hm3d-root", str(args.hm3d_root),
            *runtime,
            "--verify-frame-parity",
            "--verify-ray-leakage",
            "--seed", str(args.seed),
            "--output", str(output / "package"),
        ],
        logs,
    )

    runtime_ray_report = output / "package/qa/ray_leakage_runtime.json"
    if not runtime_ray_report.is_file():
        raise SystemExit(
            "package stage passed without a retained RLR TraceRay report"
        )
    runtime_ray_value = json.loads(
        runtime_ray_report.read_text(encoding="utf-8")
    )
    if runtime_ray_value.get("rlr_runtime_ray_check_status") != "pass":
        raise SystemExit(
            "package RLR TraceRay verification did not pass: "
            f"{runtime_ray_value.get('rlr_runtime_ray_check_status')}"
        )

    # --- stage 3: pick a room, plan routes in it ----------------------------
    candidates = rank_rooms(
        list(inventory.get("rooms") or []),
        minimum_area_m2=args.minimum_room_area_m2,
        maximum_area_m2=args.maximum_room_area_m2,
    )
    if not candidates:
        raise SystemExit(
            f"no room in {scene_dir.name} sits in the size range "
            f"[{args.minimum_room_area_m2}, {args.maximum_room_area_m2}] m2 "
            f"with shorter side >= {_MINIMUM_SHORTER_M} m"
        )
    attempts: list[dict] = []
    chosen = None
    routes_dir = None
    for room in candidates[:_ROOM_ATTEMPTS]:
        label = f"R{room['region_id']}"
        (x0, z0), (x1, z1) = room["bbox_xz_m"]
        minimum, maximum = band_for(room["_shorter"])
        directory = output / f"routes_{label}"
        code = attempt(
            f"routes_{label}",
            [
                python,
                str(REPOSITORY / "tools/routes/compile_hm3d_dynamic_source_bank.py"),
                *runtime,
                "--scene", str(scene_glb),
                "--navmesh", str(navmesh),
                "--seed", str(args.seed),
                "--episodes-per-motion-case", str(args.episodes_per_motion_case),
                "--frame-count", str(clock.frame_count),
                "--frame-rate-hz", str(clock.frame_rate_float),
                "--sample-rate", str(clock.sample_rate_hz),
                "--clip-seconds", str(clock.clip_seconds_float),
                "--minimum-route-distance-m", str(minimum),
                "--maximum-route-distance-m", str(maximum),
                "--source1-height-m", "1.2",
                "--source2-height-m", "0.35",
                "--room-bounds", str(x0), str(z0), str(x1), str(z1),
                "--room-label", label,
                "--bank-dir", str(directory / "bank"),
                "--topdown-dir", str(directory / "topdown"),
                "--report", str(directory / "route_report.json"),
            ],
            logs,
        )
        report_path = directory / "route_report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        routed = int(report.get("floors_with_routes") or 0)
        attempts.append(
            {
                "room": label,
                "floor_area_m2": room["floor_area_m2"],
                "furnished_with": room["_furnished"],
                "exit_code": code,
                "floors_with_routes": routed,
            }
        )
        if code == 0 and routed > 0:
            chosen, routes_dir = room, directory
            break
        print(f"room {label} refused routes; trying the next candidate")
    if chosen is None or routes_dir is None:
        raise SystemExit(
            f"none of the {len(attempts)} candidate rooms accepted routes; "
            f"attempts: {json.dumps(attempts, ensure_ascii=False)}"
        )

    # the room's own floor picks the bank when the house has several
    banks = sorted((routes_dir / "bank").glob("*.bank.json"))
    if not banks:
        raise SystemExit(f"route stage passed but wrote no bank in {routes_dir}")
    floor_y = float(chosen.get("floor_y_m") or 0.0)

    def floor_of(bank: Path) -> float:
        match = _FLOOR_PATTERN.search(bank.name)
        return float(match.group(1)) if match else float("inf")

    bank = min(banks, key=lambda b: abs(floor_of(b) - floor_y))

    # --- stage 4: the episode, ending on its machine audition ---------------
    run(
        "episode",
        [
            python, str(REPOSITORY / "tools/studio/run_hm3d_episode.py"),
            *runtime,
            "--audio-python", str(args.audio_python),
            "--bank", str(bank),
            "--asset-dir", str(args.asset_dir),
            "--dataset-config", str(args.dataset_config),
            "--scene-id", scene_id,
            "--materials-json", str(args.materials_json),
            "--hrtf", str(args.hrtf),
            "--frame-count", str(clock.frame_count),
            "--frame-rate-hz", str(clock.frame_rate_float),
            "--sample-rate", str(clock.sample_rate_hz),
            "--clip-seconds", str(clock.clip_seconds_float),
            "--seed", str(args.seed),
            "--output", str(output / "episode"),
        ],
        logs,
    )

    receipt = _build_receipt(
        scene_dir=scene_dir,
        scene_id=scene_id,
        chosen=chosen,
        attempts=attempts,
        bank=bank,
        rooms_dir=rooms_dir,
        routes_dir=routes_dir,
        output=output,
        clock=clock,
        runtime_ray_report=runtime_ray_report,
    )
    (output / "end_to_end_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"receipt": str(output / "end_to_end_receipt.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
