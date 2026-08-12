#!/usr/bin/env python3
"""Concrete one-process SPEAR adapter for strict full75 room batches.

The adapter is deliberately separate from the qualified single-Episode and
ground-contact tools.  It reuses their capture primitives without changing
them, keeps one Apartment process and camera alive, and creates/destroys a
fresh actor hierarchy for every Episode.  Execution is fail-closed behind both
``--execute`` and the request's explicit ``execution_authorized=true`` field.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_module(
    "strict2h_room_batch_contract_native",
    TOOLS / "run_strict_two_human_full75_room_batch.py",
)
LIFECYCLE = _load_module(
    "strict2h_room_batch_lifecycle_native",
    TOOLS / "spear_room_batch_lifecycle.py",
)
RAW_SPOOL = _load_module(
    "strict2h_room_batch_raw_spool_native",
    TOOLS / "strict_two_human_raw_spool.py",
)
FINALIZE_QUEUE = _load_module(
    "strict2h_room_batch_finalize_queue_native",
    TOOLS / "strict_two_human_cpu_finalize_queue.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class SpearNativeRoomBatchSession:
    """One packaged room process, one camera, and fresh actors per Episode."""

    def __init__(
        self,
        *,
        batch: Any,
        spear_root: Path,
        rpc_port: int,
        warmup_frames: int,
    ) -> None:
        self.batch = batch
        self.spear_root = spear_root.resolve()
        self.rpc_port = rpc_port
        self.warmup_frames = warmup_frames
        self.closed = False
        self.prior_stable_names: list[str] = []
        self.backend = _load_module(
            "strict2h_existing_native_pixel_backend_room_batch",
            TOOLS / "capture_spear_native_pixel_episode.py",
        )
        self.spike = self.backend.SPIKE
        self.runner = self.backend.RUNNER
        configure = argparse.Namespace(
            spear_root=self.spear_root,
            rpc_port=self.rpc_port,
            graphics_adapter=int(batch.request["graphics_adapter_argument"]),
        )
        self.instance, self.runtime_spear_root = self.runner._configure_instance(
            configure, native_map=batch.native_map
        )
        self.game = self.instance.get_game()
        self.camera: Any | None = None
        self.components: dict[str, Any] = {}
        try:
            with self.instance.begin_frame():
                self.camera, self.components = self.spike._spawn_multimodal_camera(
                    self.game
                )
            with self.instance.end_frame():
                pass
        except BaseException:
            self.instance.close(force=True)
            raise

    def _stable_names(
        self, episode: Any, runtimes: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        prefix = f"lead_a_batch_{self.batch.request_sha256[:12]}_e{episode.ordinal:02d}"
        for actor_id, runtime in runtimes.items():
            stable_name = f"{prefix}_{actor_id}"
            self.game.unreal_service.set_stable_name_for_actor(
                actor=runtime["visual_actor"], stable_name=stable_name
            )
            result[actor_id.removesuffix("_actor")] = stable_name
        _require(
            set(result) == {"source1", "source2"}, "stable-name source closure drift"
        )
        _require(
            not set(result.values()) & set(self.prior_stable_names),
            "stable actor names were reused across Episodes",
        )
        return result

    def capture_episode_raw(
        self, *, episode: Any, attempt_root: Path, batch: Any
    ) -> Path:
        import cv2

        _require(not self.closed, "native room session is closed")
        _require(batch is self.batch, "native session batch identity drift")
        _require(attempt_root.is_dir(), "attempt root must be pre-created")
        scenario = episode.scenario
        frames = scenario.get("plan", {}).get("frames")
        _require(
            isinstance(frames, list)
            and len(frames) == CONTRACT.FRAME_COUNT
            and [int(item["frame_index"]) for item in frames]
            == list(range(CONTRACT.FRAME_COUNT)),
            "native adapter requires one ordered full75 Episode",
        )

        runtimes: dict[str, Any] = {}
        stable_names: dict[str, str] = {}
        reset_begin: dict[str, Any] = {}
        descriptors: list[dict[str, Any]] = []
        normal_readbacks: list[dict[str, Any]] = []
        target_readbacks: dict[str, list[dict[str, Any]]] = {
            "source1": [],
            "source2": [],
        }
        runtime_asset_samples: list[dict[str, Any]] = []
        teardown: dict[str, Any] | None = None

        with RAW_SPOOL.RawSpoolWriter(attempt_root) as writer:
            with self.instance.begin_frame():
                runtimes = self.runner._spawn_runtime_actors(
                    self.game, scenario, self.runtime_spear_root
                )
                stable_names = self._stable_names(episode, runtimes)
                self.backend._apply_exact_frame_with_ground_snap(
                    game=self.game,
                    camera=self.camera,
                    runtimes=runtimes,
                    scenario=scenario,
                    frame=frames[0],
                )
                self.game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                    bPaused=False
                )
            with self.instance.end_frame():
                pass
            self.instance.step(num_frames=self.warmup_frames)

            reset_begin = LIFECYCLE.begin_episode_segmentation(
                instance=self.instance,
                game=self.game,
                depth_component=self.components["depth"],
                prior_stable_names=self.prior_stable_names,
            )
            with self.instance.begin_frame():
                raw_descriptors = (
                    self.game.segmentation_service.get_mesh_proxy_geometry_descs(
                        include_debug_info=False, as_global=True
                    )
                )
            with self.instance.end_frame():
                pass
            descriptors = [
                self.backend._safe_descriptor(item) for item in raw_descriptors
            ]
            raw_ids_by_source = {
                source: self.backend._descriptor_raw_ids(raw_descriptors, stable_name)
                for source, stable_name in stable_names.items()
            }
            _require(
                all(raw_ids_by_source.values()),
                "native source proxy descriptors are missing",
            )

            for frame in frames:
                frame_index = int(frame["frame_index"])
                with self.instance.begin_frame():
                    readback = self.backend._apply_exact_frame_with_ground_snap(
                        game=self.game,
                        camera=self.camera,
                        runtimes=runtimes,
                        scenario=scenario,
                        frame=frame,
                    )
                    if frame_index in self.backend.RUNTIME_ASSET_SAMPLE_FRAME_INDICES:
                        runtime_asset_samples.append(
                            self.backend._runtime_asset_readbacks(
                                game=self.game,
                                scenario=scenario,
                                runtimes=runtimes,
                                stable_names=stable_names,
                                raw_descriptors=raw_descriptors,
                                frame=frame,
                            )
                        )
                with self.instance.end_frame():
                    rgb = self.backend._rgb_bgr(self.components["rgb"])
                    depth = self.backend._depth_native(self.components["depth"])
                    object_ids = self.backend._raw_object_ids(
                        self.components["object_ids"]
                    )
                _require(
                    cv2.imwrite(str(writer.rgb_path(frame_index)), rgb),
                    f"could not write RGB frame {frame_index}",
                )
                writer.write_frame("normal_depth", frame_index, depth)
                writer.write_frame("normal_object_ids", frame_index, object_ids)
                normal_readbacks.append(readback)

            for source in ("source1", "source2"):
                actor_id = f"{source}_actor"
                with self.instance.begin_frame():
                    LIFECYCLE.configure_target_only_pass(
                        game=self.game,
                        depth_component=self.components["depth"],
                        visual_actor=runtimes[actor_id]["visual_actor"],
                    )
                    # Rebuild proxies after changing the allow-list, matching the
                    # existing single-Episode native backend.
                    self.game.segmentation_service.initialize()
                    LIFECYCLE.configure_target_only_pass(
                        game=self.game,
                        depth_component=self.components["depth"],
                        visual_actor=runtimes[actor_id]["visual_actor"],
                    )
                    self.backend._apply_exact_frame_with_ground_snap(
                        game=self.game,
                        camera=self.camera,
                        runtimes=runtimes,
                        scenario=scenario,
                        frame=frames[0],
                    )
                with self.instance.end_frame():
                    pass
                self.instance.step(num_frames=LIFECYCLE.RESET_SETTLE_FRAMES)
                for frame in frames:
                    frame_index = int(frame["frame_index"])
                    with self.instance.begin_frame():
                        readback = self.backend._apply_exact_frame_with_ground_snap(
                            game=self.game,
                            camera=self.camera,
                            runtimes=runtimes,
                            scenario=scenario,
                            frame=frame,
                        )
                    with self.instance.end_frame():
                        depth = self.backend._depth_native(self.components["depth"])
                    writer.write_frame(
                        f"target_only_{source}_depth", frame_index, depth
                    )
                    target_readbacks[source].append(readback)

            runtime_assets = self.backend._bundle_runtime_asset_samples(
                runtime_asset_samples
            )
            teardown = LIFECYCLE.teardown_episode(
                instance=self.instance,
                game=self.game,
                runner=self.runner,
                runtimes=runtimes,
                depth_component=self.components["depth"],
                stable_names=list(stable_names.values()),
            )
            runtimes = {}
            self.prior_stable_names.extend(stable_names.values())

            writer.write_metadata(
                "runtime_readbacks.json",
                {
                    "schema": "avengine_native_spear_multimodal_runtime_readbacks_v1",
                    "normal": normal_readbacks,
                    "target_only": target_readbacks,
                },
            )
            writer.write_metadata("runtime_asset_readbacks.json", runtime_assets)
            writer.write_metadata(
                "normal_object_id_descriptors.json",
                {
                    "schema": "avengine_native_spear_object_id_descriptors_v1",
                    "descriptors": descriptors,
                },
            )
            writer.write_metadata(
                "capture_context.json",
                {
                    "schema": (
                        "avengine_native_strict_two_human_raw_capture_context_v1"
                    ),
                    "native_adapter_schema": CONTRACT.NATIVE_ADAPTER_SCHEMA,
                    "episode_id": episode.episode_id,
                    "native_map": batch.native_map,
                    "frame_indices": list(range(CONTRACT.FRAME_COUNT)),
                    "camera_pose_ids": [
                        frame["camera_state"]["pose_hash"] for frame in frames
                    ],
                    "stable_names": stable_names,
                    "segmentation_begin": reset_begin,
                    "episode_teardown": teardown,
                    "formal_episode_count": 0,
                    "qualification_claim": False,
                    "ground_contact_release_qualified": False,
                    "release_blockers": [
                        "ground_contact_requires_separate_terminal qualification"
                    ],
                },
            )
            return writer.publish_ready(
                batch_request_sha256=batch.request_sha256,
                episode_id=episode.episode_id,
                input_binding_sha256=episode.bindings["binding_sha256"],
                teardown=teardown,
                motion_realism_release_qualified=batch.request[
                    "motion_realism_release_qualified"
                ],
            )

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.game.segmentation_service.proxy_component_manager is not None:
                with self.instance.begin_frame():
                    self.game.segmentation_service.terminate()
                with self.instance.end_frame():
                    pass
            if self.camera is not None:
                LIFECYCLE.close_shared_camera(
                    instance=self.instance,
                    game=self.game,
                    camera=self.camera,
                    components=self.components,
                )
        finally:
            self.instance.close(force=True)
            self.closed = True


def run(args: argparse.Namespace) -> Path:
    batch = CONTRACT.resolve_request(args.request.resolve())
    _require(args.execute, "--execute is required for native room-batch capture")
    CONTRACT.require_execution_authorized(batch)
    finalizer_python = REPOSITORY / ".venv/bin/python"
    finalizer_script = TOOLS / "finalize_strict_two_human_raw_episode.py"

    def session_factory(resolved: Any) -> SpearNativeRoomBatchSession:
        return SpearNativeRoomBatchSession(
            batch=resolved,
            spear_root=args.spear_root,
            rpc_port=args.rpc_port,
            warmup_frames=args.warmup_frames,
        )

    def queue_factory(policy: Mapping[str, Any]) -> Any:
        return FINALIZE_QUEUE.ProcessFinalizeQueue(
            policy=policy,
            finalizer_python=finalizer_python,
            finalizer_script=finalizer_script,
            repo_root=REPOSITORY,
        )

    return CONTRACT.execute_batch(
        batch,
        session_factory=session_factory,
        finalize_queue_factory=queue_factory,
        resume=args.resume,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--spear-root", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39486)
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run(parse_args(argv))
    print(f"STRICT_TWO_HUMAN_ROOM_BATCH_OK receipt={receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
