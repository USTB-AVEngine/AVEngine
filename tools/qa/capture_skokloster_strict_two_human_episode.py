#!/usr/bin/env python3
"""Bind the native pixel runner to the isolated Skokloster package archive."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
BASE_PATH = REPOSITORY / "tools/qa/capture_spear_native_pixel_episode.py"
BASE_SPEC = importlib.util.spec_from_file_location("skok_native_pixel_base", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot import native pixel runner: {BASE_PATH}")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

# The shared runner establishes the repository source import root before this
# standalone archive wrapper imports AVEngine's namespaced host/game client.
from avengine.backends.spear_ue import client as spear_client
from avengine.backends.spear_ue.launch import parallel_instance_settings

PACKAGED_MAP = (
    "/Game/MyAssets/Audioset/Scenes/skokloster_castle/Maps/skokloster_castle_strict"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _configure_explicit_archive(
    args: argparse.Namespace, *, native_map: str
) -> tuple[Any, Path]:
    """Use the reviewed archive without changing the shared native runner."""

    _require(native_map == PACKAGED_MAP, "Skokloster map drift")
    spear_root = args.spear_root.resolve()
    executable = args.spear_executable.resolve()
    _require(executable.is_file(), f"packaged executable is missing: {executable}")
    _require(
        "Standalone-Skokloster-Development" in str(executable),
        "only the isolated Skokloster archive is allowed",
    )
    _require(spear_root.is_dir(), f"SPEAR root is missing: {spear_root}")

    settings = parallel_instance_settings(
        args.rpc_port, graphics_adapter=args.graphics_adapter
    )
    config = spear_client.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = str(executable)
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = (
        BASE.RUNNER.INITIALIZE_CLIENT_MAX_TIME_SECONDS
    )
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = (
        BASE.RUNNER.CLIENT_INTERNAL_TIMEOUT_SECONDS
    )
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = native_map
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = (
        1.0 / BASE.RUNNER.FPS
    )
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = settings["rpc_port"]
    config.SPEAR.INSTANCE.TEMP_DIR = settings["temp_dir"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = settings["log"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.renderoffscreen = None
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = settings[
        "shared_memory_initial_unique_id"
    ]
    if settings["graphics_adapter"] is not None:
        config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.graphicsadapter = settings[
            "graphics_adapter"
        ]
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
    config.freeze()
    spear_client.configure_system(config=config)
    try:
        instance = spear_client.Instance(config=config)
    except BaseException:
        BASE.RUNNER._cleanup_failed_constructor(
            executable=executable,
            temporary_directory=Path(settings["temp_dir"]),
        )
        raise
    return instance, spear_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-plan", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--audio-wav", required=True, type=Path)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--spear-executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rpc-port", type=int, default=39831)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument("--frame-index", action="append", type=int)
    parser.add_argument(
        "--authorize-gpu-capture",
        action="store_true",
        help="Required only after the external GPU and fresh-pixel authorization gate.",
    )
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.graphics_adapter < 0 or args.warmup_frames < 0:
        parser.error("GPU and warmup values must be non-negative")
    return args


def run(args: argparse.Namespace) -> Path:
    _require(args.authorize_gpu_capture, "GPU capture authorization flag is absent")
    _require(not args.output.exists(), f"refusing to replace output: {args.output}")
    suite = json.loads(args.suite_plan.read_text(encoding="utf-8"))
    _require(suite.get("native_map") == PACKAGED_MAP, "suite map drift")
    _require(
        suite.get("packaged_executable") == str(args.spear_executable.resolve()),
        "suite/executable binding drift",
    )

    def configure(base_args: argparse.Namespace, *, native_map: str):
        base_args.spear_executable = args.spear_executable
        return _configure_explicit_archive(base_args, native_map=native_map)

    BASE.RUNNER._configure_instance = configure
    return BASE.run(args)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
