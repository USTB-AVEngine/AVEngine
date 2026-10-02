"""Explicit external runtime/data inputs; import never initializes Habitat."""

import os
from pathlib import Path


def configured_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


TASKS_ROOT = configured_path("AVENGINE_ROOM_TASKS_ROOT")
MEDIA_ROOT = configured_path("AVENGINE_ROOM_MEDIA_ROOT")
VERDICT_ROOT = configured_path("AVENGINE_ROOM_VERDICT_ROOT")


def habitat_runtime_options():
    options = {
        key: os.environ.get(env)
        for key, env in [
            ("runtime_prefix", "AVENGINE_HABITAT_RUNTIME_PREFIX"),
            ("magnum_python_site", "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"),
            ("mp3d_root", "AVENGINE_MP3D_ROOT"),
            ("rlr_sdk_root", "AVENGINE_RLR_SDK_ROOT"),
        ]
    }
    if not all(
        options[k] for k in ("runtime_prefix", "magnum_python_site", "mp3d_root")
    ):
        raise ValueError(
            "supply AVENGINE_HABITAT_RUNTIME_PREFIX, AVENGINE_HABITAT_MAGNUM_PYTHON_SITE, AVENGINE_MP3D_ROOT"
        )
    return dict(options, allow_mp3d_environment=False)
