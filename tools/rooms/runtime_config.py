"""Shared deployment defaults for the room-curation tools.

Environment overrides apply to new processes. The fallback values retain the
original smy deployment so moving scripts does not change existing commands.
No runtime is imported or initialized when this module is imported.
"""

from __future__ import annotations

import os
from pathlib import Path

EXTERNAL_ROOT = Path(
    os.environ.get("AVENGINE_EXTERNAL_ROOT", "/data/avengine_external")
)
DATA_ROOT = Path(
    os.environ.get("AVENGINE_HABITAT_DATA_ROOT", "/data/datasets/habitat_data")
)
RUNTIME_PREFIX = os.environ.get(
    "AVENGINE_ROOM_RUNTIME_PREFIX",
    str(
        EXTERNAL_ROOT
        / "runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z"
    ),
)
MAGNUM_SITE = os.environ.get(
    "AVENGINE_ROOM_MAGNUM_SITE",
    str(
        EXTERNAL_ROOT
        / "runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages"
    ),
)
RLR_SDK_ROOT = os.environ.get(
    "AVENGINE_ROOM_RLR_SDK_ROOT", str(EXTERNAL_ROOT / "rlr-sdk/RLRAudioPropagationPkg")
)
MP3D_ROOT = str(DATA_ROOT)
HM3D_ROOT = DATA_ROOT / "versioned_data/hm3d-1.0/hm3d"
TASKS_ROOT = EXTERNAL_ROOT / "studio/tasks"
MEDIA_ROOT = EXTERNAL_ROOT / "studio/room_curation_media"
VERDICT_ROOT = EXTERNAL_ROOT / "studio/room_curation"
ROOM_PYTHON = os.environ.get(
    "AVENGINE_ROOM_PYTHON", "/data/smy/miniconda3/envs/avengine-runtime/bin/python"
)


def habitat_runtime_options(**overrides):
    """Return fresh options, with explicit CLI arguments taking precedence."""
    options = dict(
        runtime_prefix=RUNTIME_PREFIX,
        magnum_python_site=MAGNUM_SITE,
        rlr_sdk_root=RLR_SDK_ROOT,
        mp3d_root=MP3D_ROOT,
        allow_mp3d_environment=False,
    )
    options.update(overrides)
    return options
