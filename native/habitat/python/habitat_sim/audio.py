#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# Modifications Copyright (c) AVEngine contributors.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Modern, explicit RLR acoustic-scene bindings used by AVEngine.

These bindings are independent of the legacy :class:`AudioSensor`. The
``RLR_ADAPTER_ENABLED`` capability is true only when
``AVENGINE_HABITAT_BUILD_RLR_ADAPTER=ON`` produced concrete modern bindings.
With the adapter off, compatibility attributes are ``None`` and the modern
names are omitted from ``__all__``. That option does not enable legacy
``AudioSensor``; ``habitat_sim.audio_enabled`` therefore remains ``False``
unless the separate legacy-audio option is enabled.
"""

from habitat_sim.bindings import (
    RLRAcousticContext,
    RLR_ADAPTER_ENABLED,
    RLRChannelLayoutType,
    RLRContextConfiguration,
    RLRListenerReceipt,
    RLRMaterialUploadReceipt,
    RLROwnedIR,
    RLRRayResult,
    RLRSceneReadbackReport,
    RLRSceneUploadReport,
    RLRSourceReceipt,
)

__all__ = ["RLR_ADAPTER_ENABLED"]
if RLR_ADAPTER_ENABLED:
    __all__.extend(
        [
            "RLRAcousticContext",
            "RLRChannelLayoutType",
            "RLRContextConfiguration",
            "RLRListenerReceipt",
            "RLRMaterialUploadReceipt",
            "RLROwnedIR",
            "RLRRayResult",
            "RLRSceneReadbackReport",
            "RLRSceneUploadReport",
            "RLRSourceReceipt",
        ]
    )
