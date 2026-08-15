#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# Modifications Copyright (c) AVEngine contributors.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Modern, explicit RLR acoustic-scene bindings used by AVEngine.

These bindings are independent of the legacy :class:`AudioSensor`. In builds
without ``HABITAT_WITH_AUDIO=ON``, the exported names are ``None`` and
``habitat_sim.audio_enabled`` remains the authoritative capability flag.
"""

from habitat_sim._ext.habitat_sim_bindings import (
    RLRAcousticContext,
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

__all__ = [
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
