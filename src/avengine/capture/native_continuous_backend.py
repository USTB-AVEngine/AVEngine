"""Explicit visual-backend dispatch; no implicit conversion of Kujiale to Habitat."""
from __future__ import annotations


def create_native_video(house, scene, inputs, assets, **kwargs):
    backend = house.get('input', {}).get('visual_backend', 'habitat')
    if backend == 'habitat':
        from avengine.capture.continuous_native_video import NativeContinuousVideo
        return NativeContinuousVideo(house, scene, inputs, assets, **kwargs)
    if backend == 'ue_spear':
        from avengine.capture.continuous_ue_video import UENativeContinuousVideo
        return UENativeContinuousVideo(house, scene, inputs, assets, **kwargs)
    raise ValueError(f'Unsupported native visual backend: {backend!r}')
