"""v7 whole-house admission: SO-exposed MP3D is training-only."""
from __future__ import annotations


def eligible_house_splits(result):
    """Exposure exceptions never waive acoustic or CPU visual admission.

    HM3D remains restricted to the clean-list policy. The owner-authorized
    MP3D exception permits training; it cannot create an unseen-house holdout.
    """
    if result.get("status") != "pass" or result.get("visual", {}).get("status") != "pass":
        return ()
    if result.get("family") == "mp3d":
        return ("train",)
    if result.get("family") != "hm3d" or result.get("known_exposure", True):
        return ()
    original = result.get("input", {}).get("original_split")
    if original == "train":
        return ("train",)
    if original == "val":
        return ("development", "test")
    return ()
