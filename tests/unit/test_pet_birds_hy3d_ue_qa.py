"""Native room context is required even when an isolated bird is matte."""
import numpy as np
from tools.assets.v7_pet_birds_hy3d_ue_qa import audit_native_images


def images():
    isolated = np.zeros((64, 64, 3), dtype=np.uint8)
    isolated[24:40, 24:40] = (180, 140, 50)
    scene = np.full_like(isolated, 70)
    scene[24:40, 24:40] = isolated[24:40, 24:40]
    return scene, isolated


def test_matte_bird_without_room_does_not_pass():
    _, isolated = images()
    receipt = audit_native_images(isolated, isolated)
    assert receipt["saturated_rgb_ge245_ratio"] == 0
    assert receipt["native_room_context_pixels"] == 0
    assert receipt["status"] == "fail"


def test_matte_bird_in_visible_native_room_passes():
    scene, isolated = images()
    receipt = audit_native_images(scene, isolated)
    assert receipt["native_room_context_pixels"] == 3840
    assert receipt["status"] == "pass"


def test_white_saturation_does_not_pass_with_visible_room():
    scene, isolated = images()
    isolated[24:40, 24:40] = 255
    scene[24:40, 24:40] = 255
    receipt = audit_native_images(scene, isolated)
    assert receipt["native_room_context_pixels"] == 3840
    assert receipt["saturated_rgb_ge245_ratio"] == 1
    assert receipt["status"] == "fail"
