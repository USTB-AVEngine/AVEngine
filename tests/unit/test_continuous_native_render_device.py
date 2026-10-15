import pytest
from avengine.capture.continuous_native_video import native_render_device

pytestmark = pytest.mark.fast_unit


def test_cpu_default_keeps_software_only_selection():
    env = {'LIBGL_ALWAYS_SOFTWARE': '1', 'LD_PRELOAD': '/task/libegl_cpu_only.so'}
    assert native_render_device({}, env) == -1
    with pytest.raises(RuntimeError):
        native_render_device({}, {})


def test_gpu_is_logical_zero_inside_the_one_explicit_visible_device():
    env = {'CUDA_VISIBLE_DEVICES': 'GPU-task-owned-device'}
    assert native_render_device({'render_backend': 'habitat_gpu'}, env) == 0
    assert env == {'CUDA_VISIBLE_DEVICES': 'GPU-task-owned-device'}


@pytest.mark.parametrize('visible', ['', '-1', '0,1'])
def test_gpu_never_implicitly_uses_every_device(visible):
    with pytest.raises(RuntimeError):
        native_render_device({'render_backend': 'habitat_gpu'},
                             {'CUDA_VISIBLE_DEVICES': visible})


def test_gpu_cannot_silently_render_with_software_selector():
    with pytest.raises(RuntimeError):
        native_render_device({'render_backend': 'habitat_gpu'},
            {'CUDA_VISIBLE_DEVICES': '0', 'LIBGL_ALWAYS_SOFTWARE': '1'})


def test_ue_route_is_not_a_habitat_backend():
    with pytest.raises(ValueError):
        native_render_device({'render_backend': 'ue_spear'}, {})
