from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from avengine.m5.visual import _instantiate_actor_with_semantic_template
from avengine.m5_1.mixed_capture import (
    M5_1_ACTOR_SHADER_TYPE,
    M5_1_LIGHT_SETUP_KEY,
    MixedCaptureError,
    _actor_render_creation_evidence,
    _bind_m5_1_scene_lighting,
    _instantiate_human,
)


class _ShaderType:
    def __init__(self, name: str) -> None:
        self.name = name.upper()


class _Attributes:
    def __init__(self) -> None:
        self.semantic_id = 0
        self._shader_type = _ShaderType("phong")

    @property
    def shader_type(self) -> _ShaderType:
        return self._shader_type

    @shader_type.setter
    def shader_type(self, value: str) -> None:
        self._shader_type = _ShaderType(value)


class _Node:
    semantic_id = 0


class _Actor:
    def __init__(self, attributes: _Attributes) -> None:
        self.creation_attributes = attributes
        self.motion_type: object | None = None
        self.joint_positions: list[float] = []
        self.root_scene_node = _Node()

    def get_link_ids(self) -> list[int]:
        return []

    def get_link_name(self, link_id: int) -> str:
        assert link_id == -1
        return "root"


class _TemplateManager:
    def __init__(self) -> None:
        self.attributes = _Attributes()

    def load_configs(self, path: str) -> list[int]:
        assert path.endswith("human.ao_config.json")
        return [0]

    def get_template_handles(self, prefix: str) -> list[str]:
        assert prefix in {"human", "base"}
        return ["base"]

    def get_template_by_handle(self, handle: str) -> _Attributes:
        assert handle == "base"
        return self.attributes

    def register_template(self, attributes: _Attributes, handle: str) -> int:
        assert attributes is self.attributes
        assert handle.startswith("base.")
        return 1


class _ObjectManager:
    def __init__(self, templates: _TemplateManager) -> None:
        self.templates = templates
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def add_articulated_object_by_template_handle(
        self, handle: str, *args: Any, **kwargs: Any
    ) -> _Actor:
        self.calls.append((handle, args, kwargs))
        return _Actor(self.templates.attributes)


class _Simulator:
    def __init__(self) -> None:
        self.templates = _TemplateManager()
        self.objects = _ObjectManager(self.templates)
        self.metadata_mediator = SimpleNamespace(
            ao_template_manager=self.templates
        )

    def get_articulated_object_manager(self) -> _ObjectManager:
        return self.objects


def _habitat_sim() -> SimpleNamespace:
    return SimpleNamespace(
        physics=SimpleNamespace(MotionType=SimpleNamespace(KINEMATIC="kinematic"))
    )


def test_common_m5_actor_helper_preserves_default_instantiation_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = _Simulator()
    binding = object()
    monkeypatch.setattr(
        "avengine.m5.visual.bind_habitat_link_layout",
        lambda *args, **kwargs: binding,
    )
    bundle = SimpleNamespace(
        joint_mapping={"joint_order": ["root"], "runtime_joint_order": ["root"]}
    )

    actor, retained_binding = _instantiate_actor_with_semantic_template(
        simulator,
        bundle=bundle,
        habitat_sim=_habitat_sim(),
        base_handle="base",
        semantic_id=221,
        actor_index=1,
    )

    assert retained_binding is binding
    assert actor.creation_attributes.shader_type.name == "PHONG"
    assert simulator.objects.calls[0][1:] == ((), {})


def test_common_m5_actor_helper_binds_explicit_light_key_and_pbr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = _Simulator()
    monkeypatch.setattr(
        "avengine.m5.visual.bind_habitat_link_layout",
        lambda *args, **kwargs: object(),
    )
    bundle = SimpleNamespace(
        joint_mapping={"joint_order": ["root"], "runtime_joint_order": ["root"]}
    )

    actor, _ = _instantiate_actor_with_semantic_template(
        simulator,
        bundle=bundle,
        habitat_sim=_habitat_sim(),
        base_handle="base",
        semantic_id=221,
        actor_index=1,
        light_setup_key=M5_1_LIGHT_SETUP_KEY,
        shader_type=M5_1_ACTOR_SHADER_TYPE,
    )

    assert actor.creation_attributes.shader_type.name == "PBR"
    assert simulator.objects.calls[0][1:] == (
        (),
        {"light_setup_key": M5_1_LIGHT_SETUP_KEY},
    )


@pytest.mark.parametrize("current_light_count", [0, 3])
def test_m5_1_scene_lighting_copies_zero_or_populated_setup_and_reads_hbao(
    current_light_count: int,
) -> None:
    current = [object() for _ in range(current_light_count)]
    registered: dict[str, list[object]] = {}
    simulator = SimpleNamespace(
        config=SimpleNamespace(sim_cfg=SimpleNamespace(enable_hbao=True)),
        get_current_light_setup=lambda: list(current),
        set_light_setup=lambda setup, key: registered.__setitem__(key, list(setup)),
        get_light_setup=lambda key: list(registered[key]),
    )
    configuration = SimpleNamespace(
        sim_cfg=SimpleNamespace(enable_hbao=True)
    )

    evidence = _bind_m5_1_scene_lighting(simulator, configuration)

    assert evidence["hbao"]["configuration_readback"] is True
    assert evidence["hbao"]["simulator_readback"] is True
    lighting = evidence["scene_lighting"]
    assert lighting["actor_light_setup_key"] == M5_1_LIGHT_SETUP_KEY
    assert lighting["current_light_count"] == current_light_count
    assert lighting["registered_light_count"] == current_light_count
    assert lighting["registered_setup_matches_current"] is True


def test_m5_1_scene_lighting_fails_closed_when_hbao_did_not_read_back() -> None:
    simulator = SimpleNamespace(
        config=SimpleNamespace(sim_cfg=SimpleNamespace(enable_hbao=False))
    )
    configuration = SimpleNamespace(
        sim_cfg=SimpleNamespace(enable_hbao=True)
    )
    with pytest.raises(MixedCaptureError, match="HBAO"):
        _bind_m5_1_scene_lighting(simulator, configuration)


def test_human_helper_and_actor_evidence_retain_pbr_and_creation_light_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = _Simulator()
    binding = object()
    monkeypatch.setattr(
        "avengine.m5_1.mixed_capture.bind_local_tr_habitat_layout",
        lambda *args, **kwargs: binding,
    )
    package = SimpleNamespace(
        habitat_ao_config=Path("/tmp/human.ao_config.json"),
        mapping=SimpleNamespace(root_joint_id="root"),
    )

    actor, retained_binding, blocks = _instantiate_human(
        simulator,
        package=package,
        habitat_sim=_habitat_sim(),
        semantic_id=220,
        light_setup_key=M5_1_LIGHT_SETUP_KEY,
        shader_type=M5_1_ACTOR_SHADER_TYPE,
    )
    evidence = _actor_render_creation_evidence(
        actor,
        actor_id="human0",
        requested_shader_type=M5_1_ACTOR_SHADER_TYPE,
        light_setup_key=M5_1_LIGHT_SETUP_KEY,
    )

    assert retained_binding is binding
    assert blocks == ()
    assert simulator.objects.calls[0][1:] == (
        (),
        {"light_setup_key": M5_1_LIGHT_SETUP_KEY},
    )
    assert evidence["creation_shader_type_readback"] == "pbr"
    assert evidence["creation_light_setup_key_argument"] == M5_1_LIGHT_SETUP_KEY
    assert evidence["native_per_actor_light_key_readback"] == (
        "not_exposed_by_pinned_habitat_binding"
    )
