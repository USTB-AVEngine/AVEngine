from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from avengine.timeline.visual import _instantiate_actor_with_semantic_template
from avengine.capture.mixed_capture import (
    M5_1_ACTOR_SHADER_TYPE,
    M5_1_LIGHT_SETUP_KEY,
    M5_1_PBR_CONFIG_HANDLE,
    MixedCaptureError,
    _actor_render_creation_evidence,
    _bind_m5_1_scene_lighting,
    _instantiate_human,
    _prepare_m5_1_installed_pbr_ibl,
    _readback_m5_1_pbr_ibl,
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


class _PbrAttributes:
    enable_direct_lights = True
    enable_ibl = True
    map_mat_txtr_to_linear = True
    map_ibl_txtr_to_linear = True
    map_output_to_srgb = True
    use_direct_tonemap = False
    use_ibl_tonemap = True
    use_burley_diffuse = True

    def __init__(self) -> None:
        self.ibl_brdfLUT_filename = "relative-lut.png"
        self.ibl_environment_map_filename = "relative-environment.hdr"


class _PbrManager:
    def __init__(self) -> None:
        self.attributes = _PbrAttributes()
        self.templates: dict[str, _PbrAttributes] = {}
        self.loaded_path: str | None = None

    def create_template(
        self, path: str, *, register_template: bool
    ) -> _PbrAttributes:
        assert register_template is False
        self.loaded_path = path
        return self.attributes

    def register_template(
        self, attributes: _PbrAttributes, handle: str
    ) -> int:
        assert attributes is self.attributes
        self.templates[handle] = attributes
        return 7

    def get_template_by_handle(
        self, handle: str
    ) -> _PbrAttributes | None:
        return self.templates.get(handle)


class _PbrMediator:
    def __init__(self) -> None:
        self.pbr_shader_template_manager = _PbrManager()
        self.current_handle = "builtin"

    def set_curr_default_pbr_attributes_handle(self, handle: str) -> bool:
        if handle not in self.pbr_shader_template_manager.templates:
            return False
        self.current_handle = handle
        return True

    def get_curr_default_pbr_attributes_handle(self) -> str:
        return self.current_handle


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
        "avengine.timeline.visual.bind_habitat_link_layout",
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
        "avengine.timeline.visual.bind_habitat_link_layout",
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
def test_capture_scene_lighting_copies_zero_or_populated_setup_and_reads_hbao(
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
    assert lighting["required_zero_direct_lights"] is False


def test_installed_m5_1_scene_lighting_requires_zero_direct_lights() -> None:
    current = [object()]
    registered: dict[str, list[object]] = {}
    simulator = SimpleNamespace(
        config=SimpleNamespace(sim_cfg=SimpleNamespace(enable_hbao=True)),
        get_current_light_setup=lambda: list(current),
        set_light_setup=lambda setup, key: registered.__setitem__(
            key, list(setup)
        ),
        get_light_setup=lambda key: list(registered[key]),
    )
    configuration = SimpleNamespace(
        sim_cfg=SimpleNamespace(enable_hbao=True)
    )

    with pytest.raises(MixedCaptureError, match="zero direct lights"):
        _bind_m5_1_scene_lighting(
            simulator,
            configuration,
            require_zero_direct_lights=True,
        )

    current.clear()
    evidence = _bind_m5_1_scene_lighting(
        simulator,
        configuration,
        require_zero_direct_lights=True,
    )
    assert evidence["scene_lighting"]["current_light_count"] == 0
    assert evidence["scene_lighting"]["required_zero_direct_lights"] is True


def test_capture_scene_lighting_fails_closed_when_hbao_did_not_read_back() -> None:
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
        "avengine.capture.mixed_capture.bind_local_tr_habitat_layout",
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


def test_installed_pbr_ibl_is_selected_before_simulator_and_read_back(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "runtime-prefix"
    config_path = prefix / "config/brown_photostudio.pbr_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")
    asset_root = tmp_path / "external-pbr"
    lut = asset_root / "bluts/brdflut_ldr_512x512.png"
    environment = asset_root / "env_maps/brown_photostudio_02_1k.hdr"
    lut.parent.mkdir(parents=True)
    environment.parent.mkdir(parents=True)
    lut.write_bytes(b"lut")
    environment.write_bytes(b"environment")
    runtime = SimpleNamespace(
        prefix=prefix.resolve(),
        pbr_asset_root=asset_root.resolve(),
    )
    configuration = SimpleNamespace(
        sim_cfg=SimpleNamespace(),
        metadata_mediator=None,
    )
    mediator = _PbrMediator()
    habitat_sim = SimpleNamespace(
        metadata=SimpleNamespace(
            MetadataMediator=lambda _sim_cfg: mediator,
        ),
    )

    evidence = _prepare_m5_1_installed_pbr_ibl(
        configuration,
        installed_runtime=runtime,
        habitat_sim=habitat_sim,
    )

    assert configuration.metadata_mediator is mediator
    assert mediator.current_handle == M5_1_PBR_CONFIG_HANDLE
    assert mediator.pbr_shader_template_manager.loaded_path == str(
        config_path.resolve()
    )
    assert evidence["status"] == "pass"
    assert evidence["phase"] == "before_simulator"
    assert evidence["registration_id"] == 7
    assert evidence["absolute_brdf_lut_path"] == str(lut.resolve())
    assert evidence["absolute_environment_map_path"] == str(
        environment.resolve()
    )
    after = _readback_m5_1_pbr_ibl(
        mediator,
        config_path=config_path.resolve(),
        asset_root=asset_root.resolve(),
        phase="after_simulator",
    )
    assert after["phase"] == "after_simulator"
    assert after["registration_id"] is None


def test_installed_pbr_ibl_requires_explicit_asset_root(
    tmp_path: Path,
) -> None:
    configuration = SimpleNamespace(sim_cfg=SimpleNamespace())
    habitat_sim = SimpleNamespace(
        metadata=SimpleNamespace(
            MetadataMediator=lambda _sim_cfg: _PbrMediator(),
        ),
    )
    runtime = SimpleNamespace(
        prefix=(tmp_path / "runtime-prefix").resolve(),
        pbr_asset_root=None,
    )

    with pytest.raises(MixedCaptureError, match="explicit PBR asset root"):
        _prepare_m5_1_installed_pbr_ibl(
            configuration,
            installed_runtime=runtime,
            habitat_sim=habitat_sim,
        )


def test_installed_pbr_ibl_rejects_current_handle_drift(
    tmp_path: Path,
) -> None:
    mediator = _PbrMediator()
    with pytest.raises(MixedCaptureError, match="current PBR config handle"):
        _readback_m5_1_pbr_ibl(
            mediator,
            config_path=tmp_path / "config.json",
            asset_root=tmp_path / "assets",
            phase="after_simulator",
        )
