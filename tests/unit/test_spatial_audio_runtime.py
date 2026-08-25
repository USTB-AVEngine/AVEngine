from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.m3.runtime import CompiledAcousticScene, RuntimeAnchor, RuntimeContractError
from avengine.spatial_audio import runtime


def _scene() -> CompiledAcousticScene:
    database = b"{}"
    return CompiledAcousticScene(
        manifest_path=__file__,
        manifest_sha256="1" * 64,
        manifest={},
        package_id="test_scene",
        package_content_sha256="2" * 64,
        material_database_path=__file__,
        material_database_bytes=database,
        material_database_sha256=hashlib.sha256(database).hexdigest(),
        material_categories_document={},
        rlr_material_database={},
        material_categories=("wall",),
        objects=({},),
        geometry_records={},
        triangle_count_by_material={"wall": 1},
        qa_reports={},
    )


def _simulation() -> runtime.M4SimulationConfig:
    return runtime.M4SimulationConfig.from_mapping(
        {
            "frequency_bands": 4,
            "direct_sh_order": 3,
            "indirect_sh_order": 1,
            "direct_ray_count": 8,
            "indirect_ray_count": 8,
            "indirect_ray_depth": 2,
            "source_ray_count": 8,
            "source_ray_depth": 2,
            "max_diffraction_order": 1,
            "thread_count": 1,
            "sample_rate_hz": 16000.0,
            "max_ir_seconds": 1.0,
            "unit_scale": 1.0,
            "global_volume": 1.0,
            "speed_of_sound_m_s": 343.0,
            "direct": True,
            "indirect": True,
            "diffraction": True,
            "transmission": False,
            "mesh_simplification": False,
            "temporal_coherence": False,
            "channel_layout": {"type": "ambisonics", "channel_count": 4},
        }
    )


class _FakeConfiguration:
    pass


class _FakeContext:
    instances = []
    source_receipt_tamper = None
    listener_receipt_tamper = None
    ir_metadata_tamper = None
    source_index_tamper = None
    listener_index_tamper = None

    def __init__(self, configuration):
        self.configuration = configuration
        self.sources = []
        self.channel_count = None
        self.realized = False
        _FakeContext.instances.append(self)

    def load_acoustic_scene(self, database, categories, objects):
        return object()

    def add_source(self, source_id, position, radius):
        self.sources.append(
            {
                "source_id": source_id,
                "position": tuple(float(value) for value in np.asarray(position, dtype=np.float32)),
                "radius": float(np.float32(radius)),
            }
        )
        return len(self.sources) - 1

    def add_listener(
        self, listener_id, position, orientation, layout, channel_count, radius, hrtf
    ):
        self.listener_id = listener_id
        self.listener_position = tuple(
            float(value) for value in np.asarray(position, dtype=np.float32)
        )
        self.listener_orientation = tuple(
            float(value) for value in np.asarray(orientation, dtype=np.float32)
        )
        self.listener_layout = layout
        self.channel_count = channel_count
        self.listener_radius = float(np.float32(radius))
        self.hrtf_file_path = hrtf
        return 0

    @staticmethod
    def _tamper(receipt, declaration):
        if declaration is not None:
            field, value = declaration
            setattr(receipt, field, value)
        return receipt

    def source_index(self, source_id):
        index = next(
            index
            for index, source in enumerate(self.sources)
            if source["source_id"] == source_id
        )
        return index if self.source_index_tamper is None else self.source_index_tamper

    def listener_index(self, listener_id):
        if listener_id != self.listener_id:
            raise KeyError(listener_id)
        return 0 if self.listener_index_tamper is None else self.listener_index_tamper

    def source_registration_receipts(self):
        return [
            self._tamper(
                SimpleNamespace(
                    source_id=source["source_id"],
                    position=source["position"],
                    radius=source["radius"],
                    native_index=index,
                    native_realized=self.realized,
                ),
                self.source_receipt_tamper if index == 0 else None,
            )
            for index, source in enumerate(self.sources)
        ]

    def listener_registration_receipts(self):
        return [
            self._tamper(
                SimpleNamespace(
                    listener_id=self.listener_id,
                    position=self.listener_position,
                    orientation_wxyz=self.listener_orientation,
                    layout_type=self.listener_layout,
                    channel_count=self.channel_count,
                    radius=self.listener_radius,
                    hrtf_file_path=self.hrtf_file_path,
                    native_index=0,
                    native_realized=self.realized,
                ),
                self.listener_receipt_tamper,
            )
        ]

    def _layout_metadata(self):
        values = {
            "ambisonics": {
                "layout_id": "rlr_foa_acn_n3d_world_v1",
                "normalization": "N3D",
                "coordinate_frame": "avengine_world",
                "channel_labels": ["W", "Y", "Z", "X"],
            },
            "binaural": {
                "layout_id": "rlr_binaural_lr_v1",
                "normalization": "not_applicable",
                "coordinate_frame": "listener_local",
                "channel_labels": ["left", "right"],
            },
            "mono": {
                "layout_id": "rlr_mono_v1",
                "normalization": "not_applicable",
                "coordinate_frame": "not_directional",
                "channel_labels": ["mono"],
            },
        }
        return values[self.listener_layout]

    def simulate_owned(self):
        self.realized = True
        metadata = self._layout_metadata()
        responses = []
        for index, source in enumerate(self.sources):
            response = SimpleNamespace(
                listener_id=self.listener_id,
                source_id=source["source_id"],
                listener_native_index=0,
                source_native_index=index,
                sample_rate=16000.0,
                channel_count=self.channel_count,
                sample_count=8,
                layout_type=self.listener_layout,
                ambisonic_order=1 if self.listener_layout == "ambisonics" else 0,
                channel_ordering=(
                    "ACN" if self.listener_layout == "ambisonics" else "native"
                ),
                samples=np.full(
                    (self.channel_count, 8), index + 1, dtype=np.float32
                ),
                **metadata,
            )
            responses.append(
                self._tamper(
                    response,
                    self.ir_metadata_tamper if index == 0 else None,
                )
            )
        return responses

    def indirect_ray_efficiency(self):
        return 0.5


class _FakeLayout:
    Mono = "mono"
    Binaural = "binaural"
    Ambisonics = "ambisonics"


def _install_fake_runtime(monkeypatch):
    _FakeContext.instances.clear()
    _FakeContext.source_receipt_tamper = None
    _FakeContext.listener_receipt_tamper = None
    _FakeContext.ir_metadata_tamper = None
    _FakeContext.source_index_tamper = None
    _FakeContext.listener_index_tamper = None
    module = SimpleNamespace(
        RLRContextConfiguration=_FakeConfiguration,
        RLRAcousticContext=_FakeContext,
        RLRChannelLayoutType=_FakeLayout,
    )
    monkeypatch.setattr(runtime, "load_habitat_runtime", lambda: (module, {"fake": True}))
    monkeypatch.setattr(
        runtime,
        "_native_configuration",
        lambda habitat_module, simulation: (_FakeConfiguration(), simulation.to_dict()),
    )
    monkeypatch.setattr(runtime, "_upload_report", lambda value: {"verified": True})
    monkeypatch.setattr(runtime, "_verify_upload_report", lambda scene, report: None)
    return module


def test_render_canonicalizes_requested_order_and_maps_every_pair(monkeypatch):
    _install_fake_runtime(monkeypatch)
    source_b = RuntimeAnchor("source_b", (2.0, 0.0, 0.0))
    source_a = RuntimeAnchor("source_a", (1.0, 0.0, 0.0))
    listener = RuntimeAnchor("listener0", (0.0, 0.0, 0.0))

    result = runtime.render_named_sources(
        _scene(),
        _simulation(),
        sources=(source_b, source_a),
        listener=listener,
    )

    assert result.requested_source_order == ("source_b", "source_a")
    assert result.canonical_native_source_order == ("source_a", "source_b")
    assert [
        source["source_id"] for source in _FakeContext.instances[-1].sources
    ] == ["source_a", "source_b"]
    assert list(result.by_pair()) == [
        ("listener0", "source_a"),
        ("listener0", "source_b"),
    ]
    assert all(pair.samples.shape == (4, 8) for pair in result.pairs)
    assert result.pairs[0].channel_labels == ("W", "Y", "Z", "X")
    assert result.pairs[0].normalization == "N3D"
    assert result.endpoint_receipts["authority"] == "native_registration_readback"
    assert [
        receipt["canonical_native_index"]
        for receipt in result.endpoint_receipts["sources"]
    ] == [0, 1]
    assert all(
        receipt["native_realized"]
        for receipt in result.endpoint_receipts["sources"]
    )
    assert result.endpoint_receipts["listener"]["native_realized"] is True


def test_source_ids_are_portably_constrained():
    with pytest.raises(RuntimeContractError, match="ASCII"):
        runtime.canonical_source_order((RuntimeAnchor("声源", (0.0, 0.0, 0.0)),))
    with pytest.raises(RuntimeContractError, match="may contain only"):
        runtime.canonical_source_order((RuntimeAnchor("source/a", (0.0, 0.0, 0.0)),))
    with pytest.raises(RuntimeContractError, match="must start"):
        runtime.canonical_source_order(
            (RuntimeAnchor("_source", (0.0, 0.0, 0.0)),)
        )
    assert runtime.canonical_source_order(
        (RuntimeAnchor("s" * 128, (0.0, 0.0, 0.0)),)
    ) == ("s" * 128,)
    with pytest.raises(RuntimeContractError, match="at most 128"):
        runtime.canonical_source_order(
            (RuntimeAnchor("s" * 129, (0.0, 0.0, 0.0)),)
        )


def test_listener_id_uses_the_same_portable_length_contract(monkeypatch):
    _install_fake_runtime(monkeypatch)
    with pytest.raises(RuntimeContractError, match="must start"):
        runtime.render_named_sources(
            _scene(),
            _simulation(),
            sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
            listener=RuntimeAnchor(".listener", (0.0, 0.0, 0.0)),
        )
    with pytest.raises(RuntimeContractError, match="at most 128"):
        runtime.render_named_sources(
            _scene(),
            _simulation(),
            sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
            listener=RuntimeAnchor("l" * 129, (0.0, 0.0, 0.0)),
        )


@pytest.mark.parametrize(
    ("layout_type", "channel_count", "layout_id", "labels", "normalization", "frame"),
    [
        (
            "ambisonics",
            4,
            "rlr_foa_acn_n3d_world_v1",
            ("W", "Y", "Z", "X"),
            "N3D",
            "avengine_world",
        ),
        (
            "binaural",
            2,
            "rlr_binaural_lr_v1",
            ("left", "right"),
            "not_applicable",
            "listener_local",
        ),
        (
            "mono",
            1,
            "rlr_mono_v1",
            ("mono",),
            "not_applicable",
            "not_directional",
        ),
    ],
)
def test_render_requires_exact_native_layout_metadata(
    monkeypatch,
    layout_type,
    channel_count,
    layout_id,
    labels,
    normalization,
    frame,
):
    _install_fake_runtime(monkeypatch)
    result = runtime.render_named_sources(
        _scene(),
        _simulation(),
        sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
        listener=RuntimeAnchor("listener", (0.0, 0.0, 0.0)),
        layout_type=layout_type,
        channel_count=channel_count,
    )

    pair = result.pairs[0]
    assert pair.layout_id == layout_id
    assert pair.channel_labels == labels
    assert pair.normalization == normalization
    assert pair.coordinate_frame == frame
    assert result.endpoint_receipts["listener"]["layout_type"] == layout_type
    assert result.endpoint_receipts["listener"]["channel_count"] == channel_count


@pytest.mark.parametrize(
    ("field", "tampered", "error"),
    [
        ("source_id", "different_source", "order/ID"),
        ("native_index", 7, "native_index"),
        ("position", (9.0, 0.0, 0.0), "position"),
        ("radius", 0.5, "radius"),
        ("native_realized", False, "not realized"),
    ],
)
def test_source_native_receipt_tamper_is_rejected(
    monkeypatch, field, tampered, error
):
    _install_fake_runtime(monkeypatch)
    _FakeContext.source_receipt_tamper = (field, tampered)

    with pytest.raises(RuntimeContractError, match=error):
        runtime.render_named_sources(
            _scene(),
            _simulation(),
            sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
            listener=RuntimeAnchor("listener", (0.0, 0.0, 0.0)),
        )


@pytest.mark.parametrize(
    ("field", "tampered", "error"),
    [
        ("listener_id", "different_listener", "ID differs"),
        ("native_index", 2, "native_index"),
        ("position", (9.0, 0.0, 0.0), "position"),
        ("orientation_wxyz", (0.0, 1.0, 0.0, 0.0), "orientation"),
        ("layout_type", _FakeLayout.Mono, "layout differs"),
        ("channel_count", 1, "channel count"),
        ("radius", 0.5, "radius"),
        ("hrtf_file_path", "other.sofa", "HRTF path"),
        ("native_realized", False, "not realized"),
    ],
)
def test_listener_native_receipt_tamper_is_rejected(
    monkeypatch, field, tampered, error
):
    _install_fake_runtime(monkeypatch)
    _FakeContext.listener_receipt_tamper = (field, tampered)

    with pytest.raises(RuntimeContractError, match=error):
        runtime.render_named_sources(
            _scene(),
            _simulation(),
            sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
            listener=RuntimeAnchor("listener", (0.0, 0.0, 0.0)),
        )


@pytest.mark.parametrize(
    ("field", "tampered", "error"),
    [
        ("layout_id", "wrong_layout", "layout_id"),
        ("normalization", "wrong", "normalization"),
        ("coordinate_frame", "wrong", "coordinate_frame"),
        ("channel_labels", ["X", "Y", "Z", "W"], "channel labels"),
    ],
)
def test_native_ir_metadata_tamper_is_rejected(monkeypatch, field, tampered, error):
    _install_fake_runtime(monkeypatch)
    _FakeContext.ir_metadata_tamper = (field, tampered)

    with pytest.raises(RuntimeContractError, match=error):
        runtime.render_named_sources(
            _scene(),
            _simulation(),
            sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
            listener=RuntimeAnchor("listener", (0.0, 0.0, 0.0)),
        )


def test_native_index_lookup_must_match_receipts(monkeypatch):
    _install_fake_runtime(monkeypatch)
    _FakeContext.source_index_tamper = 3
    with pytest.raises(RuntimeContractError, match="source lookup"):
        runtime.render_named_sources(
            _scene(),
            _simulation(),
            sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
            listener=RuntimeAnchor("listener", (0.0, 0.0, 0.0)),
        )

    _install_fake_runtime(monkeypatch)
    _FakeContext.listener_index_tamper = 3
    with pytest.raises(RuntimeContractError, match="listener lookup"):
        runtime.render_named_sources(
            _scene(),
            _simulation(),
            sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
            listener=RuntimeAnchor("listener", (0.0, 0.0, 0.0)),
        )


def test_direct_only_probe_policy_is_explicit():
    probe = runtime.direct_only_simulation(
        _simulation(), layout_type="ambisonics", channel_count=4
    )
    assert probe.direct is True
    assert probe.indirect is False
    assert probe.diffraction is False
    assert probe.transmission is False
    assert probe.temporal_coherence is False


def test_ambisonic_channel_count_must_be_square():
    value = _simulation().to_dict()
    value["channel_layout"] = {"type": "ambisonics", "channel_count": 2}
    with pytest.raises(RuntimeContractError, match="encode an order"):
        runtime.M4SimulationConfig.from_mapping(value)


def test_current_installed_render_forwards_only_explicit_runtime_inputs(
    monkeypatch,
) -> None:
    module = _install_fake_runtime(monkeypatch)
    calls = {}

    def load_current_runtime(**kwargs):
        calls.update(kwargs)
        return module, {
            "runtime_identity": {
                "identity_schema": "avengine_current_installed_rlr_runtime_v1",
                "mode": "current-installed",
            }
        }

    monkeypatch.setattr(runtime, "load_habitat_runtime", load_current_runtime)
    result = runtime.render_named_sources(
        _scene(),
        _simulation(),
        sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
        listener=RuntimeAnchor("listener", (0.0, 0.0, 0.0)),
        runtime_mode="current-installed",
        runtime_prefix="/current/habitat",
        rlr_sdk_root="/current/sdk",
        magnum_python_site="/current/magnum/python",
    )

    assert calls == {
        "runtime_mode": "current-installed",
        "runtime_prefix": "/current/habitat",
        "rlr_sdk_root": "/current/sdk",
        "magnum_python_site": "/current/magnum/python",
    }
    assert result.runtime["runtime_identity"]["mode"] == "current-installed"


def test_foa_render_rejects_an_hrtf_before_native_context(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)

    with pytest.raises(RuntimeContractError, match="HRTF file is valid only"):
        runtime.render_named_sources(
            _scene(),
            _simulation(),
            sources=(RuntimeAnchor("source", (1.0, 0.0, 0.0)),),
            listener=RuntimeAnchor("listener", (0.0, 0.0, 0.0)),
            layout_type="ambisonics",
            channel_count=4,
            hrtf_file_path="unexpected.sofa",
        )

    assert not _FakeContext.instances
