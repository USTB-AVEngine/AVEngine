from __future__ import annotations

import math

import pytest

from avengine.qa.spear_unreal_capabilities import (
    SpearUnrealCapabilityError,
    live_handle,
    read_handle_capability,
    resolve_component_bone_names,
    set_numeric_property_with_readback,
)


class _Proxy:
    def __init__(self, handle: int) -> None:
        self.uobject = handle


class _PropertyMesh(_Proxy):
    GetStaticMesh = _Proxy(999)

    def get_property_value(self, *, property_name: str, as_handle: bool) -> int:
        assert property_name == "StaticMesh" and as_handle is True
        return 42


class _GetterMesh(_Proxy):
    def GetStaticMesh(self, *, as_handle: bool) -> int:
        assert as_handle is True
        return 41

    def get_property_value(self, **_: object) -> int:
        raise AssertionError("callable getter must be authoritative")


class _FailingGetterMesh(_Proxy):
    def GetStaticMesh(self, *, as_handle: bool) -> int:
        assert as_handle is True
        raise RuntimeError("live getter failure")

    def get_property_value(self, **_: object) -> int:
        return 99


class _Bones:
    def __init__(self, names: list[str], *, requested_aliases: bool = False) -> None:
        self.names = names
        self.requested_aliases = requested_aliases

    def GetNumBones(self) -> int:
        return len(self.names)

    def GetBoneName(self, *, BoneIndex: int) -> str:
        return self.names[BoneIndex]

    def GetBoneIndex(self, *, BoneName: str) -> int:
        if BoneName in self.names:
            return self.names.index(BoneName)
        if self.requested_aliases:
            normalized = "".join(c for c in BoneName.casefold() if c.isalnum())
            matches = [
                index
                for index, name in enumerate(self.names)
                if "".join(c for c in name.casefold() if c.isalnum()) == normalized
            ]
            return matches[0] if len(matches) == 1 else -1
        return -1


class _Numeric(_Proxy):
    def __init__(self, handle: int, *, drift: bool = False) -> None:
        super().__init__(handle)
        self.value = 0.0
        self.drift = drift

    def set_property_value(self, *, property_name: str, property_value: float) -> None:
        assert property_name == "FOVAngle"
        self.value = property_value
        if self.drift:
            self.uobject += 1

    def get_property_value(self, *, property_name: str) -> float:
        assert property_name == "FOVAngle"
        return self.value


def test_live_handle_accepts_integer_and_proxy_but_rejects_bool() -> None:
    assert live_handle(7, owner="raw") == 7
    assert live_handle(_Proxy(8), owner="proxy") == 8
    with pytest.raises(
        SpearUnrealCapabilityError, match="did not expose a live Unreal handle"
    ):
        live_handle(True, owner="bool")


def test_handle_readback_supports_callable_and_property_proxy() -> None:
    getter = read_handle_capability(
        _GetterMesh(1),
        owner="mesh",
        getter_name="GetStaticMesh",
        property_name="StaticMesh",
        getter_kwargs={"as_handle": True},
        property_kwargs={"as_handle": True},
    )
    prop = read_handle_capability(
        _PropertyMesh(2),
        owner="mesh",
        getter_name="GetStaticMesh",
        property_name="StaticMesh",
        getter_kwargs={"as_handle": True},
        property_kwargs={"as_handle": True},
    )
    assert (getter["handle"], getter["strategy"]) == (41, "callable_getter")
    assert (prop["handle"], prop["strategy"]) == (42, "property_readback")


def test_handle_readback_does_not_hide_callable_runtime_failure() -> None:
    with pytest.raises(RuntimeError, match="live getter failure"):
        read_handle_capability(
            _FailingGetterMesh(3),
            owner="mesh",
            getter_name="GetStaticMesh",
            property_name="StaticMesh",
            getter_kwargs={"as_handle": True},
            property_kwargs={"as_handle": True},
        )


def test_bone_resolution_accepts_exact_and_sanitized_live_names() -> None:
    exact = resolve_component_bone_names(
        _Bones(["Bip01 L Foot"]), ["Bip01 L Foot"], owner="actor"
    )
    sanitized = resolve_component_bone_names(
        _Bones(["Bip01-L-Foot"]), ["Bip01 L Foot"], owner="actor"
    )
    assert exact["resolutions"][0]["resolution_mode"] == "direct_exact_fname"
    resolution = sanitized["resolutions"][0]
    assert resolution["actual_live_name"] == "Bip01-L-Foot"
    assert resolution["resolution_mode"] == "sanitized_live_fname_required"


def test_bone_resolution_records_runtime_alias_capability() -> None:
    result = resolve_component_bone_names(
        _Bones(["Bip01-L-Foot"], requested_aliases=True),
        ["Bip01 L Foot"],
        owner="actor",
    )
    assert result["resolutions"][0]["resolution_mode"] == (
        "requested_alias_fname_accepted"
    )


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ([], "has no live bones"),
        (["Bip01 L Foot", "Bip01-L-Foot"], "exactly once"),
    ],
)
def test_bone_resolution_rejects_zero_and_ambiguous_matches(
    names: list[str], message: str
) -> None:
    with pytest.raises(SpearUnrealCapabilityError, match=message):
        resolve_component_bone_names(_Bones(names), ["Bip01 L Foot"], owner="actor")


def test_numeric_property_readback_supports_one_or_many_components() -> None:
    one = set_numeric_property_with_readback(
        {"rgb": _Numeric(1)},
        owner="legacy camera",
        property_name="FOVAngle",
        requested_value=90.0,
    )
    many = set_numeric_property_with_readback(
        {"rgb": _Numeric(2), "depth": _Numeric(3), "object_ids": _Numeric(4)},
        owner="multimodal camera",
        property_name="FOVAngle",
        requested_value=90.0,
        required_names=("rgb", "depth", "object_ids"),
    )
    assert one["observed_by_component"] == {"rgb": 90.0}
    assert len(many["component_handles"]) == 3


def test_numeric_property_readback_rejects_alias_drift_and_nonfinite() -> None:
    with pytest.raises(SpearUnrealCapabilityError, match="distinct"):
        set_numeric_property_with_readback(
            {"rgb": _Numeric(1), "depth": _Numeric(1)},
            owner="camera",
            property_name="FOVAngle",
            requested_value=90.0,
        )
    with pytest.raises(SpearUnrealCapabilityError, match="handle drift"):
        set_numeric_property_with_readback(
            {"rgb": _Numeric(2, drift=True)},
            owner="camera",
            property_name="FOVAngle",
            requested_value=90.0,
        )
    with pytest.raises(SpearUnrealCapabilityError, match="non-finite"):
        set_numeric_property_with_readback(
            {"rgb": _Numeric(3)},
            owner="camera",
            property_name="FOVAngle",
            requested_value=math.nan,
        )
