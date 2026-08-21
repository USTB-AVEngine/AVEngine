from __future__ import annotations

from pathlib import Path

import pytest

from avengine.backends.rlr import sdk as sdk_module
from avengine.backends.rlr.sdk import (
    ExternalRlrSdkError,
    RLR_SDK_ROOT_ENVIRONMENT_VARIABLE,
    discover_external_rlr_sdk,
    preload_external_rlr_sdk,
)


def _sdk_layout(root: Path) -> Path:
    header = root / "headers" / "RLRAudioPropagation.h"
    library = root / "libs" / "linux" / "x64" / "libRLRAudioPropagation.so"
    header.parent.mkdir(parents=True)
    library.parent.mkdir(parents=True)
    header.write_text("// external SDK fixture\n", encoding="utf-8")
    library.write_bytes(b"external SDK fixture")
    return root


def test_sdk_requires_explicit_non_checkout_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RLR_SDK_ROOT_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(ExternalRlrSdkError, match="Set AVENGINE_RLR_SDK_ROOT"):
        discover_external_rlr_sdk()


def test_sdk_resolves_only_the_cmake_declared_layout(tmp_path: Path) -> None:
    root = _sdk_layout(tmp_path / "external-rlr-sdk")

    sdk = discover_external_rlr_sdk(root)

    assert sdk.root == root.resolve()
    assert sdk.header == (root / "headers" / "RLRAudioPropagation.h").resolve()
    assert sdk.library == (
        root / "libs" / "linux" / "x64" / "libRLRAudioPropagation.so"
    ).resolve()


def test_sdk_rejects_checkout_root_and_symlink_escape(tmp_path: Path) -> None:
    checkout = tmp_path / "historical-rlr"
    (checkout / ".git").mkdir(parents=True)
    checkout_sdk = _sdk_layout(checkout / "RLRAudioPropagationPkg")
    with pytest.raises(ExternalRlrSdkError, match="outside a Git checkout"):
        discover_external_rlr_sdk(checkout_sdk)

    root = _sdk_layout(tmp_path / "external-rlr-sdk")
    outside = tmp_path / "outside-libRLRAudioPropagation.so"
    outside.write_bytes(b"wrong target")
    library = root / "libs" / "linux" / "x64" / "libRLRAudioPropagation.so"
    library.unlink()
    library.symlink_to(outside)
    with pytest.raises(ExternalRlrSdkError, match="must resolve inside"):
        discover_external_rlr_sdk(root)


def test_preload_accepts_only_the_declared_loaded_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = discover_external_rlr_sdk(_sdk_layout(tmp_path / "external-rlr-sdk"))
    monkeypatch.setattr(sdk_module, "_PRELOADED_RLR_SDK_HANDLES", {})
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        sdk_module.ctypes,
        "CDLL",
        lambda path, mode: calls.append((path, mode)) or object(),
    )
    monkeypatch.setattr(sdk_module, "_mapped_rlr_libraries", lambda: (sdk.library,))

    preload_external_rlr_sdk(sdk)
    preload_external_rlr_sdk(sdk)

    assert calls == [(str(sdk.library), calls[0][1])]


@pytest.mark.parametrize(
    "include_declared",
    [False, True],
    ids=["wrong-only", "multiple"],
)
def test_preload_rejects_a_binding_neighbor_or_other_rlr_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_declared: bool
) -> None:
    sdk = discover_external_rlr_sdk(_sdk_layout(tmp_path / "external-rlr-sdk"))
    neighbor = tmp_path / "binding" / "libRLRAudioPropagation.so"
    neighbor.parent.mkdir()
    neighbor.write_bytes(b"forbidden neighbor")
    monkeypatch.setattr(sdk_module, "_PRELOADED_RLR_SDK_HANDLES", {})
    monkeypatch.setattr(sdk_module.ctypes, "CDLL", lambda _path, mode: object())
    observed = (
        (sdk.library, neighbor.resolve())
        if include_declared
        else (neighbor.resolve(),)
    )
    monkeypatch.setattr(
        sdk_module,
        "_mapped_rlr_libraries",
        lambda: tuple(sorted(observed)),
    )

    with pytest.raises(ExternalRlrSdkError, match="do not resolve only"):
        preload_external_rlr_sdk(sdk)
