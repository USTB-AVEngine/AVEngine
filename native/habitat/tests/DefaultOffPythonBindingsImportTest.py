#!/usr/bin/env python3

# Copyright (c) AVEngine contributors.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Strict isolated regression for the default-off RLR Python facade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_RLR_SYMBOLS = (
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
)


def _is_within(candidate: str, root: Path) -> bool:
    try:
        Path(candidate).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _module_file(module: object) -> str | None:
    module_file = getattr(module, "__file__", None)
    return module_file if isinstance(module_file, str) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--habitat-site", type=Path, required=True)
    parser.add_argument("--magnum-site", type=Path, required=True)
    parser.add_argument("--python-site", type=Path, required=True)
    parser.add_argument("--old-checkout", type=Path, required=True)
    args = parser.parse_args()

    _require(sys.flags.no_site, "run this regression with python -S")
    habitat_site = args.habitat_site.resolve()
    magnum_site = args.magnum_site.resolve()
    python_site = args.python_site.resolve()
    old_checkout = args.old_checkout.resolve()
    for required_root in (habitat_site, magnum_site, python_site):
        _require(required_root.is_dir(), f"missing required site: {required_root}")

    _require(
        not any(
            isinstance(entry, str) and _is_within(entry, old_checkout)
            for entry in sys.path
        ),
        "old checkout was present in sys.path before controlled site injection",
    )
    sys.path[:0] = [str(habitat_site), str(magnum_site), str(python_site)]
    _require(
        not any(
            isinstance(entry, str) and _is_within(entry, old_checkout)
            for entry in sys.path
        ),
        "old checkout entered sys.path after controlled site injection",
    )

    import habitat_sim
    from habitat_sim import audio, bindings
    from habitat_sim._ext import habitat_sim_bindings

    for module_name, module in (
        ("habitat_sim.bindings", bindings),
        ("habitat_sim.audio", audio),
        ("habitat_sim", habitat_sim),
    ):
        _require(
            getattr(module, "RLR_ADAPTER_ENABLED", None) is False,
            f"{module_name} did not report default-off RLR capability",
        )
        for symbol in _RLR_SYMBOLS:
            _require(
                getattr(module, symbol, object()) is None,
                f"{module_name}.{symbol} is not the default-off None stub",
            )

    for module_name, module in (("habitat_sim.audio", audio), ("habitat_sim", habitat_sim)):
        exported = getattr(module, "__all__", ())
        _require(
            all(symbol not in exported for symbol in _RLR_SYMBOLS),
            f"{module_name} star-exports an unavailable RLR symbol",
        )

    module_files = {
        "habitat_sim": _module_file(habitat_sim),
        "habitat_sim.bindings": _module_file(bindings),
        "habitat_sim.audio": _module_file(audio),
        "habitat_sim_bindings": _module_file(habitat_sim_bindings),
    }
    for module_name, module_file in module_files.items():
        _require(module_file is not None, f"{module_name} has no file path")
        _require(
            not _is_within(module_file, old_checkout),
            f"{module_name} was imported from the old checkout: {module_file}",
        )
    _require(
        _is_within(module_files["habitat_sim"], habitat_site),
        "habitat_sim facade did not come from the requested staging site",
    )
    _require(
        _is_within(module_files["habitat_sim_bindings"], habitat_site),
        "habitat_sim bindings did not come from the requested staging site",
    )

    for module_name, module in tuple(sys.modules.items()):
        module_file = _module_file(module)
        if module_file is not None:
            _require(
                not _is_within(module_file, old_checkout),
                f"{module_name} was loaded from the old checkout: {module_file}",
            )

    print(
        json.dumps(
            {
                "facade": module_files["habitat_sim"],
                "binding": module_files["habitat_sim_bindings"],
                "rlr_adapter_enabled": False,
                "rlr_symbols": list(_RLR_SYMBOLS),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
