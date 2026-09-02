"""Historical assemblers refuse to run without an explicit reproduction flag,
while the helpers the current generators need live in a neutral module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

import design_qa_v3_pilot_batch as legacy_batch  # noqa: E402
import filter_cross_time_points as legacy_filter  # noqa: E402
import qa_v3_actor_selection as selection  # noqa: E402


def test_legacy_mains_refuse_without_the_reproduction_flag(tmp_path, capsys):
    assert legacy_batch.main(["--output-root", str(tmp_path / "out"), "--seed", "s",
                              "--params", str(tmp_path / "params.json")]) == 2
    assert "historical" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    assert legacy_filter.main(["--inputs-root", str(tmp_path), "--programs-dir", str(tmp_path),
                               "--params", str(tmp_path / "params.json"),
                               "--out", str(tmp_path / "out.json")]) == 2
    assert "historical" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_selection_helpers_have_one_home_and_stay_importable_from_the_legacy_module():
    assert legacy_batch._actor_entry is selection._actor_entry
    assert legacy_batch._selection_doc is selection._selection_doc
    assert legacy_batch._mesh_package_for is selection._mesh_package_for
    import design_qa_v3_scene_batch as scene_batch
    import build_qa_v3_n_actor_canary as n_actor
    assert scene_batch._selection_doc is selection._selection_doc
    assert n_actor._actor_entry is selection._actor_entry
