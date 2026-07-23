import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from avengine_v43.hdf5_data import (
    EXPECTED_SPLIT_COUNTS,
    HDF5_SCHEMA,
    OUTPUT_FRAME_COUNT,
    SAMPLE_COUNT,
    SOURCE_COUNT,
    Hdf5QueryBank,
    SPLIT_TO_CODE,
)
from avengine_v43.labels import LegacyV4AudioError
from train import _circular_error, _is_clap_state_key


def _write_complete_h5(path: Path, *, ambiguous_captions: bool = False) -> None:
    string_type = h5py.string_dtype(encoding="utf-8")
    split_codes = np.concatenate(
        [
            np.full(count, SPLIT_TO_CODE[split], dtype=np.uint8)
            for split, count in EXPECTED_SPLIT_COUNTS.items()
        ]
    )
    caption_ids = np.tile(
        np.asarray([0, 0 if ambiguous_captions else 1], dtype=np.uint8),
        (1000, 1),
    )
    with h5py.File(path, "x") as file:
        file.attrs["schema"] = HDF5_SCHEMA
        file.attrs["completed_sample_count"] = 1000
        file.attrs["caption_table_json"] = json.dumps(
            ["human speech", "dog barking"]
        )
        file.create_dataset(
            "mixture",
            shape=(1000, SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=(1, SAMPLE_COUNT, 2),
        )
        file.create_dataset(
            "azimuth_deg",
            data=np.zeros(
                (1000, SOURCE_COUNT, OUTPUT_FRAME_COUNT),
                dtype=np.float32,
            ),
        )
        file.create_dataset("caption_id", data=caption_ids)
        file.create_dataset("split_code", data=split_codes)
        file.create_dataset(
            "sample_id",
            data=np.asarray(
                [f"sample_{index:04d}" for index in range(1000)],
                dtype=object,
            ),
            dtype=string_type,
        )


def test_hdf5_query_contract_is_exactly_1600_200_200(tmp_path):
    path = tmp_path / "dataset.h5"
    _write_complete_h5(path)
    with Hdf5QueryBank(path) as bank:
        assert {
            split: len(bank.queries(split))
            for split in EXPECTED_SPLIT_COUNTS
        } == {
            "train": 1600,
            "validation": 200,
            "test": 200,
        }
        first_two = bank.queries("train")[:2]
        assert first_two[0].sample_id == first_two[1].sample_id
        assert first_two[0].caption != first_two[1].caption
        assert bank.read_mixtures(first_two).shape == (2, SAMPLE_COUNT, 2)


def test_hdf5_contract_rejects_ambiguous_text_queries(tmp_path):
    path = tmp_path / "ambiguous.h5"
    _write_complete_h5(path, ambiguous_captions=True)
    with pytest.raises(LegacyV4AudioError, match="distinct captions"):
        Hdf5QueryBank(path)


def test_circular_error_wraps_at_front():
    predicted = np.asarray([359.0, 1.0, 180.0])
    target = np.asarray([1.0, 359.0, 0.0])
    assert _circular_error(predicted, target).tolist() == [2.0, 2.0, 180.0]


def test_checkpoint_filter_excludes_every_clap_alias():
    assert _is_clap_state_key("CLAP.text_branch.weight")
    assert _is_clap_state_key("text_encoder.CLAP.text_branch.weight")
    assert not _is_clap_state_key("text_fc.0.weight")


def test_formal_runner_has_no_old_localization_checkpoint_cli():
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")
    assert "--checkpoint" not in source
    assert "old_localization_checkpoint" not in source


def test_training_uses_only_the_selected_cuda_rng():
    root = Path(__file__).parents[1]
    source = (root / "train.py").read_text(encoding="utf-8")
    smoke = (root / "run_training_smoke.py").read_text(encoding="utf-8")
    for forbidden in (
        "manual_seed_all",
        "get_rng_state_all",
        "set_rng_state_all",
    ):
        assert forbidden not in source
        assert forbidden not in smoke
