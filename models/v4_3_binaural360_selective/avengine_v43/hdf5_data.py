"""Single-file HDF5 query bank for the 1,000-sample AVEngine closure."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .labels import LegacyV4AudioError


HDF5_SCHEMA = "avengine_v43_binaural360_hdf5_v1"
EXPECTED_SPLIT_COUNTS = {
    "train": 800,
    "validation": 100,
    "test": 100,
}
SPLIT_TO_CODE = {
    "train": 0,
    "validation": 1,
    "test": 2,
}
SAMPLE_COUNT = 80_000
OUTPUT_FRAME_COUNT = 75
SOURCE_COUNT = 2


def _load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for the model dataset cache") from exc
    return h5py


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_json_attribute(value: Any) -> Any:
    return json.loads(_decode_string(value))


@dataclass(frozen=True)
class QueryRef:
    """One text target over a shared two-source mixture."""

    sample_index: int
    source_index: int
    sample_id: str
    caption: str
    azimuth_deg: np.ndarray


class Hdf5QueryBank:
    """Validated read-only access to one HDF5 training closure."""

    def __init__(self, path: Path, *, preload_mixtures: bool = False):
        h5py = _load_h5py()
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.file = h5py.File(self.path, "r")
        self._validate()
        self._mixtures = (
            np.asarray(self.file["mixture"][:], dtype=np.float32)
            if preload_mixtures
            else None
        )
        if self._mixtures is not None and self._mixtures.shape != (
            1000,
            SAMPLE_COUNT,
            2,
        ):
            raise LegacyV4AudioError("preloaded HDF5 mixture shape is invalid")
        self.caption_table = tuple(
            _decode_json_attribute(self.file.attrs["caption_table_json"])
        )
        self.sample_ids = tuple(
            _decode_string(value) for value in self.file["sample_id"][:]
        )
        split_codes = np.asarray(self.file["split_code"][:], dtype=np.uint8)
        caption_ids = np.asarray(self.file["caption_id"][:], dtype=np.uint8)
        azimuth = np.asarray(self.file["azimuth_deg"][:], dtype=np.float32)
        self._queries_by_split: dict[str, list[QueryRef]] = {}
        for split, code in SPLIT_TO_CODE.items():
            refs: list[QueryRef] = []
            for sample_index in np.flatnonzero(split_codes == code):
                for source_index in range(SOURCE_COUNT):
                    refs.append(
                        QueryRef(
                            sample_index=int(sample_index),
                            source_index=source_index,
                            sample_id=self.sample_ids[int(sample_index)],
                            caption=self.caption_table[
                                int(caption_ids[sample_index, source_index])
                            ],
                            azimuth_deg=azimuth[
                                sample_index,
                                source_index,
                            ].astype(np.float64),
                        )
                    )
            if len(refs) != EXPECTED_SPLIT_COUNTS[split] * SOURCE_COUNT:
                raise LegacyV4AudioError(
                    f"{split} HDF5 query count is invalid: {len(refs)}"
                )
            self._queries_by_split[split] = refs

    def _validate(self) -> None:
        file = self.file
        if _decode_string(file.attrs.get("schema")) != HDF5_SCHEMA:
            raise LegacyV4AudioError("HDF5 dataset schema is invalid")
        expected_shapes = {
            "mixture": (1000, SAMPLE_COUNT, 2),
            "azimuth_deg": (1000, SOURCE_COUNT, OUTPUT_FRAME_COUNT),
            "caption_id": (1000, SOURCE_COUNT),
            "split_code": (1000,),
            "sample_id": (1000,),
        }
        for key, expected_shape in expected_shapes.items():
            if key not in file or tuple(file[key].shape) != expected_shape:
                raise LegacyV4AudioError(
                    f"HDF5 dataset {key} must have shape {expected_shape}"
                )
        if int(file.attrs.get("completed_sample_count", -1)) != 1000:
            raise LegacyV4AudioError("HDF5 dataset cache is incomplete")
        split_codes = np.asarray(file["split_code"][:], dtype=np.uint8)
        for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
            actual_count = int(np.sum(split_codes == SPLIT_TO_CODE[split]))
            if actual_count != expected_count:
                raise LegacyV4AudioError(
                    f"{split} HDF5 sample count is {actual_count}, "
                    f"expected {expected_count}"
                )
        caption_ids = np.asarray(file["caption_id"][:], dtype=np.uint8)
        if np.any(caption_ids[:, 0] == caption_ids[:, 1]):
            raise LegacyV4AudioError(
                "HDF5 text-only mixtures must have two distinct captions"
            )
        if not np.all(np.isfinite(file["azimuth_deg"][:])):
            raise LegacyV4AudioError("HDF5 azimuth labels are non-finite")

    def queries(self, split: str) -> Sequence[QueryRef]:
        try:
            return self._queries_by_split[split]
        except KeyError as exc:
            raise LegacyV4AudioError(f"unknown split: {split}") from exc

    def read_mixtures(self, refs: Sequence[QueryRef]) -> np.ndarray:
        """Read normalized float32 mixtures for a query batch."""

        if self._mixtures is not None:
            values = np.asarray(
                self._mixtures[[ref.sample_index for ref in refs]],
                dtype=np.float32,
            )
        else:
            mixtures_by_index = {
                ref.sample_index: np.asarray(
                    self.file["mixture"][ref.sample_index],
                    dtype=np.float32,
                )
                for ref in refs
            }
            values = np.stack(
                [mixtures_by_index[ref.sample_index] for ref in refs]
            )
        if (
            values.shape != (len(refs), SAMPLE_COUNT, 2)
            or not np.all(np.isfinite(values))
        ):
            raise LegacyV4AudioError("HDF5 mixture batch is invalid")
        return values

    def identity(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "schema": _decode_string(self.file.attrs["schema"]),
            "sample_count": int(self.file.attrs["completed_sample_count"]),
            "mixtures_preloaded_to_ram": self._mixtures is not None,
        }

    def close(self) -> None:
        self._mixtures = None
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
