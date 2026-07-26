"""Single-file HDF5 query bank for the 1,000-sample AVEngine closure."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .labels import LegacyV4AudioError


HDF5_SCHEMA = "avengine_v43_binaural360_hdf5_v1"
EVALUATION_HDF5_SCHEMA = "avengine_v43_binaural360_evaluation_hdf5_v1"
EVALUATION_BUILDER_PRODUCER_SCHEMA = (
    "avengine_v43_evaluation_hdf5_builder_producer_v1"
)
EVALUATION_BUILDER_CODE_FILES = {
    "build_evaluation_hdf5.py",
    "avengine_v43/hdf5_data.py",
    "avengine_v43/labels.py",
    "run_training_smoke.py",
}
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


def _identity_digest(*, sample_ids: Sequence[str], episode_ids: Sequence[str]) -> str:
    payload = json.dumps(
        {
            "sample_ids": list(sample_ids),
            "episode_ids": list(episode_ids),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_builder_producer(value: Any) -> bool:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != EVALUATION_BUILDER_PRODUCER_SCHEMA
        or not isinstance(value.get("code"), Mapping)
        or set(value["code"]) != EVALUATION_BUILDER_CODE_FILES
        or not isinstance(value.get("runtime"), Mapping)
    ):
        return False
    if not all(
        isinstance(record, Mapping) and _valid_sha256(record.get("sha256"))
        for record in value["code"].values()
    ):
        return False
    return all(
        isinstance(value["runtime"].get(key), str)
        and bool(value["runtime"][key])
        for key in (
            "python_implementation",
            "python_version",
            "python_executable",
            "platform",
            "numpy_version",
            "h5py_version",
            "soundfile_version",
            "libsndfile_version",
        )
    )


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
        self.episode_ids = (
            tuple(
                _decode_string(value) for value in self.file["episode_id"][:]
            )
            if "episode_id" in self.file
            and tuple(self.file["episode_id"].shape) == (1000,)
            else ()
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
        input_identity = _decode_json_attribute(
            self.file.attrs["input_identity_json"]
        )
        if not isinstance(input_identity, Mapping):
            raise LegacyV4AudioError("HDF5 input identity is invalid")
        return {
            "path": str(self.path),
            "schema": _decode_string(self.file.attrs["schema"]),
            "sample_count": int(self.file.attrs["completed_sample_count"]),
            "input_identity": dict(input_identity),
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


class EvaluationQueryBank:
    """Validated variable-size held-out room evaluation cache."""

    def __init__(self, path: Path, *, preload_mixtures: bool = False):
        h5py = _load_h5py()
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.file = h5py.File(self.path, "r")
        try:
            count = self._validate()
            self.room_id = _decode_string(self.file.attrs["room_id"]).strip()
            self.input_identity = dict(
                _decode_json_attribute(self.file.attrs["input_identity_json"])
            )
            self.builder_producer_identity = dict(
                _decode_json_attribute(
                    self.file.attrs["builder_producer_identity_json"]
                )
            )
            self.caption_table = tuple(
                _decode_json_attribute(self.file.attrs["caption_table_json"])
            )
            self.sample_ids = tuple(
                _decode_string(value) for value in self.file["sample_id"][:]
            )
            self.episode_ids = tuple(
                _decode_string(value) for value in self.file["episode_id"][:]
            )
            self._mixtures = (
                np.asarray(self.file["mixture"][:], dtype=np.float32)
                if preload_mixtures
                else None
            )
            if self._mixtures is not None and (
                self._mixtures.shape != (count, SAMPLE_COUNT, 2)
                or not np.all(np.isfinite(self._mixtures))
            ):
                raise LegacyV4AudioError(
                    "preloaded evaluation HDF5 mixtures are invalid"
                )
            caption_ids = np.asarray(self.file["caption_id"][:], dtype=np.uint8)
            azimuth = np.asarray(self.file["azimuth_deg"][:], dtype=np.float32)
            refs = []
            for sample_index, sample_id in enumerate(self.sample_ids):
                for source_index in range(SOURCE_COUNT):
                    refs.append(
                        QueryRef(
                            sample_index=sample_index,
                            source_index=source_index,
                            sample_id=sample_id,
                            caption=self.caption_table[
                                int(caption_ids[sample_index, source_index])
                            ],
                            azimuth_deg=azimuth[sample_index, source_index].astype(
                                np.float64
                            ),
                        )
                    )
            self._queries = tuple(refs)
        except Exception:
            self.file.close()
            raise

    def _validate(self) -> int:
        if _decode_string(self.file.attrs.get("schema")) != EVALUATION_HDF5_SCHEMA:
            raise LegacyV4AudioError("evaluation HDF5 schema is invalid")
        count = int(self.file.attrs.get("completed_sample_count", -1))
        if count < 1:
            raise LegacyV4AudioError("evaluation HDF5 is empty or incomplete")
        raw_room_id = self.file.attrs.get("room_id")
        if not isinstance(raw_room_id, (str, bytes)):
            raise LegacyV4AudioError("evaluation HDF5 room_id is invalid")
        room_id = _decode_string(raw_room_id).strip()
        if not room_id or _decode_string(raw_room_id) != room_id:
            raise LegacyV4AudioError("evaluation HDF5 room_id is invalid")
        if (
            self.file.attrs.get("research_only") not in (True, np.bool_(True))
            or self.file.attrs.get("qualification_claim")
            not in (False, np.bool_(False))
            or _decode_string(self.file.attrs.get("split")) != "test"
            or int(self.file.attrs.get("sample_rate_hz", -1)) != 16_000
            or float(self.file.attrs.get("duration_seconds", -1.0)) != 5.0
            or int(self.file.attrs.get("source_count", -1)) != SOURCE_COUNT
            or int(self.file.attrs.get("output_frame_count", -1))
            != OUTPUT_FRAME_COUNT
        ):
            raise LegacyV4AudioError("evaluation HDF5 contract attributes are invalid")
        try:
            input_identity = _decode_json_attribute(
                self.file.attrs["input_identity_json"]
            )
            builder_producer = _decode_json_attribute(
                self.file.attrs["builder_producer_identity_json"]
            )
            caption_table = _decode_json_attribute(
                self.file.attrs["caption_table_json"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LegacyV4AudioError(
                "evaluation HDF5 JSON attributes are invalid"
            ) from exc
        if (
            not isinstance(input_identity, Mapping)
            or not isinstance(input_identity.get("dataset_index"), str)
            or not input_identity["dataset_index"]
            or input_identity["dataset_index"] != input_identity["dataset_index"].strip()
            or not isinstance(input_identity.get("trajectory_bank"), str)
            or not input_identity["trajectory_bank"]
            or input_identity["trajectory_bank"]
            != input_identity["trajectory_bank"].strip()
            or not _valid_sha256(input_identity.get("trajectory_bank_sha256"))
            or not isinstance(input_identity.get("rir_plan"), str)
            or not input_identity["rir_plan"]
            or input_identity["rir_plan"] != input_identity["rir_plan"].strip()
            or not _valid_sha256(input_identity.get("rir_plan_sha256"))
            or input_identity.get("builder_producer_identity") != builder_producer
            or not _valid_builder_producer(builder_producer)
        ):
            raise LegacyV4AudioError("evaluation HDF5 input identity is invalid")
        if (
            not isinstance(caption_table, list)
            or not caption_table
            or len(caption_table) > np.iinfo(np.uint8).max + 1
            or any(
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                for value in caption_table
            )
            or len(set(caption_table)) != len(caption_table)
        ):
            raise LegacyV4AudioError("evaluation HDF5 caption table is invalid")
        expected_shapes = {
            "mixture": (count, SAMPLE_COUNT, 2),
            "azimuth_deg": (count, SOURCE_COUNT, OUTPUT_FRAME_COUNT),
            "caption_id": (count, SOURCE_COUNT),
            "sample_id": (count,),
            "episode_id": (count,),
        }
        for key, shape in expected_shapes.items():
            if key not in self.file or tuple(self.file[key].shape) != shape:
                raise LegacyV4AudioError(
                    f"evaluation HDF5 {key} must have shape {shape}"
                )
        if (
            np.dtype(self.file["mixture"].dtype) != np.dtype(np.float32)
            or np.dtype(self.file["azimuth_deg"].dtype) != np.dtype(np.float32)
            or np.dtype(self.file["caption_id"].dtype) != np.dtype(np.uint8)
        ):
            raise LegacyV4AudioError(
                "evaluation HDF5 numeric dataset dtypes are invalid"
            )
        caption_ids = np.asarray(self.file["caption_id"][:], dtype=np.uint8)
        if (
            np.any(caption_ids[:, 0] == caption_ids[:, 1])
            or np.any(caption_ids >= len(caption_table))
        ):
            raise LegacyV4AudioError("evaluation queries must have distinct captions")
        azimuth = np.asarray(self.file["azimuth_deg"][:], dtype=np.float32)
        if (
            not np.all(np.isfinite(azimuth))
            or np.any(azimuth < 0.0)
            or np.any(azimuth >= 360.0)
        ):
            raise LegacyV4AudioError("evaluation azimuth labels are non-finite")
        raw_sample_ids = tuple(
            _decode_string(value) for value in self.file["sample_id"][:]
        )
        raw_episode_ids = tuple(
            _decode_string(value) for value in self.file["episode_id"][:]
        )
        sample_ids = tuple(value.strip() for value in raw_sample_ids)
        episode_ids = tuple(value.strip() for value in raw_episode_ids)
        if (
            any(not value for value in sample_ids)
            or sample_ids != raw_sample_ids
            or len(set(sample_ids)) != count
            or any(not value for value in episode_ids)
            or episode_ids != raw_episode_ids
            or len(set(episode_ids)) != count
        ):
            raise LegacyV4AudioError(
                "evaluation HDF5 sample_id/episode_id values are invalid"
            )
        return count

    def queries(self, split: str) -> Sequence[QueryRef]:
        if split != "test":
            raise LegacyV4AudioError("evaluation HDF5 contains only the test split")
        return self._queries

    def read_mixtures(self, refs: Sequence[QueryRef]) -> np.ndarray:
        if self._mixtures is not None:
            values = np.asarray(
                self._mixtures[[ref.sample_index for ref in refs]], dtype=np.float32
            )
        else:
            by_index = {
                ref.sample_index: np.asarray(
                    self.file["mixture"][ref.sample_index], dtype=np.float32
                )
                for ref in refs
            }
            values = np.stack([by_index[ref.sample_index] for ref in refs])
        if values.shape != (len(refs), SAMPLE_COUNT, 2) or not np.all(
            np.isfinite(values)
        ):
            raise LegacyV4AudioError("evaluation mixture batch is invalid")
        return values

    def identity(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "schema": EVALUATION_HDF5_SCHEMA,
            "room_id": self.room_id,
            "sample_count": len(self.sample_ids),
            "episode_count": len(self.episode_ids),
            "sample_episode_sequence_sha256": _identity_digest(
                sample_ids=self.sample_ids,
                episode_ids=self.episode_ids,
            ),
            "input_identity": dict(self.input_identity),
            "builder_producer_identity": dict(
                self.builder_producer_identity
            ),
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


def open_query_bank(path: Path, *, preload_mixtures: bool = False):
    """Open a fixed training closure or a variable-size evaluation cache."""

    h5py = _load_h5py()
    resolved = path.resolve()
    with h5py.File(resolved, "r") as file:
        schema = _decode_string(file.attrs.get("schema"))
    if schema == HDF5_SCHEMA:
        return Hdf5QueryBank(resolved, preload_mixtures=preload_mixtures)
    if schema == EVALUATION_HDF5_SCHEMA:
        return EvaluationQueryBank(resolved, preload_mixtures=preload_mixtures)
    raise LegacyV4AudioError(f"unsupported query-bank schema: {schema}")
