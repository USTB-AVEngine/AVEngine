"""M5.1 research source, event, and legacy-flag contracts."""

from .source_contracts import (
    ALL_FLAG_IDS,
    PAIR_FLAG_IDS,
    SOURCE_FLAG_IDS,
    SourceContractError,
    bind_source_manifest_hashes,
    load_source_manifest,
    sample_boundary,
    validate_source_manifest,
)

__all__ = [
    "ALL_FLAG_IDS",
    "PAIR_FLAG_IDS",
    "SOURCE_FLAG_IDS",
    "SourceContractError",
    "bind_source_manifest_hashes",
    "load_source_manifest",
    "sample_boundary",
    "validate_source_manifest",
]
