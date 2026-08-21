from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from avengine.actor_envelope import (
    ActorEnvelopeError,
    AxisAlignedBounds,
    aabb_corners,
    action_sample_times,
    build_action_envelope,
    materialize_world_aabb,
    union_action_aabb,
)
from avengine.m2.skinning import compile_skinning
from test_m2_skinning import _fixture_payload, make_document


def _compiled(tmp_path: Path):
    source = tmp_path / "actor.glb"
    source.parent.mkdir(parents=True, exist_ok=True)
    _, _, payload = _fixture_payload()
    source.write_bytes(payload)
    return compile_skinning(make_document(source_path=source))


def test_action_envelope_samples_full_action_and_binds_authority_formal_zero(
    tmp_path: Path,
) -> None:
    compiled = _compiled(tmp_path)

    envelope = build_action_envelope(
        compiled, "Walking", sample_rate_hz=2.0, padding_m=0.02
    )

    assert envelope.sample_times_seconds == (0.0, 0.5, 1.0)
    assert envelope.sampled_bounds == AxisAlignedBounds(
        minimum=(0.0, -2.0, 0.0), maximum=(2.0, 1.0, 2.0)
    )
    assert envelope.padded_bounds == AxisAlignedBounds(
        minimum=(-0.02, -2.02, -0.02), maximum=(2.02, 1.02, 2.02)
    )
    assert envelope.source_asset.path == str((tmp_path / "actor.glb").resolve())
    assert envelope.qualification_state == "planning_only"
    assert envelope.qualification_claim is False
    assert envelope.formal_eligible is False
    assert len(envelope.formal_ineligibility_reasons) == 3

    manifest = envelope.to_manifest_record()
    assert manifest["authority"]["source_asset"] == {
        "path": str((tmp_path / "actor.glb").resolve())
    }
    assert manifest["authority"]["sampling"]["sample_count"] == 3
    assert manifest["authority"]["sampling"]["coverage_kind"] == (
        "authored_keys_plus_fixed_rate_samples"
    )
    assert manifest["authority"]["sampling"]["continuous_conservative_claim"] is False
    assert manifest["qualification"]["formal_eligible"] is False


def test_action_sample_times_include_every_authored_channel_key(tmp_path: Path) -> None:
    compiled = _compiled(tmp_path)
    action = compiled.action("Walking")
    channel = action.channels[2]
    keyed_channel = replace(
        channel,
        timestamps_seconds=(0.0, 0.49, 1.0),
        values=(channel.values[0], channel.values[0], channel.values[1]),
    )
    keyed_action = replace(
        action,
        channels=(*action.channels[:2], keyed_channel),
    )
    keyed = replace(compiled, actions=(keyed_action,))

    assert action_sample_times(keyed, "Walking", sample_rate_hz=2.0) == (
        0.0,
        0.49,
        0.5,
        1.0,
    )


def test_envelope_requires_an_existing_regular_source_path(tmp_path: Path) -> None:
    compiled = _compiled(tmp_path)
    source = Path(compiled.source_path)  # type: ignore[arg-type]
    source.unlink()
    with pytest.raises(ActorEnvelopeError, match="does not exist"):
        build_action_envelope(compiled, "Walking")

    source.mkdir()
    with pytest.raises(ActorEnvelopeError, match="regular file"):
        build_action_envelope(compiled, "Walking")


def test_envelope_resolves_a_source_symlink_to_its_regular_target(
    tmp_path: Path,
) -> None:
    compiled = _compiled(tmp_path)
    source = Path(compiled.source_path)  # type: ignore[arg-type]
    alias = tmp_path / "actor_alias.glb"
    alias.symlink_to(source)

    envelope = build_action_envelope(
        compiled,
        "Walking",
        source_asset_path=alias,
    )

    assert envelope.source_asset.to_record() == {"path": str(source.resolve())}


def test_envelope_record_is_deterministic_and_reflects_bound_sampling(
    tmp_path: Path,
) -> None:
    compiled = _compiled(tmp_path)

    first = build_action_envelope(
        compiled, "Walking", sample_rate_hz=2.0, padding_m=0.02
    )
    repeated = build_action_envelope(
        compiled, "Walking", sample_rate_hz=2.0, padding_m=0.02
    )
    changed = build_action_envelope(
        compiled, "Walking", sample_rate_hz=2.0, padding_m=0.03
    )

    assert first.to_manifest_record() == repeated.to_manifest_record()
    assert changed.to_manifest_record() != first.to_manifest_record()
    assert action_sample_times(
        compiled, "Walking", sample_rate_hz=3.0
    ) == pytest.approx((0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0))


def test_union_action_aabb_rejects_ambiguous_or_non_monotonic_sampling(
    tmp_path: Path,
) -> None:
    compiled = _compiled(tmp_path)

    with pytest.raises(ActorEnvelopeError, match="cannot be empty"):
        union_action_aabb(compiled, "Walking", ())
    with pytest.raises(ActorEnvelopeError, match="strictly increasing"):
        union_action_aabb(compiled, "Walking", (0.0, 0.5, 0.5, 1.0))


def test_world_materialization_transforms_all_eight_corners() -> None:
    local = AxisAlignedBounds(minimum=(-1.0, -2.0, -3.0), maximum=(2.0, 4.0, 5.0))
    actor_from_asset = (
        (0.0, -1.0, 0.0, 10.0),
        (1.0, 0.0, 0.0, 20.0),
        (0.0, 0.0, 1.0, 30.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    world = materialize_world_aabb(local, actor_from_asset)

    assert len(aabb_corners(local)) == 8
    assert len(world.transformed_corners) == 8
    assert len(set(world.transformed_corners)) == 8
    assert world.bounds == AxisAlignedBounds(
        minimum=(6.0, 19.0, 27.0), maximum=(12.0, 22.0, 35.0)
    )
    assert world.formal_eligible is False
    assert world.qualification_claim is False

    transformed = np.asarray(world.transformed_corners)
    assert math.isclose(float(np.min(transformed[:, 0])), 6.0)
    assert math.isclose(float(np.max(transformed[:, 1])), 22.0)


def test_world_materialization_fails_closed_on_projective_or_singular_matrix() -> None:
    local = AxisAlignedBounds(minimum=(0.0, 0.0, 0.0), maximum=(1.0, 1.0, 1.0))
    projective = np.eye(4)
    projective[3, 0] = 0.1
    singular = np.eye(4)
    singular[2, 2] = 0.0

    with pytest.raises(ActorEnvelopeError, match="affine"):
        materialize_world_aabb(local, projective)
    with pytest.raises(ActorEnvelopeError, match="nonsingular"):
        materialize_world_aabb(local, singular)
