"""Unit tests for the so_time_aligned_media RoPE mode (audio+spatial path).

Runs _build_segment_positions through a minimal stub instance so no model
weights are needed. Video segments are exercised in the AV training smoke,
not here (they require the HF vision position helper).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spatial_omni.model.modeling_so_thinker import Qwen2_5OmniSpatialThinkerForConditionalGeneration  # noqa: E402

VIDEO_ID, AUDIO_ID, SPATIAL_ID, TEXT_ID = 901, 902, 903, 7


def _build(tokens, time_aligned):
    stub = Qwen2_5OmniSpatialThinkerForConditionalGeneration.__new__(
        Qwen2_5OmniSpatialThinkerForConditionalGeneration
    )
    out = stub._build_segment_positions(
        valid_tokens=torch.tensor(tokens, dtype=torch.long),
        video_token_id=VIDEO_ID,
        audio_token_id=AUDIO_ID,
        spatial_token_id=SPATIAL_ID,
        video_grid_thw=None,
        second_per_grids=None,
        spatial_merge_size=2,
        position_id_per_seconds=25,
        video_idx=0,
        time_aligned=time_aligned,
    )
    return out["position_ids"]


def _tokens(n_text_prefix=3, n_audio=10, n_spatial=4, n_text_suffix=5):
    return (
        [TEXT_ID] * n_text_prefix
        + [AUDIO_ID] * n_audio
        + [SPATIAL_ID] * n_spatial
        + [TEXT_ID] * n_text_suffix
    )


def test_legacy_mode_keeps_sequential_chaining():
    pos = _build(_tokens(), time_aligned=False)
    t = pos[0].tolist()
    # prefix text 0..2, audio 3..12, spatial 13..16, suffix 17..21
    assert t == list(range(22))


def test_aligned_mode_shares_media_origin():
    pos = _build(_tokens(), time_aligned=True)
    t = pos[0].tolist()
    assert t[:3] == [0, 1, 2]                      # text prefix unchanged
    assert t[3:13] == [3 + k for k in range(10)]   # audio: origin 3, 1 id/token
    # spatial: 4 tokens spread over the audio span (10 ids): 0,2.5,5,7.5 -> round
    assert t[13:17] == [3 + 0, 3 + 2, 3 + 5, 3 + 8]
    # suffix text continues after the global max (audio max = 12)
    assert t[17:] == [13, 14, 15, 16, 17]


def test_aligned_spatial_token_matches_cooccurring_audio_token():
    pos = _build(_tokens(n_audio=125, n_spatial=12), time_aligned=True)
    t = pos[0].tolist()
    audio_t = t[3:128]
    spatial_t = t[128:140]
    # spatial token 5 covers ~[2.08s, 2.5s) of a 5s clip; the audio token at
    # 2.1s is index ~52 -> temporal ids must land within one spatial hop.
    assert abs(spatial_t[5] - audio_t[52]) <= 6
    # strictly non-decreasing and inside the audio span
    assert spatial_t == sorted(spatial_t)
    assert spatial_t[0] == audio_t[0]
    assert spatial_t[-1] <= audio_t[-1]


def test_all_three_rows_equal_for_audio_and_spatial():
    pos = _build(_tokens(), time_aligned=True)
    assert torch.equal(pos[0], pos[1]) and torch.equal(pos[1], pos[2])
