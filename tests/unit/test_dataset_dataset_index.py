from avengine.dataset.dataset_index import (
    assign_episode_splits,
    summarize_split_distribution,
)


def _episodes():
    rows = []
    sizes = (7, 6, 6, 6)
    for pair_index in range(4):
        source1 = f"asset_{pair_index}_source1"
        source2 = f"asset_{pair_index}_source2"
        for motion_index, size in enumerate(sizes):
            for ordinal in range(size):
                rows.append(
                    {
                        "episode_id": (
                            f"pair{pair_index}_motion{motion_index}_{ordinal:02d}"
                        ),
                        "motion_case": f"motion_{motion_index}",
                        "asset_ids_by_source_slot": {
                            "source1": source1,
                            "source2": source2,
                        },
                    }
                )
    return rows


def test_split_is_exact_deterministic_and_episode_isolated():
    episodes = _episodes()
    first = assign_episode_splits(episodes)
    second = assign_episode_splits(tuple(reversed(episodes)))
    assert first == second
    assert {split: tuple(first.values()).count(split) for split in set(first.values())} == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }


def test_split_distribution_preserves_every_asset_pair_and_motion():
    episodes = _episodes()
    report = summarize_split_distribution(
        episodes, assign_episode_splits(episodes)
    )
    assert report["train"]["episode_count"] == 80
    assert report["validation"]["episode_count"] == 10
    assert report["test"]["episode_count"] == 10
    for split in report.values():
        assert set(split["motion_case_counts"]) == {
            "motion_0",
            "motion_1",
            "motion_2",
            "motion_3",
        }
        assert len(split["ordered_asset_pair_counts"]) == 4


def test_split_supports_one_audio_variant_for_each_of_1000_visual_episodes():
    episodes = []
    for pair_index in range(4):
        for motion_index in range(4):
            for ordinal in range(62):
                episodes.append(
                    {
                        "episode_id": (
                            f"pair{pair_index}_motion{motion_index}_{ordinal:03d}"
                        ),
                        "motion_case": f"motion_{motion_index}",
                        "asset_ids_by_source_slot": {
                            "source1": f"asset_{pair_index}_source1",
                            "source2": f"asset_{pair_index}_source2",
                        },
                    }
                )
    for index in range(8):
        episodes.append(
            {
                "episode_id": f"remainder_{index:02d}",
                "motion_case": f"motion_{index % 4}",
                "asset_ids_by_source_slot": {
                    "source1": f"asset_{index % 4}_source1",
                    "source2": f"asset_{index % 4}_source2",
                },
            }
        )
    assignments = assign_episode_splits(
        episodes,
        train_count=800,
        validation_count=100,
        test_count=100,
    )
    assert len(assignments) == 1000
    assert {
        split: tuple(assignments.values()).count(split)
        for split in ("train", "validation", "test")
    } == {"train": 800, "validation": 100, "test": 100}
