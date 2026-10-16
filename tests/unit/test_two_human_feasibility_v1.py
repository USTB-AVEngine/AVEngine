import json
from pathlib import Path

REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "lead_a"
    / "two_human_feasibility_v1.json"
)


def test_two_human_feasibility_is_fail_closed() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["schema"] == "avengine_two_human_feasibility_v1"
    assert report["status"] == "ready_for_canary_cpu_registration"
    assert report["approved_adults"] == [
        "rocketbox_adults_male_adult_01",
        "rocketbox_adults_female_adult_01",
        "rocketbox_professions_construction_male_01",
    ]
    assert report["excluded_adults"] == [
        {
            "identity_id": "rocketbox_professions_medical_female_01",
            "reason": "evidence_contradiction",
            "evidence": report["excluded_adults"][0]["evidence"],
        }
    ]
    assert "glb_roundtrip" in report["excluded_adults"][0]["evidence"]

    canary = report["canary_pair"]
    assert canary["target_identity_id"] == "rocketbox_adults_male_adult_01"
    assert canary["distractor_identity_id"] == (
        "rocketbox_adults_female_adult_01"
    )
    assert canary["target_identity_id"] != canary["distractor_identity_id"]
    assert canary["require_distinct_original_identity"] is True
    assert canary["material_variant_identity_allowed"] is False

    assert report["children"]["voice_policy"] == "silent_only"
    assert report["children"]["allowed_as_speaking_target"] is False
    assert report["current_blocker"] == (
        "exact_two_human_rir_and_sparse_native_gate"
    )
