from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avengine.qa.design_viewpoints import (
    DesignViewpointError,
    load_design_viewpoint_bank,
)
from tools.m6x import finalize_apartment_viewpoint_review as finalizer


def _inputs(tmp_path: Path, *, retained_count: int) -> tuple[Path, Path]:
    source = tmp_path / "source"
    requests = source / "requests"
    requests.mkdir(parents=True)
    viewpoints = []
    request_paths = {}
    decisions = []
    for index in range(6):
        viewpoint_id = f"view-{index}"
        request = requests / f"{viewpoint_id}.json"
        request.write_text("{}\n", encoding="utf-8")
        request_paths[viewpoint_id] = f"requests/{viewpoint_id}.json"
        camera = {
            "habitat_translation_m": [float(index), 1.2, float(index + 1)],
            "habitat_yaw_deg": float(index * 30),
            "ue_location_cm": [float(index * 100), float((index + 1) * 100), 120.0],
            "ue_yaw_deg": float(index * 30),
        }
        viewpoints.append(
            {
                "viewpoint_id": viewpoint_id,
                "camera": camera,
                "native_pixel_validation_status": "pending",
                "native_validation_pending": ["first_person_composition"],
                "corridor_prefilter_pass_ids": ["c0"],
            }
        )
        decisions.append(
            {
                "viewpoint_id": viewpoint_id,
                "decision": "retain" if index < retained_count else "reject",
                "reason": "proxy review retained"
                if index < retained_count
                else "duplicate",
                "camera": camera,
            }
        )
    prefilter = source / "prefilter.json"
    prefilter.write_text(
        json.dumps(
            {
                "artifact_status": "deterministic_2d_raster_prefilter_only",
                "native_pixel_validation_status": "pending",
                "studio_used": False,
                "corridors": [],
                "viewpoints": viewpoints,
                "request_paths": request_paths,
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"decisions": decisions}), encoding="utf-8")
    return prefilter, review


def test_three_retained_proxies_publish_only_truthful_blocker_no_clobber(
    tmp_path: Path,
) -> None:
    prefilter, review = _inputs(tmp_path, retained_count=3)
    output = tmp_path / "blocked-output"
    args = SimpleNamespace(prefilter=prefilter, review_decision=review, output=output)

    published, status = finalizer.run(args)

    assert published == output
    assert status == "blocked"
    blocker = json.loads((output / "BLOCKED.json").read_text(encoding="utf-8"))
    assert blocker["status"] == "blocked_insufficient_retained_proxy_viewpoints"
    assert blocker["retained_proxy_count"] == 3
    assert blocker["episode_counted"] is False
    assert not (output / "viewpoint_bank_apartment_v1.json").exists()
    with pytest.raises(
        finalizer.ApartmentViewpointReviewFinalizerError,
        match="refusing to replace output or staging path",
    ):
        finalizer.run(args)


def test_four_proxy_bank_stays_pending_and_production_loader_rejects(
    tmp_path: Path,
) -> None:
    prefilter, review = _inputs(tmp_path, retained_count=4)
    output = tmp_path / "pending-bank-output"

    published, status = finalizer.run(
        SimpleNamespace(
            prefilter=prefilter,
            review_decision=review,
            output=output,
        )
    )

    assert published == output
    assert status == "pending_bank"
    bank = output / "viewpoint_bank_apartment_v1.json"
    document = json.loads(bank.read_text(encoding="utf-8"))
    assert document["retained_proxy_count"] == 4
    assert document["native_pixel_validation_status"] == "pending"
    assert all(
        item["native_pixel_validation_status"] == "pending"
        for item in document["viewpoints"]
    )
    with pytest.raises(
        DesignViewpointError,
        match="native_pixel_validation_status=pending",
    ):
        load_design_viewpoint_bank(bank)
