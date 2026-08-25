from __future__ import annotations

import json
from pathlib import Path

from avengine.cli import main


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ROOT / "examples" / "m6" / "canary" / "controlled_one_active_of_two_request.json"


def test_registry_cli_validates_controlled_request(capsys) -> None:
    assert main(["m6", "validate-controlled-request", str(REQUEST)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "pass"
    assert output["research_only"] is True
    assert output["qualification_claim"] is False


def test_registry_cli_rejects_missing_controlled_request(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "m6",
                "validate-controlled-request",
                str(tmp_path / "missing.json"),
            ]
        )
        == 2
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "fail"
