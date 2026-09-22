#!/usr/bin/env python3
"""Generate, augment and audit the original 25 QA types from one reusable config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avengine.qa.binding_bank_merge import merge_binding_groups_into_bank
from avengine.qa.prior_bank import curate_bank
from avengine.qa.release_gate import write_release_receipt
from run_audio_answer_balance import load, write, run as run_audio


def run(config_path, output, *, resume=False):
    config = load(config_path)
    if bool(config.get("bank_in")) == bool(config.get("bank_config")):
        raise ValueError("provide exactly one of bank_in or bank_config")
    if config.get("audio_balance") and config.get("audio_summary"):
        raise ValueError("audio_balance and audio_summary are mutually exclusive")
    output = Path(output).resolve()
    if resume:
        if load(output/"run_config.json") != config:
            raise ValueError("resume config differs; use a fresh output")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output/"run_config.json", config)
    import fcntl
    with (output/".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/"complete.json").exists():
            result = load(output/"complete.json")
            # Read the real final bank, not just a launcher completion flag.
            load(Path(result["bank"])/"report.json")
            print(json.dumps({"status": "already_completed", **result}), flush=True)
            _enforce_release(config, result)
            return result
        if config.get("bank_in"):
            bank = Path(config["bank_in"]).resolve()
        else:
            bank = output/"ordinary_bank"
            command = [sys.executable, str(ROOT/"tools/dataset/generate_retained_qa_bank.py"),
                       "--config", config["bank_config"], "--output", str(bank)]
            if bank.exists():
                command.append("--resume")
            subprocess.run(command, cwd=ROOT, check=True)
            load(bank/"report.json")
        for i, export in enumerate(config.get("binding_exports", [])):
            completed = output/f"merge_{i:02d}.json"
            if completed.exists():
                bank = Path(load(completed)["bank"])
                continue
            attempt = len(list(output.glob(f"binding_{i:02d}_attempt_*"))) + 1
            merged = output/f"binding_{i:02d}_attempt_{attempt:02d}"
            merge_binding_groups_into_bank(export, merged, bank_in=bank)
            bank = merged
            write(completed, {"bank": str(bank)})
        audio_summary = None
        if config.get("audio_balance"):
            audio = output/"audio_balance"
            run_audio(config["audio_balance"], audio, resume=audio.exists())
            audio_summary = audio/"summary.json"
        elif config.get("audio_summary"):
            audio_summary = Path(config["audio_summary"])
        if audio_summary and load(audio_summary).get("status") not in {"completed", "completed_with_deficits"}:
            raise ValueError("audio plan is incomplete; finish it or export the base bank separately")
        attempt = len(list(output.glob("bank_attempt_*"))) + 1
        final = output/f"bank_attempt_{attempt:02d}"
        report = curate_bank(bank, final, audio_summary=audio_summary,
                             split_reference=config.get("split_reference"), seed=config.get("seed", 0))
        # Hold the declared standard here rather than leaving it to whoever reads the
        # receipt. The decision is written beside the bank either way; require_release
        # turns it into a failure so a production run cannot quietly ship a bank that a
        # blind constant answer already scores well on.
        policy = load(Path(config["release_policy"])) if config.get("release_policy") else None
        if (final/"private"/"splits.jsonl").is_file():
            gate = write_release_receipt(final, policy=policy)
            release_status, blocked = gate["status"], gate["blocked_rules"]
        else:
            release_status, blocked = "not_gated_without_splits", []
        result = {"bank": str(final), "question_count": report["exported_question_count"],
                  "status": report["status"], "audio_summary": str(audio_summary) if audio_summary else None,
                  "release_status": release_status, "blocked_rules": blocked}
        write(output/"complete.json", result)
        print(json.dumps(result), flush=True)
        _enforce_release(config, result)
        return result


def _enforce_release(config, result):
    """Fail the run when the config asked for a bank that meets the standard."""
    if not config.get("require_release"):
        return
    status = result.get("release_status")
    if status != "release":
        raise ValueError(
            "the bank does not meet the declared release policy (%s): %s"
            % (status, ", ".join(result.get("blocked_rules") or []) or "no split assignment")
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
