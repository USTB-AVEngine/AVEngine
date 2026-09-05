"""Finish one captured research Episode without duplicating its media per QA."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from avengine.qa.unified_catalog import generate_unified_questions, normalize_episode_bundle
from avengine.rooms.qa_episode import read_json, write_json
from avengine.rooms.qa_evidence import (
    audit_native_structural_clearance, build_pixel_appearance_review, derive_actor_occluders,
    review_imported_pose_clearance,
)


def build_audio_command(
    request: Mapping[str, Any], plan: Mapping[str, Any], episode_root: Path,
    audio_root: Path, *, repository: Path,
) -> list[str]:
    runtime = request["runtime"]
    command = [
        sys.executable, str(repository / "tools/acoustics/render_frame_readback_sequential_speech.py"),
        "--frame-readbacks", str(episode_root / "capture/frame_readbacks.json"),
        "--package-manifest", str(plan["resources"]["acoustic_package"]),
        "--voice-binding", str(episode_root / "plan/voice_bindings.json"),
        "--audio-plan", str(episode_root / "plan/episode_plan.json"),
        "--output", str(audio_root),
        "--runtime-prefix", str(runtime["runtime_prefix"]),
        "--rlr-sdk-root", str(runtime["rlr_sdk_root"]),
        "--magnum-python-site", str(runtime["magnum_python_site"]),
        "--rir-cache", str(audio_root.parent / (audio_root.name + "_rir_cache")),
        "--rir-stride", str(request.get("rir_stride", 3)),
    ]
    if runtime.get("hrtf"):
        command += ["--hrtf", str(runtime["hrtf"])]
    return command


def finalize_qa_episode(
    episode_root: Path, derived_root: Path, *, repository: Path,
    request: Mapping[str, Any] | None = None, audio_report: Path | None = None,
    appearance_review: Path | None = None,
) -> dict[str, Any]:
    """Reuse completed native capture; create a fresh audio/QA/export attempt."""
    from avengine.dataset.episode_export import export_episode_bundle

    root, derived = episode_root.resolve(), derived_root.resolve()
    if derived.exists():
        raise FileExistsError(f"refusing existing derived output: {derived}")
    plan_path = root / "plan/episode_plan.json"
    plan = read_json(plan_path)
    request = dict(request or read_json(root / "request.json"))
    cap = root / "capture"
    native_receipt = read_json(cap / "research_receipt.json")
    if native_receipt.get("native_pixel", {}).get("status") != "pass":
        raise ValueError("complete native RGB/depth/instance capture is required before finalization")
    frame_path = cap / "frame_readbacks.json"
    frame_readbacks = read_json(frame_path)
    if len(frame_readbacks.get("camera", [])) != int(plan["clock"]["frame_count"]):
        raise ValueError("native readbacks do not cover the episode clock")
    derived.mkdir(parents=True)
    commands = {}
    if audio_report is None:
        audio_root = derived / "audio"
        command = build_audio_command(request, plan, root, audio_root, repository=repository)
        commands["audio"] = command
        write_json(derived / "commands.json", commands)
        with (derived / "audio.log").open("x") as log:
            subprocess.run(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT, check=True)
        audio_report = audio_root / "research_report.json"
    report = read_json(audio_report)
    for key, expected in (("frame_readbacks", frame_path), ("audio_plan", plan_path)):
        supplied = report.get(key)
        if not isinstance(supplied, str) or Path(supplied).resolve() != expected.resolve():
            raise ValueError(f"reused audio {key} does not reference this captured Episode")
    if report.get("input_capture_status") == "visual_failed":
        raise ValueError("audio rendered from a failed visual capture cannot finalize an Episode")
    mixture = Path(report["mixture_path"]).resolve()
    program_path = Path(report["audio_program"]).resolve()
    if appearance_review is None:
        review = build_pixel_appearance_review(cap, plan)
        appearance_review = derived / "appearance_review.json"
        write_json(appearance_review, review)
    else:
        review = read_json(appearance_review)
    truth_path = cap / "pixel_visibility_truth.json"
    truth = read_json(truth_path)
    occluders = derive_actor_occluders(cap / "native_pixel_masks_depth_authority_v1.npz", truth)
    occluder_path = derived / "actor_occluders.json"
    write_json(occluder_path, occluders)
    occluder_registry = {
        aid: {"display_label": f"person in the {value['value']} top"}
        for aid, value in review.get("actors", {}).items()
        if value.get("status") in {"pass", "reviewed", "astra_reviewed"}
    }
    audio_readback = {
        **report, "channel_count": 2, "sample_rate_hz": plan["clock"]["sample_rate_hz"],
        "sample_count": plan["clock"]["sample_count"], "channel_order": ["left", "right"],
        "proof": "actual_lossless_stereo_WAV_readback",
    }
    raw = {
        "episode_id": plan["episode_id"], "plan": plan,
        "frame_readbacks": frame_readbacks, "pixel_visibility_truth": truth,
        "audio_program": read_json(program_path), "audio_readback": audio_readback,
        "research_report": report,
        "voice_bindings": plan["voice_bindings"], "appearance_review": review,
        "occluder_evidence": occluders, "occluder_registry": occluder_registry,
        "frame_readbacks_path": str(frame_path), "pixel_visibility_truth_path": str(truth_path),
    }
    facts = normalize_episode_bundle(raw)
    questions = generate_unified_questions(facts, qa_ids=request.get("qa_ids"), seed=str(plan["seed"]))
    questions.pop("input_facts", None)
    facts_path = derived / "facts.json"
    write_json(facts_path, facts)
    questions["normalized_facts_path"] = str(facts_path)
    questions_path = derived / "questions.json"
    write_json(questions_path, questions)
    write_json(derived / "input_refs.json", {
        "plan": str(plan_path), "frame_readbacks": str(frame_path),
        "pixel_visibility_truth": str(truth_path), "audio_program": str(program_path),
        "audio_report": str(audio_report.resolve()), "appearance_review": str(appearance_review.resolve()),
        "occluder_evidence": str(occluder_path), "occluder_registry": occluder_registry,
    })
    preview = derived / "preview.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-i", str(cap / "ue_visual_only.mp4"), "-i", str(mixture),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
               "-ac", "2", "-ar", "16000", "-movflags", "+faststart", str(preview)]
    commands["preview"] = command
    subprocess.run(command, check=True)
    write_json(derived / "commands.json", commands)
    evidence = [
        {"path": str(frame_path), "role": "native_frame_readbacks", "required": True},
        {"path": str(truth_path), "role": "question_pixel_evidence", "required": True},
        {"path": str(audio_report.resolve()), "role": "audio_render_report", "required": True},
        {"path": str(program_path), "role": "actual_audio_program", "required": True},
        {"path": str(appearance_review.resolve()), "role": "coarse_color_observability", "required": True},
        {"path": str(occluder_path), "role": "actor_occluder_evidence", "required": True},
        {"path": str(cap / "research_receipt.json"), "role": "native_capture_receipt", "required": True},
        {"path": str(preview), "role": "AAC_preview_model_uses_separate_lossless_WAV", "required": True},
        {"path": str(facts_path), "role": "private_normalized_facts", "required": True},
    ]
    for name in ["metric_depth_native.npz", "normal_object_ids_uint32.npz",
                 "native_pixel_masks_depth_authority_v1.npz"]:
        evidence.append({"path": str(cap / name), "role": name,
                         "retention": "extended", "required": True})
    # Replay parameters and spatial/clock indexes outlive the numeric RIR cache.
    rir_metadata = set()
    keyframes = audio_report.parent / "dynamic_rir_keyframes.json"
    if keyframes.is_file():
        rir_metadata.add(keyframes.resolve())
    cache_info = report.get("dynamic_rir", {}).get("cache", {})
    cache_path = cache_info.get("path")
    if cache_path and Path(cache_path).is_dir():
        rir_metadata.update(path.resolve() for path in Path(cache_path).rglob("*.json"))
    for key, value in cache_info.items():
        if key.endswith("_path") and isinstance(value, str) and Path(value).is_file():
            rir_metadata.add(Path(value).resolve())
    for path in sorted(rir_metadata):
        evidence.append({"path": str(path), "role": "retained_RIR_replay_metadata", "required": True})
    audio_folder = Path(report.get("audio_root", audio_report.parent / "audio"))
    if audio_folder.exists():
        evidence.append({"path": str(audio_folder.resolve()), "role": "source_dry_wet_stems",
                         "retention": "extended", "required": True})
    layout = read_json(root / "plan/room_layout.json")
    structure = audit_native_structural_clearance(frame_readbacks, layout)
    structure = review_imported_pose_clearance(structure, frame_readbacks, layout, plan)
    structure_path = derived / "structural_body_check.json"
    write_json(structure_path, structure)
    evidence.append({"path": str(structure_path), "role": "native_structural_body_check", "required": True})
    if (root / "producer_version.json").exists():
        evidence.append({"path": str(root / "producer_version.json"), "role": "producer_version", "required": True})
    shared = [{"path": plan["resources"]["manifest"], "role": "room_manifest"},
              {"path": str(Path(plan["resources"]["acoustic_package"]).parent),
               "role": "shared_acoustic_geometry_and_materials"}]
    for role, value in layout.get("resources", {}).items():
        path = value.get("resolved") if isinstance(value, Mapping) else None
        if path and Path(path).exists():
            shared.append({"path": path, "role": role})
    export_request = {
        "schema": "avengine_episode_export_request_v1",
        "evidence_retention": {"mode": "extended", "include_extended": True,
                               "preserve_required": True},
        "rooms": [{"room_id": plan["scene"]["room_id"], "shared_resources": shared,
                   "episodes": [{
                       "episode_id": plan["episode_id"], "status": "research_only",
                       "plan": str(plan_path),
                       "actual": str(cap / "native_pixel_runtime_readbacks.json"),
                       "media": {"video_master": str(cap / "ue_visual_only.mp4"),
                                 "stereo_wav": str(mixture)},
                       "qa": str(questions_path), "evidence": evidence,
                   }]}],
        "model_evaluations": [],
    }
    export_path = derived / "export_request.json"
    write_json(export_path, export_request)
    export = export_episode_bundle(request_path=export_path, output_root=derived / "export", gzip_json=True)
    result = {
        "status": "research_only", "episode_id": plan["episode_id"], "questions": questions["counts"],
        "coverage": questions["coverage"], "preview": str(preview),
        "lossless_stereo_wav": str(mixture), "export": str(derived / "export"),
        "model_evaluation": "not_run", "formal_admission": False,
        "structural_body_check": structure["status"],
        "numeric_RIR_cleanup": "pending_room_batch_completion",
        "export_manifest": export,
    }
    write_json(derived / "result.json", result)
    return result
