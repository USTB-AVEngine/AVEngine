"""QA-09 requests and preflight evidence; native QA remains the truth authority.

The family asks for an existential reappearance over a complete ten-second
window. A single visible frame after *any* full occlusion makes the original
answer yes. Construction therefore checks the whole window, especially for no,
not only two chosen probe times. Shadow and route geometry belongs to joint_core.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

from avengine.qa.observation_predicates import full_occlusion_frames, reappeared_frames


@dataclass(frozen=True)
class ReappearanceDemand:
    target_asset_id: str
    occluder_asset_id: str
    answer: str
    window_start_s: float = 0.0
    window_duration_s: float = 10.0
    frame_rate_hz: int = 15
    minimum_hidden_s: float = 1.0
    minimum_visible_s: float = 1.0
    minimum_visible_pixels: int = 100
    minimum_target_pixels: int = 100

    def __post_init__(self) -> None:
        if self.answer not in {"yes", "no"}:
            raise ValueError("QA-09 answer must be yes or no")
        if not self.target_asset_id or not self.occluder_asset_id or self.target_asset_id == self.occluder_asset_id:
            raise ValueError("QA-09 needs distinct registered target and occluder")
        if not isinstance(self.frame_rate_hz, int) or self.frame_rate_hz <= 0:
            raise ValueError("positive integer mask rate required")
        for name in ("window_duration_s", "minimum_hidden_s", "minimum_visible_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("positive finite QA-09 durations required")
        if not math.isfinite(self.window_start_s) or self.window_start_s < 0:
            raise ValueError("nonnegative finite window start required")
        if self.window_start_s + self.window_duration_s > 30.0:
            raise ValueError("QA-09 window must lie in the thirty-second parent")
        for seconds in (self.window_start_s, self.window_duration_s):
            if abs(seconds * self.frame_rate_hz - round(seconds * self.frame_rate_hz)) > 1e-8:
                raise ValueError("window must align with actual mask frames")
        if min(self.minimum_visible_pixels, self.minimum_target_pixels) < 1:
            raise ValueError("positive construction pixel margins required")

    @property
    def frame_count(self) -> int:
        return round(self.window_duration_s * self.frame_rate_hz)

    def relation(self) -> dict[str, Any]:
        """Serialize the request for the native adapter; this is never truth."""
        return {
            "target_asset_id": self.target_asset_id,
            "occluder_asset_id": self.occluder_asset_id,
            "required_state": "fully_occluded",
            "minimum_occluded_fraction": 1.0,
            "minimum_stable_seconds": self.minimum_hidden_s,
            # Existing placement scoring needs a target root during the pause.
            # The complete-window probe below supplies the real certificate.
            "query_times_s": [self.window_start_s + 4.5 + i / self.frame_rate_hz
                              for i in range(math.ceil(self.minimum_hidden_s * self.frame_rate_hz))],
            "foreground_depth_fraction": .55,
            "preferred_lateral_m": 0.0,
            "native_mask_and_unique_registered_identity_required": True,
            "qa09_demand": asdict(self),
        }


def requested_answer(state: Mapping[str, Any], seed: int = 0) -> str:
    """Balance actual deduplicated outputs, not admitted plans or retries."""
    tables = state.get("answer_counts") or state.get("family_answer_counts") or {}
    counts = tables.get("QA-09")
    if counts is None:
        counts = state.get("answer_branch_distribution", {}).get("families", {}).get("QA-09", {}).get("answers", {})
    yes, no = (int(counts.get(answer, 0)) for answer in ("yes", "no"))
    if yes < 0 or no < 0:
        raise ValueError("negative delivered QA-09 count")
    if yes == no:
        return "yes" if seed % 2 == 0 else "no"
    return "yes" if yes < no else "no"


def _runs(indices: Sequence[int], fps: int, minimum_s: float) -> list[list[int]]:
    result: list[list[int]] = []
    current: list[int] = []
    required = math.ceil(minimum_s * fps - 1e-9)
    for index in indices:
        if current and index != current[-1] + 1:
            if len(current) >= required:
                result.append([current[0], current[-1]])
            current = []
        current.append(index)
    if len(current) >= required:
        result.append([current[0], current[-1]])
    return result


def certify_mask_sequence(
    frames: Sequence[Mapping[str, Any]],
    occluder_frames: Sequence[Mapping[str, Any]],
    demand: ReappearanceDemand,
    *,
    target_instance_id: str,
    occluder_instance_id: str,
    registered_instance_ids: Sequence[str],
) -> dict[str, Any]:
    """Check actual target-only/modal summaries before expensive media/audio.

    Indices are relative to the requested window. Original QA predicates supply
    the measured branch; one-second intervals, 100 pixels and unique registered
    identity are additional construction margins, never edits to QA thresholds.
    """
    errors: list[str] = []
    ids = {str(x) for x in registered_instance_ids}
    tid, oid = str(target_instance_id), str(occluder_instance_id)
    if tid == oid or not {tid, oid} <= ids:
        errors.append("target_or_occluder_not_uniquely_registered")
    actual_indices = [r.get("frame_index") for r in frames]
    complete = actual_indices == list(range(demand.frame_count))
    if not complete:
        errors.append("incomplete_or_unordered_window")
    known_states = {"out_of_view", "visible_clear", "visible_occluded", "fully_occluded"}
    if any(r.get("state") not in known_states for r in frames):
        errors.append("unknown_visibility_state")
    # No sparse sample can establish absence. Return before interpreting a
    # malformed sequence, including duplicate frame indices.
    if errors:
        return {"status": "fail", "errors": errors, "observed_answer": None,
                "complete_window": complete, "demand": asdict(demand)}
    fully = [int(i) for i in full_occlusion_frames(frames)]
    after = [int(i) for i in reappeared_frames(frames, fully)]
    observed = ("yes" if after else "no") if fully else None
    if not fully:
        errors.append("no_fully_occluded_frame")
    if observed != demand.answer:
        errors.append("observed_branch_differs_from_requested")
    occ: dict[int, Mapping[str, Any]] = {}
    for r in occluder_frames:
        if str(r.get("target_instance_id")) != tid:
            continue
        index = r.get("frame_index")
        if index in occ:
            errors.append("duplicate_occluder_frame")
        occ[index] = r
    hidden: list[int] = []
    clear: list[int] = []
    for r in frames:
        i = int(r["frame_index"])
        target_pixels, visible_pixels = r.get("target_pixels", -1), r.get("visible_pixels", -1)
        if target_pixels < 0 or visible_pixels < 0 or visible_pixels > target_pixels:
            errors.append("invalid_native_pixel_counts")
            continue
        if r["state"] == "visible_clear" and visible_pixels >= demand.minimum_visible_pixels:
            clear.append(i)
        if r["state"] != "fully_occluded":
            continue
        witness = occ.get(i, {})
        if (target_pixels >= demand.minimum_target_pixels and visible_pixels == 0
                and witness.get("occluder_instance_ids") == [oid]
                and witness.get("unknown_occluder_pixels") == 0
                and witness.get("target_only_pixels") == target_pixels
                and witness.get("context_pixels") == 0
                and witness.get("occluded_pixels") == target_pixels):
            hidden.append(i)
    hidden_runs = _runs(hidden, demand.frame_rate_hz, demand.minimum_hidden_s)
    clear_runs = _runs(clear, demand.frame_rate_hz, demand.minimum_visible_s)
    first_hidden = min(fully) if fully else demand.frame_count
    seen_before = [run for run in clear_runs if run[1] < first_hidden]
    seen_after = [run for run in clear_runs if any(run[0] > h[1] for h in hidden_runs)]
    if not hidden_runs:
        errors.append("no_stable_registered_full_occlusion")
    if not seen_before:
        errors.append("no_stable_visibility_before_first_occlusion")
    if demand.answer == "yes" and not seen_after:
        errors.append("no_stable_clear_reappearance")
    if demand.answer == "no" and fully and hidden != list(range(first_hidden, demand.frame_count)):
        errors.append("negative_branch_not_hidden_through_window_end")
    return {
        "status": "pass" if not errors else "fail",
        "errors": sorted(set(errors)), "observed_answer": observed,
        "complete_window": complete, "demand": asdict(demand),
        "fully_occluded_frames": fully, "reappeared_frames": after,
        "registered_hidden_intervals_frames": hidden_runs,
        "clear_before_intervals_frames": seen_before,
        "clear_after_intervals_frames": seen_after,
        "scope": "construction preflight; final original QA independently evaluates native full stream",
    }


def probe_native_sequence(video: Any, scene: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
                          demand: ReappearanceDemand) -> dict[str, Any]:
    """Reversibly sample every requested native frame after actual motion setup.

    The integration-owned native relation dispatcher calls this only once both
    requested entries are placed. It must run before final video/audio rendering.
    No prediction, mask or QA answer is written into the production truth stream.
    """
    import numpy as np
    from avengine.capture.continuous_native_visibility import NativeVisibilityStream

    if scene.get("video_fps", 15) != demand.frame_rate_hz:
        raise ValueError("QA-09 preflight must use the actual native mask frame rate")
    by_asset = {e["declaration"]["asset_id"]: e for e in entries}
    target, occluder = (by_asset[aid] for aid in (demand.target_asset_id, demand.occluder_asset_id))
    tid, oid = (str(e["declaration"]["native_instance_id"]) for e in (target, occluder))
    identities = [{"endpoint_id": e["endpoint_id"], "actor_id": e["declaration"]["native_instance_id"]}
                  for e in video.actors]
    stream = NativeVisibilityStream(video)
    frames: list[Mapping[str, Any]] = []
    try:
        for index in range(demand.frame_count):
            at = demand.window_start_s + index / demand.frame_rate_hz
            for entry in entries:
                if entry.get("motion_episode"):
                    video._apply_motion_entry(entry, round(at * 30))
            video._native_pair_pixels(scene, entries)
            modal = np.asarray(video.sim.get_sensor_observations()["semantic"]).copy()
            result = stream.capture(index, modal, identities)
            frames.append(result["per_instance"][tid]["frames"][0])
        certificate = certify_mask_sequence(frames, stream.occluder_frames, demand,
            target_instance_id=tid, occluder_instance_id=oid,
            registered_instance_ids=[str(r["actor_id"]) for r in identities])
    finally:
        for entry in entries:
            if entry.get("motion_episode"):
                video._apply_motion_entry(entry, 0)
        video._native_pair_pixels(scene, entries)
        stream.close()
    certificate["native_control_restoration"] = stream.receipt()
    certificate["native_mask_frames"] = list(frames)
    certificate["native_occluder_frames"] = [r for r in stream.occluder_frames if r["target_instance_id"] == tid]
    return certificate
