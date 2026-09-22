#!/usr/bin/env python3
"""Put everyday semantic speech on retained visual worlds and ask QA-26 to QA-28.

Two modes share one chooser:

* ``paired`` takes binding groups whose two visual variants swap the speakers'
  appearance (v0/v1) and renders both speaker orders (a0/a1). Every answer must
  flip when the side that changed is the side the question depends on, which
  is the direct evidence that it needs that modality.
* ``single`` takes any rendered episode with exactly two human actors and
  renders one semantic audio variant for it.

Only audio is rendered. The chooser rotates scenarios and answers least-used
first, so a constant answer cannot learn which utterance a scene tends to
carry, and gives each actor a voice of the gender its picture shows.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MALE_VOICES = ("中文男", "英文男")
FEMALE_VOICES = ("中文女",)


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def actor_genders(source):
    from avengine.rooms.qa_delivery import _asset_registry
    from avengine.dataset.source_capabilities import _gender
    plan = load(Path(source) / "plan/episode_plan.json")
    request = load(Path(source) / "request.json")
    registry = _asset_registry(ROOT, request.get("source_registry"))
    out = []
    for actor in plan["visual_plan"]["actors"]:
        record = registry[actor["asset_id"]]
        if record.get("source_class") not in (None, "articulated_human") and "human" not in actor["asset_id"]:
            raise ValueError(f"{actor['actor_id']} is not a human speaker")
        label = _gender((record.get("realized_attributes") or {}).get("sex_or_gender_label"))
        if label not in ("female", "male"):
            raise ValueError(f"{actor['asset_id']} has no readable gender to choose a voice by")
        out.append(label)
    return out


def member_name(source, scenario_id):
    """Rendered episodes all end in .../<name>/episode; name the member after <name>."""
    source = Path(source)
    return f"{source.parent.name if source.name == 'episode' else source.name}__{scenario_id}"


class Chooser:
    """Least-used scenario, then least-used answers, then a voice per gender."""

    def __init__(self, manifest, seed):
        self.sounds = {(s["scenario_id"], s["utterance_index"], s["voice_preset"]): s for s in manifest["sounds"]}
        self.scenarios = [sc["id"] for sc in manifest["dialogues"]["scenarios"]]
        self.size = {sc["id"]: len(sc["utterances"]) for sc in manifest["dialogues"]["scenarios"]}
        self.rng = random.Random(seed)
        self.scenario_use, self.answer_use, self.relation_use = Counter(), Counter(), Counter()

    def _voices(self, gender):
        return MALE_VOICES if gender == "male" else FEMALE_VOICES

    def pick(self, genders, *, exclude=()):
        order = sorted((s for s in self.scenarios if s not in exclude),
                       key=lambda s: (self.scenario_use[s], self.rng.random()))
        want_same = self.relation_use["same"] <= self.relation_use["different"]
        for sid in order:
            answers = sorted(range(self.size[sid]), key=lambda i: (self.answer_use[(sid, i)], self.rng.random()))
            for a in answers:
                for b in answers:
                    if a == b:
                        continue
                    options = [[v for v in self._voices(genders[0]) if (sid, a, v) in self.sounds],
                               [v for v in self._voices(genders[1]) if (sid, b, v) in self.sounds]]
                    if not all(options):
                        continue
                    pairs = [(x, y) for x in options[0] for y in options[1]]
                    pairs.sort(key=lambda p: ((p[0] == p[1]) != want_same, self.rng.random()))
                    va, vb = pairs[0]
                    self.scenario_use[sid] += 1
                    self.answer_use[(sid, a)] += 1
                    self.answer_use[(sid, b)] += 1
                    self.relation_use["same" if va == vb else "different"] += 1
                    return sid, [self.sounds[(sid, a, va)]["sound_asset_id"],
                                 self.sounds[(sid, b, vb)]["sound_asset_id"]]
        raise ValueError(f"no scenario has audible recordings for voices of genders {genders}")


def render_member(source, output, manifest_path, sound_ids, *, swap, seed, audio_report=None):
    from avengine.qa.audio_variants import prepare_audio_variant
    from avengine.qa.semantic_questions import generate_semantic_questions
    from avengine.qa.unified_catalog import model_input_questions
    from avengine.rooms.qa_delivery import finalize_qa_episode
    from avengine.dataset.binding_group_native import check_requested_visibility
    source, output = Path(source), Path(output)
    if (output / "result.json").is_file() and (output / "delivery/facts.json").is_file():
        # Rendered already: only the questions are asked again, from the same facts.
        result = load(output / "result.json")
    else:
        prepare_audio_variant(source, output, manifest_path, sound_ids, repository=ROOT, swap=swap, seed=seed)
        request = load(output / "request.json")
        plan = load(output / "plan/episode_plan.json")
        check_requested_visibility(plan, request, output / "capture")
        refs = load(source / "delivery/input_refs.json")
        result = finalize_qa_episode(output, output / "delivery", repository=ROOT, request=request,
                                     appearance_review=Path(refs["appearance_review"]), audio_report=audio_report)
        write(output / "result.json", result)
    facts = load(output / "delivery/facts.json")
    questions = generate_semantic_questions(facts, load(manifest_path), variant_id=output.name,
                                            seed="everyday-semantic")
    write(output / "semantic_questions.json", questions)
    write(output / "semantic_public.json", model_input_questions(questions))
    return {"member_id": output.name, "source": str(source), "episode_id": facts["episode_id"],
            "audio_report": load(output / "delivery/input_refs.json")["audio_report"],
            "video_only": str((source / "capture/ue_visual_only.mp4").resolve()),
            "preview": str(output / "delivery/preview.mp4"), "facts": str(output / "delivery/facts.json"),
            "sound_ids": sound_ids, "swap": swap, "status": result.get("status"),
            "question_count": len(questions["items"]),
            "qa_counts": dict(Counter(q["qa_id"] for q in questions["items"])),
            "deferred": questions["deferred"],
            "answers": {q["question_id"]: q["truth"]["value"] for q in questions["items"]},
            "answers_by_stem": {(q["qa_id"] + "|" + q["forms"]["open"]["question_zh"]): q["truth"]["value"]
                                for q in questions["items"]}}


def _single(job):
    try:
        return render_member(job["source"], job["output"], job["manifest"], job["sound_ids"],
                             swap=False, seed=job["seed"])
    except Exception as error:
        return {"member_id": Path(job["output"]).name, "source": job["source"], "error": repr(error),
                "traceback": traceback.format_exc(limit=6)}


def _paired(job):
    """One scenario on one binding group: v0/v1 x a0/a1, then the flip checks."""
    group, out = Path(job["group"]), Path(job["output"])
    try:
        members = {}
        for audio in ("a0", "a1"):
            report = None
            for visual in ("v0", "v1"):
                info = render_member(group / "variants" / f"{visual}_a0", out / f"{visual}_{audio}", job["manifest"],
                                     job["sound_ids"], swap=audio == "a1", seed=job["seed"], audio_report=report)
                report = Path(info["audio_report"])
                members[(visual, audio)] = info
        checks = []
        # Holding audio and swapping appearance flips the appearance-bound types
        # and must leave QA-28 alone: nobody moved. Holding video and swapping who
        # says what flips all three. A stem asked on one side must be asked on the
        # other, so no expectation can be met by comparing nothing.
        for held, pairs, flips in (
                ("audio", [(("v0", x), ("v1", x)) for x in ("a0", "a1")], ("QA-26", "QA-27")),
                ("video", [((x, "a0"), (x, "a1")) for x in ("v0", "v1")], ("QA-26", "QA-27", "QA-28"))):
            for left, right in pairs:
                l, r = members[left]["answers_by_stem"], members[right]["answers_by_stem"]
                unmatched = sorted(l.keys() ^ r.keys())
                common = sorted(l.keys() & r.keys())
                should_flip = [k for k in common if k.split("|")[0] in flips]
                should_hold = [k for k in common if k.split("|")[0] not in flips]
                flipped = [k for k in should_flip if l[k] != r[k]]
                changed = [k for k in should_hold if l[k] != r[k]]
                checks.append({"held_identical": held, "members": [members[left]["member_id"], members[right]["member_id"]],
                               "should_flip": len(should_flip), "flipped": len(flipped),
                               "should_hold": len(should_hold), "changed_anyway": len(changed),
                               "stems_on_one_side_only": unmatched,
                               "passes": bool(should_flip) and len(flipped) == len(should_flip)
                                         and not changed and not unmatched})
        return {"group": group.name, "scenario_id": job["scenario_id"], "sound_ids": job["sound_ids"],
                "members": [m for m in members.values()], "checks": checks,
                "all_checks_pass": all(c["passes"] for c in checks)}
    except Exception as error:
        return {"group": group.name, "scenario_id": job["scenario_id"], "error": repr(error),
                "traceback": traceback.format_exc(limit=6)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("paired", "single"))
    p.add_argument("--manifest", type=Path, required=True, help="admitted speech manifest")
    p.add_argument("--sources", type=Path, nargs="+", required=True,
                   help="binding group roots (paired) or rendered episode roots (single)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenarios-per-source", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--resume", action="store_true",
                   help="reuse the rendered members under --output and ask their questions again; "
                        "the same inputs must choose the same jobs")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    manifest = load(a.manifest)
    chooser = Chooser(manifest, a.seed)
    jobs, skipped = [], []
    for index, source in enumerate(a.sources):
        root = source / "variants/v0_a0" if a.mode == "paired" else source
        try:
            genders = actor_genders(root)
        except Exception as error:
            skipped.append({"source": str(source), "reason": repr(error)})
            continue
        if len(genders) != 2:
            skipped.append({"source": str(source), "reason": f"{len(genders)} visual actors, need two"})
            continue
        used = []
        for k in range(a.scenarios_per_source):
            sid, ids = chooser.pick(genders, exclude=used)
            used.append(sid)
            name = member_name(source, sid)
            job = {"manifest": str(a.manifest.resolve()), "sound_ids": ids, "scenario_id": sid,
                   "seed": a.seed + 97 * index + k, "output": str(a.output / name)}
            job.update({"group": str(source)} if a.mode == "paired" else {"source": str(source)})
            jobs.append(job)
    if a.resume and (a.output / "jobs.json").is_file():
        before = [(j["output"], j["sound_ids"]) for j in load(a.output / "jobs.json")["jobs"]]
        if before != [(j["output"], j["sound_ids"]) for j in jobs]:
            raise SystemExit("--resume chose different jobs from the ones already rendered here")
    write(a.output / "jobs.json", {"jobs": jobs, "skipped": skipped,
                                   "scenario_use": dict(chooser.scenario_use),
                                   "voice_relation_use": dict(chooser.relation_use)})
    work = _paired if a.mode == "paired" else _single
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(work, job) for job in jobs]):
            row = future.result()
            results.append(row)
            print(json.dumps({k: row.get(k) for k in ("member_id", "group", "scenario_id", "question_count",
                                                      "qa_counts", "all_checks_pass", "error")},
                             ensure_ascii=False), flush=True)
            write(a.output / "progress.json", {"results": results})
    ok = [r for r in results if "error" not in r]
    summary = {"mode": a.mode, "jobs": len(jobs), "succeeded": len(ok), "failed": len(results) - len(ok),
               "skipped_sources": len(skipped)}
    if a.mode == "paired":
        summary["all_paired_checks_pass"] = all(r["all_checks_pass"] for r in ok)
        summary["failed_checks"] = [(r["group"], r["scenario_id"]) for r in ok if not r["all_checks_pass"]]
        summary["questions"] = sum(m["question_count"] for r in ok for m in r["members"])
    else:
        summary["questions"] = sum(r["question_count"] for r in ok)
    write(a.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
