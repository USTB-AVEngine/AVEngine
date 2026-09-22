"""Everyday speech-meaning questions bound to who said it and where they were.

Three question types use an authored meaning for each recorded utterance:

* QA-26 names a speaker by appearance and asks what they said,
* QA-27 names what was said and asks what the speaker looks like,
* QA-28 names what was said and asks which direction the speaker was in.

Every speaker of a scenario must give a different answer to the same
question. Otherwise the stem names the only matching utterance and the answer
needs no binding at all: QA-26 and QA-27 without the picture, QA-28 without
spatial hearing. QA-28 also needs the speakers in different sectors, or
localizing any voice would answer it without understanding one.

The stems never state when an utterance happened; a time would identify it
without its meaning.
"""
from collections import defaultdict
from copy import deepcopy

from avengine.qa import unified_catalog as u
from avengine.qa.binding_questions import _at_event_for_emitter, BindingQuestionError
from avengine.qa.choice_support import apply_choice_support

SEMANTIC_QA_IDS = u.EXTENSION_QA_IDS
_SOURCE = "authored_utterance_semantics_with_native_audio_visual_binding"


def _domain(scenario):
    """Every answer the scenario can have, so a heard pair is two of four."""
    return {row["answer"]: {k: v for k, v in row.items() if k != "text"}
            for row in scenario.get("utterances", ())}


def _voice_record(rows):
    presets = [row[2].get("voice_preset") for row in rows]
    genders = [row[2].get("voice_gender") for row in rows]
    return {"voice_presets": presets,
            "voice_relation": "same" if len(set(presets)) == 1 else "different",
            "voice_genders": genders}


def _ask(deferred, qa_id, where, **fields):
    """Build one item; a question the evidence cannot support is recorded, not raised."""
    try:
        item = u._question_item(qa_id=qa_id, **fields)
    except u._Deferred as error:
        deferred.append({"qa_id": qa_id, **where, "code": error.code, "reason": error.detail})
        return None
    return item


def generate_semantic_questions(raw_facts, manifest, *, variant_id="semantic", seed="semantic"):
    """Use authored meaning annotations only after matching the rendered sound."""
    facts = u._restore_normalized_frame_keys(raw_facts)
    sounds = {r["sound_asset_id"]: r for r in manifest["sounds"]}
    scenarios = {s["id"]: s for s in manifest["dialogues"]["scenarios"]}
    groups = defaultdict(list)
    deferred = []
    try:
        reviewed = u._reviewed_appearances(facts)
    except u._Deferred as error:
        # No speaker can be named by appearance (off screen, say); QA-28 needs none.
        reviewed = {}
        deferred.append({"qa_id": "QA-26", "code": error.code, "reason": error.detail})
    for event in facts["events"]:
        sound = sounds.get(event.get("sound_asset_id"))
        if sound is None or sound.get("scenario_id") not in scenarios:
            continue
        if event.get("transcript") != sound.get("transcript"):
            raise ValueError("semantic annotation transcript differs from the rendered event")
        groups[sound["scenario_id"]].append((event.get("actor_id"), event, sound))
    try:
        u._require_stereo(facts)
        spatial = True
    except Exception as error:  # a mono render cannot say where anyone was
        spatial = False
        deferred.append({"qa_id": "QA-28", "reason": f"no spatial audio: {error}"})
    items = []
    for scenario_id, rows in groups.items():
        scenario = scenarios[scenario_id]
        by_actor = defaultdict(list)
        for row in rows:
            by_actor[row[0]].append(row)
        if len(by_actor) < 2:
            deferred.append({"scenario_id": scenario_id, "reason": "fewer than two speakers of this scenario"})
            continue
        if any(len({r[2]["semantic_answer"]["answer"] for r in rr}) != 1 for rr in by_actor.values()):
            deferred.append({"scenario_id": scenario_id,
                             "reason": "one actor gives multiple answers; an explicit event query is required"})
            continue
        chosen = [rr[0] for rr in by_actor.values()]
        heard = [r[2]["semantic_answer"]["answer"] for r in chosen]
        if len(set(heard)) != len(heard):
            deferred.append({"scenario_id": scenario_id, "reason": "two speakers give the same answer"})
            continue
        domain = _domain(scenario)
        for answer in heard:
            domain.setdefault(answer, next(r[2]["semantic_answer"] for r in chosen
                                           if r[2]["semantic_answer"]["answer"] == answer))
        options = [{"value": key, "label_en": a["label_en"], "label_zh": a["label_zh"], "allow_value": False}
                   for key, a in domain.items()]
        classes = {key: list(dict.fromkeys([a["label_en"], a["label_zh"], *a.get("aliases", [])]))
                   for key, a in domain.items()}
        common = {"scenario_id": scenario_id, "semantic_slot": scenario["slot"],
                  "speaker_count": len(chosen), "heard_answers": heard,
                  "candidate_actor_ids": list(by_actor),
                  "candidate_semantic_answers": {a: r[0][2]["semantic_answer"]["answer"]
                                                 for a, r in by_actor.items()},
                  "variant_id": variant_id, **_voice_record(chosen),
                  "remaining_review": ["speech_content_readback", "paired_media_checks",
                                       "model_modality_ablations"]}

        # Which speakers the picture can name: reviewed and visible when they speak.
        visible = {}
        for actor, event, sound in chosen:
            if actor not in reviewed:
                continue
            try:
                visible[actor] = _at_event_for_emitter(facts, event)
            except BindingQuestionError as error:
                deferred.append({"qa_id": "QA-26", "event_id": event["event_id"], "reason": str(error)})
        phrases = {actor: u._appearance_phrases(reviewed[actor]) for actor in visible}
        distinct_looks = len({p[0] for p in phrases.values()}) == len(phrases)
        for actor, event, sound in chosen:
            if actor not in visible:
                continue
            if not distinct_looks:
                deferred.append({"qa_id": "QA-26", "scenario_id": scenario_id,
                                 "reason": "two visible speakers share an appearance description"})
                break
            appearance_en, appearance_zh = phrases[actor]
            semantic = sound["semantic_answer"]
            evidence = {**common, "target_actor_id": actor, "event_id": event["event_id"],
                        "anchor_frame": visible[actor], "sound_asset_id": sound["sound_asset_id"],
                        "authored_semantic_answer": deepcopy(semantic), "matched_transcript": event["transcript"],
                        "option_domain": "scenario_answers", "visible_speaker_count": len(visible)}
            item = _ask(deferred, "QA-26", {"scenario_id": scenario_id, "actor_id": actor},
                facts=facts, seed=seed,
                question_en=scenario["question_en"].format(appearance=appearance_en),
                question_zh=scenario["question_zh"].format(appearance=appearance_zh),
                open_answer_type="closed_set", open_truth=semantic["answer"], truth_label=semantic["label_en"],
                options=options, open_extra={"classes": classes}, evidence=evidence,
                slug=f"{variant_id}_{scenario_id}_{actor}_meaning")
            if item is not None:
                item["question_variant"] = "appearance_to_speech_meaning"
                item["required_modalities"] = ["audio", "video"]
                item["cross_modal_necessity_claim"] = False
                item["truth"]["source"] = _SOURCE
                items.append(apply_choice_support(item))

            # The reverse asks for a visual description, never a hidden actor ID or
            # a voice label that audio alone supplies. With one visible speaker the
            # picture would answer it by elimination, so it needs two.
            if len(visible) < 2:
                deferred.append({"qa_id": "QA-27", "scenario_id": scenario_id,
                                 "reason": "only one speaker is visible; the picture answers by elimination"})
                continue
            reverse = _ask(deferred, "QA-27", {"scenario_id": scenario_id, "actor_id": actor},
                facts=facts, seed=seed,
                question_en=scenario.get("reverse_question_en",
                                         "Which person's statement conveys {answer}? Describe their appearance."
                                         ).format(answer=semantic["label_en"]),
                question_zh=scenario.get("reverse_question_zh",
                                         "哪位说话者表达了“{answer}”这个意思？请描述其外观。"
                                         ).format(answer=semantic["label_zh"]),
                open_answer_type="closed_set", open_truth=reviewed[actor]["value"], truth_label=appearance_en,
                options=u._appearance_options(facts, include_values=[reviewed[a]["value"] for a in visible]),
                evidence={**evidence, "option_domain": "visible_speaker_appearances"},
                slug=f"{variant_id}_{scenario_id}_{actor}_meaning_to_appearance")
            if reverse is not None:
                reverse["question_variant"] = "speech_meaning_to_appearance"
                reverse["required_modalities"] = ["audio", "video"]
                reverse["cross_modal_necessity_claim"] = False
                reverse["truth"]["source"] = _SOURCE
                items.append(apply_choice_support(reverse))

        if not spatial:
            continue
        sectors = {}
        for actor, event, sound in chosen:
            window = u._event_start_sector_window(facts, event)
            if window is not None:
                sectors[actor] = window
        if len(sectors) != len(chosen):
            deferred.append({"qa_id": "QA-28", "scenario_id": scenario_id,
                             "reason": "a speaker starts inside a sector dead zone"})
            continue
        if len({w[1] for w in sectors.values()}) != len(sectors):
            deferred.append({"qa_id": "QA-28", "scenario_id": scenario_id,
                             "reason": "speakers share a sector; localizing any voice would answer it"})
            continue
        for actor, event, sound in chosen:
            window, sector, angle = sectors[actor]
            semantic = sound["semantic_answer"]
            # Seeded by what was said, never by who said it or by which episode
            # this is: a paired group's appearance variants are separate episodes,
            # and every variant must ask the same stem for its answers to be
            # compared. The order is independent of the sector, which is geometry.
            names_en, names_zh, order = u._named_alternatives(
                seed, "QA-28", f"{scenario_id}_{semantic['answer']}", u._DIRECTION_SECTORS,
                episode_id=scenario_id)
            evidence = {**common, "target_actor_id": actor, "event_id": event["event_id"],
                        "sound_asset_id": sound["sound_asset_id"],
                        "authored_semantic_answer": deepcopy(semantic), "matched_transcript": event["transcript"],
                        "query_frame": window[0], "azimuth_deg": angle,
                        "speaker_sectors": {a: w[1] for a, w in sectors.items()},
                        "sector_dead_zone_deg": u._SECTOR_DEAD_ZONE_DEG,
                        "answer_domain": "eight_45_degree_sectors", "wording_order": order,
                        "target_visible": actor in visible}
            stem_zh = scenario.get("direction_question_zh",
                                   "表达了“{answer}”这个意思的那位说话者，在你的哪个方向？").format(answer=semantic["label_zh"])
            stem_en = scenario.get("direction_question_en",
                                   "The speaker who meant {answer}: which direction are they from you?"
                                   ).format(answer=semantic["label_en"])
            item = _ask(deferred, "QA-28", {"scenario_id": scenario_id, "actor_id": actor},
                facts=facts, seed=seed,
                question_en=(f"{stem_en} Take the moment they start saying it, relative to the way you are "
                             f"facing, and answer with exactly one of: {', '.join(names_en)}."),
                question_zh=f"{stem_zh}以这位说话者开口说这句话时、你的朝向为准，请只回答其中之一：{'、'.join(names_zh)}。",
                open_answer_type="closed_set", open_truth=sector, truth_label=sector,
                options=[u._option(value, label_en) for value, label_en, _zh in u._DIRECTION_SECTORS],
                evidence=evidence, slug=f"{variant_id}_{scenario_id}_{actor}_meaning_to_direction")
            if item is not None:
                item["question_variant"] = "speech_meaning_to_direction"
                item["required_modalities"] = ["audio"]
                item["cross_modal_necessity_claim"] = False
                item["truth"]["source"] = "authored_utterance_semantics_with_rendered_listener_azimuth"
                items.append(apply_choice_support(item))
    return {"items": items, "deferred": deferred, "episode_id": facts["episode_id"],
            "scoring_policy": {"form_denominator": "offered_forms"},
            "claim_boundary": "Meaning annotations matched to rendered speech, observed identities and rendered "
                              "listener azimuth; empirical modality necessity is not claimed by this generator."}
