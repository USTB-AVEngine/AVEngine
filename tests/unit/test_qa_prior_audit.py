"""Prior receipts count questions once and never learn predictors from validation."""
import json
from copy import deepcopy
import pytest
from avengine.qa.prior_audit import audit_priors, split_map_from_views, write_prior_receipt


def rows(values, qa="QA-01"):
    public, answers, sources = [], [], []
    for i, value in enumerate(values):
        key = f"q{i}"
        prompt = "Did the person sound?"
        opts = [{"value": "yes", "label_en": "yes"}, {"value": "no", "label_en": "no"}]
        forms = {"open": {"answer_type": "closed_set", "truth": value, "classes": {"yes": ["yes"], "no": ["no"]}, "question_en": prompt},
                 "mcq": {"options": opts, "gold": {"correct_index": 0 if value == "yes" else 1}}}
        public.append({"question_id": key, "qa_id": qa, "forms": {"open": {"question_en": prompt}, "mcq": {"options": opts}}})
        answers.append({"question_id": key, "qa_id": qa, "forms": forms, "evidence": {}})
        sources.append({"question_id": key, "world_id": "shared_world"})
    return public, answers, sources


def test_dominance_and_binary_guessing_are_reported_not_rejected():
    p,a,s=rows(["no"]*6)
    r=audit_priors(p,a,source_rows=s)
    q=r["by_qa"]["QA-01"]
    assert q["open_answers"]["entropy_bits"] == 0
    assert q["open_answers"]["majority_share"] == 1
    assert q["mcq_random_baseline"] == .5
    assert q["questions"] == 6 and q["known_worlds"] == 1
    assert q["validation_quota_status"] == "not_checked"
    assert not q["eligible_for_supported_main_average"]
    assert r["mode"] == "report_only" and q["status"] == "under_powered"


def test_constant_is_fit_on_train_not_validation_and_forms_do_not_double_quota():
    p,a,s=rows(["no","no","yes","yes"])
    r=audit_priors(p,a,source_rows=s,splits={"q0":"train","q1":"train","q2":"valid","q3":"valid"})
    q=r["by_qa"]["QA-01"]
    assert q["train_fitted_constant_answer"] == "no"
    assert q["by_split"]["valid"]["train_constant_score"] == 0
    assert q["by_split"]["valid"]["questions"] == 2
    assert q["train_valid_answer_priors"]["total_variation"] == 1
    assert not q["eligible_for_supported_main_average"]


def test_quota_is_at_least_24_distinct_questions_not_zero_defect_admission():
    p,a,s=rows(["yes"]*25)
    splits={r["question_id"]:("train" if i==0 else "valid") for i,r in enumerate(p)}
    q=audit_priors(p,a,source_rows=s,splits=splits)["by_qa"]["QA-01"]
    assert q["validation_quota_status"] == "pass"
    assert q["status"] == "under_powered"  # Keep the separate prior flag visible.
    assert q["eligible_for_supported_main_average"]


def test_compact_binding_gold_uses_matching_public_label_not_first_option():
    p=[{"question_id":"b","qa_id":"QA-02","forms":{"open":{"question_en":"Who?"},"mcq":{"options":[{"label_en":"green person"},{"label_en":"blue person"}]}}}]
    a=[{"question_id":"b","qa_id":"QA-02","truth":{"answer_type":"closed_set","value":"blue","label":"blue person"}}]
    r=audit_priors(p,a)["by_qa"]["QA-02"]
    assert r["open_answers"]["most_common_answer"] == "blue person"
    assert r["correct_option_positions_by_option_count"]["2"]["counts"] == {"0":0,"1":1}
    a[0]["truth"]["label"]="absent"
    with pytest.raises(ValueError,match="exactly one"):
        audit_priors(p,a)


def test_missing_form_does_not_enter_open_distribution():
    p,a,s=rows(["yes","no"])
    del p[1]["forms"]["open"];del a[1]["forms"]["open"]
    q=audit_priors(p,a)["by_qa"]["QA-01"]
    assert q["questions"] == 2 and q["open_answers"]["n"] == 1


def test_views_deduplicate_forms_and_presentations_but_reject_cross_split_ids(tmp_path):
    for split in ("train","valid","test"):
        (tmp_path/f"{split}.jsonl").write_text("" if split!="train" else '\n'.join(json.dumps({"question_id":"x","form":f}) for f in ["open","mcq","open"]))
    assert split_map_from_views(tmp_path) == {"x":"train"}
    (tmp_path/"valid.jsonl").write_text(json.dumps({"question_id":"x"}))
    with pytest.raises(ValueError,match="multiple splits"):
        split_map_from_views(tmp_path)


def test_duplicate_primary_keys_and_unknown_split_ids_are_errors():
    p,a,s=rows(["yes"])
    with pytest.raises(ValueError,match="duplicate"):
        audit_priors(p+p,a)
    with pytest.raises(ValueError,match="unknown questions"):
        audit_priors(p,a,splits={"absent":"train"})


def test_receipt_is_private_no_clobber_and_marks_unsupported_input(tmp_path):
    (tmp_path/"public").mkdir();(tmp_path/"private").mkdir()
    p,a,s=rows(["yes"])
    for path,data in [("public/questions.jsonl",p),("private/answers.jsonl",a)]:
        (tmp_path/path).write_text('\n'.join(json.dumps(r) for r in data))
    public_before=(tmp_path/"public/questions.jsonl").read_bytes()
    report=write_prior_receipt(tmp_path)
    assert report["question_count"]==1
    assert (tmp_path/"public/questions.jsonl").read_bytes()==public_before
    with pytest.raises(FileExistsError):write_prior_receipt(tmp_path)
    a[0]["forms"]["open"]["answer_type"]="unsupported"
    (tmp_path/"private/answers.jsonl").write_text(json.dumps(a[0]))
    assert write_prior_receipt(tmp_path,output=tmp_path/"invalid.json")["status"]=="invalid_input"


def test_prompt_lookup_is_train_only_and_world_ids_are_not_actor_ids():
    p,a,s=rows(["yes","no","yes","no"],qa="QA-21")
    for i in range(4):
        prompt="blue?" if i%2==0 else "green?"
        p[i]["forms"]["open"]["question_en"]=prompt;a[i]["forms"]["open"]["question_en"]=prompt
    q=audit_priors(p,a,source_rows=s,splits={"q0":"train","q1":"train","q2":"valid","q3":"valid"})["by_qa"]["QA-21"]
    assert q["train_fitted_prompt_lookup"]["valid"]["mean_score"]==1
    assert q["train_fitted_prompt_lookup"]["valid"]["prompts_seen_in_train"]==2


def test_compact_binding_time_ranges_match_endpoints_across_unit_spellings():
    p=[{"question_id":"b","qa_id":"QA-19","forms":{"open":{"question_en":"When?"},"mcq":{"options":[{"label_en":"[8, 10) seconds"},{"label_en":"[0, 2) seconds"}]}}}]
    a=[{"question_id":"b","qa_id":"QA-19","truth":{"answer_type":"time_range_s","value":[0,2],"label":"[0, 2) s"}}]
    q=audit_priors(p,a)["by_qa"]["QA-19"]
    assert q["correct_option_positions_by_option_count"]["2"]["counts"] == {"0":0,"1":1}
    assert q["open_answers"]["most_common_answer"] == "[0, 2) seconds"

def test_bank_receipt_uses_private_world_split_projection(tmp_path):
    from avengine.qa.prior_audit import audit_bank
    (tmp_path/"public").mkdir(); (tmp_path/"private").mkdir()
    p,a,s=rows(["yes","no"])
    for path,data in [("public/questions.jsonl",p),("private/answers.jsonl",a),
                      ("private/sources.jsonl",s), ("private/splits.jsonl",[
                          {"question_id":"q0","split":"train"}, {"question_id":"q1","split":"valid"}])]:
        (tmp_path/path).write_text('\n'.join(json.dumps(r) for r in data))
    report=audit_bank(tmp_path)
    assert report["split_status"]=="complete"
    assert report["by_qa"]["QA-01"]["by_split"]["valid"]["open_questions"]==1


def test_template_lookup_sees_through_numbers_that_exact_prompts_do_not_share():
    """Balanced answers, but the stem's named appearance fixes each one.

    Every prompt also carries its own time, so no held-out prompt repeats a
    train prompt exactly and the exact lookup is blind to the leak.
    """
    p, a, s = rows(["yes", "no"] * 6, qa="QA-26")
    splits = {}
    for i in range(12):
        look = "blue" if a[i]["forms"]["open"]["truth"] == "yes" else "green"
        prompt = f"What did the person in {look} say at {i}.5 s?"
        p[i]["forms"]["open"]["question_en"] = a[i]["forms"]["open"]["question_en"] = prompt
        splits[f"q{i}"] = "train" if i < 8 else "valid"
    result = audit_priors(p, a, source_rows=s, splits=splits)
    q = result["by_qa"]["QA-26"]
    assert q["by_split"]["valid"]["train_constant_score"] == 0.5
    assert q["train_fitted_prompt_lookup"]["valid"]["prompts_seen_in_train"] == 0
    assert q["by_split"]["valid"]["train_template_score"] == 1.0
    assert "valid:high_template_answer_score" in q["flags"]
    assert result["aggregate"]["valid"]["train_template_micro_score"] == 1.0


def test_two_way_is_what_a_type_offers_not_its_name():
    """QA-04 now offers eight sectors; a type-name list called it binary."""
    p, a, s = rows(["yes", "no"] * 3, qa="QA-04")
    eight = [{"value": f"s{i}", "label_en": f"s{i}"} for i in range(8)]
    for i in range(6):
        a[i]["forms"]["mcq"] = {"options": eight, "gold": {"correct_index": i}}
        p[i]["forms"]["mcq"] = {"options": eight}
    q = audit_priors(p, a, source_rows=s)["by_qa"]["QA-04"]
    assert not q["binary_mcq_warning"] and q["intrinsic_mcq_domain"] is False
    p, a, s = rows(["yes", "no"] * 3, qa="QA-26")
    q = audit_priors(p, a, source_rows=s)["by_qa"]["QA-26"]
    assert q["binary_mcq_warning"] and q["intrinsic_mcq_domain"] is True
