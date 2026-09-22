import json
from pathlib import Path
import pytest
from avengine.qa.prior_bank import curate_bank
from avengine.qa.prior_audit import read_jsonl


def make_bank(root):
    for folder in ("public", "private", "media"):
        (root/folder).mkdir(parents=True)
    facts = root/"facts.json"
    facts.write_text(json.dumps({"events": [{"sound_class": "cough"}]}))
    public, answers, sources, splits = [], [], [], []
    for i in range(2):
        qid = f"q{i}"
        options = [{"value": c, "label_en": c, "label_zh": c} for c in ["cough", "speech", "dog", "cat"]]
        public.append({"question_id": qid, "qa_id": "QA-21", "forms": {"open": {"question_en": "Sound?"}, "mcq": {"options": options}}, "media": {}})
        answers.append({"question_id": qid, "qa_id": "QA-21", "evidence": {}, "truth": {"value": "cough"},
                        "forms": {"open": {"answer_type": "closed_set", "truth": "cough", "classes": {"cough": ["cough"]}},
                                  "mcq": {"options": options, "gold": {"correct_index": 0}}}})
        sources.append({"question_id": qid, "world_id": "one_world", "episode_id": "one", "facts_path": str(facts)})
        splits.append({"question_id": qid, "split": "valid"})
    for path, rows in [("public/questions.jsonl", public), ("private/answers.jsonl", answers),
                       ("private/sources.jsonl", sources), ("private/splits.jsonl", splits)]:
        (root/path).write_text('\n'.join(json.dumps(r) for r in rows))
    return root


def test_curation_keeps_truth_and_world_splits_without_overwriting_input(tmp_path):
    source = make_bank(tmp_path/"old")
    before = (source/"private/answers.jsonl").read_bytes()
    out = tmp_path/"new"
    result = curate_bank(source, out)
    assert result["exported_question_count"] == 2 and result["world_count"] == 1
    assert result["forms"] == {"open": 2}
    assert result["split_counts"] == {"valid": 2}
    assert (source/"private/answers.jsonl").read_bytes() == before
    assert all(a["truth"]["value"] == "cough" for a in read_jsonl(out/"private/answers.jsonl"))
    assert json.loads((out/"private/answer_priors.json").read_text())["split_status"] == "complete"
    with pytest.raises(FileExistsError):
        curate_bank(source, out)


def test_curation_rejects_conflicting_world_splits(tmp_path):
    source = make_bank(tmp_path/"old")
    (source/"private/splits.jsonl").write_text('\n'.join(json.dumps({"question_id":f"q{i}", "split":s}) for i,s in enumerate(["train","test"])))
    with pytest.raises(ValueError, match="multiple splits"):
        curate_bank(source, tmp_path/"new")
