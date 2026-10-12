"""Regression checks for the explicit SAO exception and train-only sampling."""
from copy import deepcopy
import pytest
from avengine.qa.v7_sao_admission import generated_source_blockers, OWNER_ACCEPTANCE
from avengine.qa.v7_recorded_audio import source_policy_blockers
from avengine.timeline.v7_audio_sources import filter_admission, source_allowed_in_split, assemble_recorded_buses, PROGRAM_SCHEMA
pytestmark=pytest.mark.fast_unit

def sample():
 return dict(sound_asset_id="sao_example",sound_class="cat_meow",source_type="generated_audio",
  source_dataset="Stable Audio Open 1.0",label_provenance="generated_stable_audio_open_1.0",
  source_license="Stability AI Community License",owner_acceptance=OWNER_ACCEPTANCE,
  allowed_splits=["train"],split="train",sha256="bound_audio",
  generation=dict(model_id="stabilityai/stable-audio-open-1.0",model_sha256="weights",model_config_sha256="config",
   prompt="cat meow",seed=42,duration_s=5,steps=100,cfg_scale=7),
  automatic_no_speech=dict(status="pass",model_id="openai/whisper-small",audio_sha256="bound_audio",
   language_forced=False,reference_text_prompted=False,human_listening_used=False,blind_decoded_text=""),
  automatic_activity_readback=dict(status="pass",audio_sha256="bound_audio",active_duration_s=2,longest_group_s=2),
  strict_v7_eligible=True,formal_status="admitted_under_listed_screens",
  screens={name:dict(status="pass",license="Stability AI Community License",original_id="sao_example",
    source_uri="https://huggingface.co/stabilityai/stable-audio-open-1.0") for name in ["license","provenance","deduplication","automatic_label"]})

def grant():
 return dict(owner_acceptance=OWNER_ACCEPTANCE,allowed_splits=["train"],classes=["cat_meow"])

def test_exception_is_explicit_and_does_not_change_real_recording_rules():
 r=sample()
 assert not generated_source_blockers(r,grant())
 assert "generated_source_owner_authorization_missing" in generated_source_blockers(r,None)
 assert "non_speech_must_be_recorded" in source_policy_blockers(dict(r,is_program_synthesized=True))

@pytest.mark.parametrize("text",["hello","你好","こんにちは","مرحبا","123","thank you"])
def test_free_multilingual_lexical_transcript_rejects(text):
 r=sample();r["automatic_no_speech"]["blind_decoded_text"]=text
 assert "generated_source_whisper_lexical_content_or_missing_receipt" in generated_source_blockers(r,grant())

@pytest.mark.parametrize("field,value",[("allowed_splits",["train","dev"]),("split","dev"),
 ("owner_acceptance","unapproved"),("source_license","CC0"),("label_provenance","real_recording_metadata")])
def test_false_rights_or_split_metadata_cannot_admit(field,value):
 r=sample();r[field]=value
 assert generated_source_blockers(r,grant())

def test_receipts_are_bound_to_actual_source_bytes_and_effective_activity():
 for screen in ["automatic_no_speech","automatic_activity_readback"]:
  r=sample();r[screen]["audio_sha256"]="different"
  assert generated_source_blockers(r,grant())
 for field in ["active_duration_s","longest_group_s"]:
  r=sample();r["automatic_activity_readback"][field]=0.99
  assert generated_source_blockers(r,grant())

@pytest.mark.parametrize("split",["dev","development","test","eval",None])
def test_generated_record_cannot_enter_heldout_or_unidentified_scene(split):
 r=sample()
 assert not source_allowed_in_split(r,split)
 if split is not None:
  admitted,rejected=filter_admission([r],split=split)
  assert not admitted and not rejected
 scene=dict(split=split,events=[],sources={})
 with pytest.raises(ValueError,match="not allowed"):
  assemble_recorded_buses(scene,dict(schema=PROGRAM_SCHEMA,records=[r],events=[]))

def test_train_retains_generated_and_unrestricted_real_inputs():
 gen=sample();real=deepcopy(gen);real.update(sound_asset_id="real",source_type="recorded")
 real.pop("allowed_splits")
 assert source_allowed_in_split(gen,"train")
 assert source_allowed_in_split(real,"dev")
 admitted,rejected=filter_admission([gen,real],split="train")
 assert len(admitted)==2 and not rejected
 admitted,rejected=filter_admission([gen,real],split="dev")
 assert [r["sound_asset_id"] for r in admitted]==["real"] and not rejected


@pytest.mark.parametrize("ns,lp",[(.6,-2),(.9,-1),(.9,float("nan")),(float("inf"),-2)])
def test_standard_no_speech_skip_requires_the_actual_upstream_probability_rule(ns,lp):
 r=sample();r["automatic_no_speech"].update(standard_no_speech_skipped=True,
  no_speech_prob=ns,avg_logprob=lp,no_speech_threshold=.6,logprob_threshold=-1.)
 assert "generated_source_standard_no_speech_probability_receipt_invalid" in generated_source_blockers(r,grant())

def test_standard_no_speech_skip_with_bound_valid_probabilities_is_admitted():
 r=sample();r["automatic_no_speech"].update(standard_no_speech_skipped=True,
  no_speech_prob=.9,avg_logprob=-2.,no_speech_threshold=.6,logprob_threshold=-1.)
 assert not generated_source_blockers(r,grant())

def test_nonfinite_active_group_cannot_satisfy_one_second_gate():
 r=sample();r["automatic_activity_readback"]["longest_group_s"]=float("nan")
 assert "generated_source_effective_activity_below_one_second_or_missing" in generated_source_blockers(r,grant())
