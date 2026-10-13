"""The SAO caller keeps the thin top-up score boundary and real-source defaults."""
from copy import deepcopy
import numpy as np
import soundfile as sf
import pytest
import avengine.qa.v7_extended_admission as module
from avengine.qa.v7_sao_admission import OWNER_ACCEPTANCE
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize("confidence,rival,expected",[
 (.1917129977,0,True),(.1917129976,0,False),(.5,.44999,True),(.5,.45001,False)])
def test_generated_builder_keeps_threshold_and_incompatible_margin(tmp_path,monkeypatch,confidence,rival,expected):
 path=tmp_path/"input.wav";sf.write(path,np.zeros(32000),16000,subtype="FLOAT")
 digest=module._digest(path)
 row=dict(sound_asset_id="candidate",sound_class="cat_meow",original_id="one",
  path=str(path),sha256=digest,candidate_classes=["cat_meow"],target_mids=["/m/07qrkrw"],
  source_type="generated_audio",source_dataset="Stable Audio Open 1.0",
  label_provenance="generated_stable_audio_open_1.0",source_license="Stability AI Community License",
  owner_acceptance=OWNER_ACCEPTANCE,allowed_splits=["train"],split="train",source_uri="model",
  generation=dict(model_id="stabilityai/stable-audio-open-1.0",model_sha256="weights",
   model_config_sha256="config",prompt="a cat meow",seed=5,duration_s=4,steps=100,cfg_scale=7),
  whisper_transcription_recording_only=dict(status="recorded",role="recording_only",participates_in_admission=False,
   blind_decoded_text="Thank you for watching!",audio_sha256=digest),
  human_speech_screen=dict(status="pass",model_id="BEATs_AS2M_cpt2",audio_sha256=digest,maximum_score=.1,
   threshold=.4,heads=[{"mid":"/m/09x0r","name":"Speech"}],calibration_sha256="calibration",
   calibration=dict(status="pass",positive_count=183,positive_pass_rate=1.,negative_count=100,negative_block_rate=.99)),
  automatic_activity_readback=dict(status="pass",audio_sha256=digest,active_duration_s=2,longest_group_s=2),
  source_metadata=dict(title="a cat meow"),official_zip_crc_verified=False,official_collection_labels=[])
 score=dict(sound_asset_id="candidate",sha256=digest,scores_by_mid={"/m/07qrkrw":confidence,"/m/rival":rival},top8=[])
 proof=dict(complete_for_listed_roots=True,records={"candidate":{"status":"pass"}},
  reference_file_count=1,reference_waveform_count=1,seconds=0,scope_limit="unit fixture")
 # Normalization is independent of the model boundary tested here.
 monkeypatch.setattr(module,"normalize_recorded",lambda row,output:dict(status="pass"))
 kwargs=dict(pool=[],old_inventory=[],old_admission={"records":[]},vctk_lineages=[],
  fsd_rows=[row],beats_rows=[score],clap_rows=[],ontology=[{"id":"/m/07qrkrw","child_ids":[]},{"id":"/m/rival","child_ids":[]}],
  collection_mids={},class_targets={"cat_meow":{"target_mids":["/m/07qrkrw"]}},deduplications=[proof],
  existing_gate_classes=("cat_meow",),existing_subtype_metadata_groups={},
  generated_source_authorization=dict(owner_acceptance=OWNER_ACCEPTANCE,allowed_splits=["train"],classes=["cat_meow"]))
 result=module.build_recorded_admission(**kwargs,output=tmp_path/"generated")
 record=result["records"][0]
 assert bool(record.get("strict_v7_eligible"))==expected
 assert record["original_group"]=="sao:one"
 gate=record["automatic_label"][0]["gate"]
 assert gate["threshold"]==.1917129977 and gate["minimum_margin"]==.05
 if expected:
  assert record["provenance_status"]=="qualified_generated_input_under_listed_screens"
 real=deepcopy(row);real["source_type"]="recorded";real["is_program_synthesized"]=True
 denied=module.build_recorded_admission(**dict(kwargs,fsd_rows=[real]),output=tmp_path/"real")
 assert not denied["records"][0].get("strict_v7_eligible")
 assert "per_clip_license_or_official_waveform_association_missing" in denied["records"][0]["current_policy_blockers"]
