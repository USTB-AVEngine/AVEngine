"""The final generation round obeys the post-admission gap, including zero gaps."""
import pytest
from tools.assets.generate_sao_candidates import candidate_plan
pytestmark=pytest.mark.fast_unit

def test_round_three_is_bounded_by_remaining_gap_and_preserves_old_round_budgets():
 bank=[dict(id=f"p{i}",text="device",duration_s=5) for i in range(12)]
 spec=dict(needed=8,round3_gap=2,prompts=bank,retry_prompts=bank,round3_prompts=bank)
 config=dict(classes={"printer":spec,"doorbell":dict(spec,round3_gap=0)})
 r3=candidate_plan(config,3,set())
 assert len(r3)==6 and {r["sound_class"] for r in r3}=={"printer"}
 assert len({r["sound_asset_id"] for r in r3})==6
 assert all(r["round"]==3 for r in r3)
 assert len(candidate_plan(config,1,set()))==48
 assert len(candidate_plan(config,2,set()))==16
