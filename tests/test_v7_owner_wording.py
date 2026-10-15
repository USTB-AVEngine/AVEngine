import copy
import pytest
from avengine.qa.v7_question_wording import natural_question_en,contains_internal_wording
from avengine.timeline.v7_english_export import english_public_question
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize('before,phrase',[
 ('Which of the 13 STARSS sound classes describes the sound starting at about 1.1 seconds?','following sound categories'),
 ('Which reviewed appearance made the first sound?','Who or what made the first sound?'),
 ('Did independent sound events 1 and 2 come from the same person?','Did sounds 1 and 2'),
 ('Is the sound inside or outside the camera frustum?','camera\'s view'),
 ('Which admitted sound category applies under the protocol v0.2?','sound category')])
def test_all_internal_question_terms_are_naturalized(before,phrase):
 result=natural_question_en(before)
 assert phrase in result and not contains_internal_wording(result)


def test_wording_preserves_exact_spoken_cues_and_all_private_answers_domains_and_choices():
 original={'qa_id':'QA-02','question_en':'Which described appearance belongs to the actor of the first independent sound event (the recorded utterance is "The jar has an orange cap.")?',
  'canonical_answer':'blue','answer_type':'closed_set','model_answer':'the man in the blue shirt',
  'choices':[{'id':'A','text':'the man in the blue shirt'},{'id':'B','text':'the woman in the green shirt'}],
  'truth':{'value':'blue','evidence':{'event_id':'same_native_id'}},'classes':{'blue':['blue']}}
 before=copy.deepcopy(original);after=english_public_question(original)
 assert '"The jar has an orange cap."'in after['question_en']
 for key in ('canonical_answer','answer_type','model_answer','truth','classes','choices'):assert after[key]==before[key]
 assert original==before


def test_forbidden_exact_quote_is_never_falsified_by_paraphrasing():
 with pytest.raises(ValueError,match='different complete utterance'):
  natural_question_en('Who said "Follow the protocol"?')
