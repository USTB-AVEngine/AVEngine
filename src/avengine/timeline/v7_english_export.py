"""Fail-closed English-only model text; metadata IDs and private truths stay private."""
from copy import deepcopy
import re
from avengine.qa.v7_english_surface import tidy_english
from avengine.qa.v7_question_wording import natural_question_en

ENGLISH_EXPORT_REVISION='v7-English-model-text-gate-20261015-v1'
CJK=re.compile('[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U0002ebef]')
REGISTRATION_NAME=re.compile(r'\bv7_[A-Za-z0-9_]+|\b(?:actor|voice|source|entity|person|man|woman|speaker)[_#\- ]+\d+\b|\badult\s+(?:male|female)\s+\d+\b|\b(?:male|female)_adult_\d+\b|\b(?:floorstanding_speaker|bookshelf_speaker|kitchen_timer|carbon_monoxide_alarm|air_conditioner|desk_telephone|doorbell_chime_unit|security_keypad|wall_clock)\b',re.I)
MODEL_KEYS={'question_en','question','prompt','instruction','instructions','choices','options','model_answer','answer','caption','description','context'}


def validate_english_model_text(row):
    if not isinstance(row.get('question_en'),str) or not row['question_en'].strip():
        raise ValueError('English export requires a nonempty question_en')
    def inspect(value,path):
        if isinstance(value,str):
            if CJK.search(value):raise ValueError('CJK in model-visible text: '+path)
            if REGISTRATION_NAME.search(value):raise ValueError('registration ID or numbered entity in model-visible text: '+path)
        elif isinstance(value,list):
            for i,child in enumerate(value):inspect(child,path+f'[{i}]')
        elif isinstance(value,dict):
            for key,child in value.items():
                if key not in {'id','question_id','logical_question_id'}:inspect(child,path+'.'+key)
    for key in MODEL_KEYS:
        if key in row:inspect(row[key],key)
    return {'status':'pass','revision':ENGLISH_EXPORT_REVISION,'model_text_CJK_count':0,'registration_name_count':0}


def english_public_question(question):
    row=deepcopy(question)
    row['question_en']=natural_question_en(tidy_english(row.get('question_en') or ''))
    # Internal bilingual authority fields are not supplied to a model.
    row.pop('question_zh',None)
    for choice in row.get('choices',[]):
        if 'text' in choice:choice['text']=natural_question_en(tidy_english(choice['text']))
        choice.pop('text_zh',None)
    if row.get('qa_id')=='V7-CONTENT-BIND':
        from avengine.qa.v7_english_surface import content_answer_en
        for choice in row.get('choices',[]):
            if 'text'in choice:choice['text']=content_answer_en(choice['text'])
    validate_english_model_text(row)
    return row
