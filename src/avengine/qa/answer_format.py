"""Public answer text only; numeric truths, choice order and scoring stay unchanged."""
from __future__ import annotations
from copy import deepcopy
import re

ANGLE_INSTRUCTION = "Answer with one integer in degrees, without the degree sign: front is 0, left is positive, right is negative, range -180 to 179."
LETTER_INSTRUCTION = "Answer with the option letter only."
NUMBER_INSTRUCTION = "Answer with the option number only."
COUNT_INSTRUCTION = "Answer with one non-negative integer only."
PAIR_INSTRUCTION = "Answer with two integers separated by one space: entity count first, then sounding entity count."
DIRECTIONS = ('front', 'front-left', 'left', 'back-left', 'back', 'back-right', 'right', 'front-right')
DIRECTION_INSTRUCTION = "Answer with one direction only: " + ", ".join(DIRECTIONS) + "."


def option_id(index, option_count):
    """A-Z for <=26 options; the existing scorer's 1-N IDs above 26."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < option_count:
        raise ValueError('invalid option index')
    return chr(65 + index) if option_count <= 26 else str(index + 1)


def option_instruction(option_count):
    if option_count < 1:
        raise ValueError('empty option domain')
    return LETTER_INSTRUCTION if option_count <= 26 else NUMBER_INSTRUCTION


def integer_text(value, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError('boolean is not an integer answer')
    text = str(value)
    if not re.fullmatch(r'[+-]?\d+(?:\.0+)?', text):
        raise ValueError('answer is not an exact integer')
    integer = int(text.split('.')[0])
    if minimum is not None and integer < minimum or maximum is not None and integer > maximum:
        raise ValueError('integer answer outside its original domain')
    return str(integer)


def direction_text(value):
    text = str(value).strip().casefold().replace('_', '-').replace(' ', '-')
    text = {'behind': 'back', 'rear': 'back', 'rear-left': 'back-left',
            'rear-right': 'back-right', 'left-front': 'front-left',
            'right-front': 'front-right'}.get(text, text)
    if text not in DIRECTIONS:
        raise ValueError('answer is not a standard eight-way direction')
    return text


def answer_text(value, *, answer_type=None, qa_id=None, legacy_text=None):
    """Serialize a training target; never round or replace the original truth."""
    if answer_type == 'angle_deg' or qa_id in {'QA-13', 'QA-25'}:
        return integer_text(value, minimum=-180, maximum=179)
    if answer_type == 'count_pair' or qa_id == 'QA-22' and isinstance(value, (list, tuple)) and len(value) == 2:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('count_pair requires entity count then sounding entity count')
        return ' '.join(integer_text(x, minimum=0) for x in value)
    if answer_type == 'count_single' or qa_id in {'QA-23', 'STARSS-SPEAKERS'} or qa_id == 'QA-22':
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError('count_single requires one count')
            value = value[0]
        return integer_text(value, minimum=0)
    if qa_id in {'QA-04', 'QA-28'}:
        return direction_text(legacy_text if legacy_text is not None else value)
    return str(value) if legacy_text is None else legacy_text


def _without_output_instruction(text):
    text = str(text or '')
    for instruction in (ANGLE_INSTRUCTION, LETTER_INSTRUCTION, NUMBER_INSTRUCTION,
                        COUNT_INSTRUCTION, PAIR_INSTRUCTION, DIRECTION_INSTRUCTION):
        text = text.replace(instruction, '')
    # Known producer output clauses only. Query facts/times and option text remain.
    for clause in ('Answer in whole degrees.', '请回答整数角度。',
                   'Return the displayed option ID.', '返回显示的选项ID。'):
        text = text.replace(clause, '')
    text = re.sub(r'Return one decimal integer from -180 (?:to|through) 179[^\n]*?\.', '', text)
    text = re.sub(r'return one decimal integer from -180 through 179[^\n]*?\.', '', text)
    text = re.sub(r'Return one non-negative decimal integer[^\n]*?\.', '', text)
    text = re.sub(r'Return exactly one of:[^\n]*?Return only that string, without explanation\.', '', text)
    return text.rstrip()


def format_prompt(text, *, form='open', answer_type=None, qa_id=None, option_count=0, truth=None):
    text = _without_output_instruction(text)
    if form == 'mcq':
        instruction = option_instruction(option_count)
    elif answer_type == 'angle_deg' or qa_id in {'QA-13', 'QA-25'}:
        instruction = ANGLE_INSTRUCTION
    elif answer_type == 'count_pair' or qa_id == 'QA-22' and isinstance(truth, (list, tuple)) and len(truth) == 2:
        instruction = PAIR_INSTRUCTION
    elif answer_type == 'count_single' or qa_id in {'QA-23', 'STARSS-SPEAKERS'} or qa_id == 'QA-22':
        instruction = COUNT_INSTRUCTION
    elif qa_id in {'QA-04', 'QA-28'}:
        instruction = DIRECTION_INSTRUCTION
    else:
        return text
    return text + ('\n' if text else '') + instruction


def normalize_export(public, private):
    """Format a v7 export copy without modifying IDs, canonical values or evidence."""
    public, private = deepcopy(public), deepcopy(private)
    gold = {r['question_id']: r for r in private}
    for row in public:
        answer = gold[row['question_id']]
        form = row.get('form', 'open')
        qa = row.get('qa_id')
        kind = row.get('answer_type')
        value = answer.get('canonical_answer', answer.get('answer'))
        if form == 'mcq':
            choices = row['choices']
            ids = [str(c['id']) for c in choices]
            index = ids.index(str(answer['answer']))
            for i, choice in enumerate(choices):
                choice['id'] = option_id(i, len(choices))
            answer['answer'] = option_id(index, len(choices))
            count = len(choices)
        else:
            # Keep raw JSON authority answers in their original types.
            # Training text is a separate export field.
            teacher = answer_text(value, answer_type=kind, qa_id=qa)
            count = 0
        for field in ('question_en', 'question_zh'):
            if field in row:
                row[field] = format_prompt(row[field], form=form, answer_type=kind,
                    qa_id=qa, option_count=count, truth=value)
        answer['teacher_answer'] = answer['answer'] if form == 'mcq' else teacher
    return public, private
