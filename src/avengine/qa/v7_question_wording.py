"""Natural model-facing wording; canonical answers and scoring stay untouched."""
import re

WORDING_REVISION='owner-natural-question-wording-20261015-150x'
FORBIDDEN=re.compile(r'\bSTARSS(?:23)?(?:_?v\d+\.\d+)?\b|\bv0\.2\b|\bprotocol\b|\badmitted\b|\breviewed appearances?\b|\bindependent sound events?\b|\bfrustum\b',re.I)
QUOTED=re.compile(r'''(?<!\w)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|“[^”]*”)''')


def contains_internal_wording(text):
    return bool(FORBIDDEN.search(str(text)))


def _plain(text):
    text=re.sub(r'Which of the 13 STARSS sound classes describes',
        'Which of the following sound categories best describes',text,flags=re.I)
    text=re.sub(r'Which reviewed appearance made the first sound\?',
        'Who or what made the first sound? Describe what it looks like.',text,flags=re.I)
    text=re.sub(r'Which described appearance belongs to the actor of',
        'Who or what made',text,flags=re.I)
    text=text.replace(' (the recorded utterance is ', ' (saying ')
    text=text.replace('Ignore occlusion.', 'Ignore anything blocking the source.')
    text=text.replace('What visibility state does the first speaker have at the end of the clip?', 'How well can you see the source of the first sound at the end of the clip?')
    text=re.sub(r'\bfully occluded\b','completely hidden behind something',text,flags=re.I)
    text=re.sub(r'\bpartially occluded\b','partly hidden behind something',text,flags=re.I)
    text=re.sub(r'\breviewed appearances\b','visible people, animals and objects',text,flags=re.I)
    text=re.sub(r'\breviewed appearance\b','appearance',text,flags=re.I)
    text=re.sub(r'\bindependent sound events\b','sounds',text,flags=re.I)
    text=re.sub(r'\bindependent sound event\b','sound',text,flags=re.I)
    text=re.sub(r'\bcamera frustum\b',"camera's view",text,flags=re.I)
    text=re.sub(r'\bfrustum\b',"camera's view",text,flags=re.I)
    text=re.sub(r'\b(?:according to|under|in) the protocol\b','',text,flags=re.I)
    text=re.sub(r'\bprotocol\b','clip',text,flags=re.I)
    text=re.sub(r'\b(?:admitted|v0\.2)\b','',text,flags=re.I)
    text=re.sub(r'\b(?:13\s+)?STARSS(?:23)?(?:_?v\d+\.\d+)?(?:\s+sound)?\s+classes\b','sound categories',text,flags=re.I)
    text=re.sub(r'\bSTARSS(?:23)?(?:_?v\d+\.\d+)?\b','',text,flags=re.I)
    text=re.sub(r'\s+',' ',text)
    return text


def natural_question_en(text):
    """Rewrite prose only; an exact quoted spoken cue is never paraphrased."""
    text=str(text);parts=[];start=0
    for match in QUOTED.finditer(text):
        parts.extend([_plain(text[start:match.start()]),match.group()]);start=match.end()
    parts.append(_plain(text[start:]));result=''.join(parts).strip()
    if contains_internal_wording(result):
        raise ValueError('spoken cue contains owner-forbidden question wording; construct a different complete utterance')
    return result
