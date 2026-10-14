"""Animation-first animal movement with an explicit, qualified sliding fallback.

The caller supplies backend-native compilers and the unchanged full admission
function. Missing animation is a normal capability result; IO, resource and
programming failures remain errors rather than being hidden by the fallback.
"""
from copy import deepcopy
from dataclasses import dataclass

SLIDE_MAX_SPEED_M_S=.08

class MotionUnavailable(ValueError):
    """An asset has no usable Walking source."""

class MotionRejected(ValueError):
    """A concrete attempted route/pose failed native qualification."""

@dataclass(frozen=True)
class MotionChoice:
    motion_mode: str
    reason: str
    animation_attempts: tuple
    payload: object = None

    def metadata(self):
        return {'motion_mode':self.motion_mode,'motion_reason':self.reason,
                'animation_attempted':True,'motion_attempts':deepcopy(list(self.animation_attempts)),
                'gait_animation_present':self.motion_mode=='walk_animation',
                'slide_speed_limit_m_s':SLIDE_MAX_SPEED_M_S if self.motion_mode=='slide_no_animation' else None}


def select_motion(*,asset_id,requested_speed_m_s,attempt_animation,attempt_slide,qualify):
    """Try Walking first, then a native-qualified registered standing slide.

    Both callbacks receive the actual requested speed. ``qualify`` must check
    the selected policy's native geometry, heading, emitter and, when required,
    moving RIR evidence. It returns a
    status plus explicit reasons. No earlier static receipt admits a route.
    """
    speed=float(requested_speed_m_s)
    if not 0<speed<=.3:raise ValueError('invalid_animal_motion_speed')
    attempts=[]
    for mode,builder,actual in [('walk_animation',attempt_animation,speed),('slide_no_animation',attempt_slide,min(speed,SLIDE_MAX_SPEED_M_S))]:
        try:
            payload=builder(actual)
            verdict=qualify(payload,mode)
            if verdict.get('status')!='pass':
                reason=';'.join(verdict.get('failures') or [verdict.get('reason','native_admission_not_pass')])
                raise MotionRejected(reason)
            attempts.append({'asset_id':asset_id,'mode':mode,'status':'pass','speed_m_s':actual})
            return MotionChoice(mode,'qualified_animation' if mode=='walk_animation' else attempts[0]['reason'],tuple(attempts),payload)
        except (MotionUnavailable,MotionRejected) as error:
            attempts.append({'asset_id':asset_id,'mode':mode,'status':'unavailable' if isinstance(error,MotionUnavailable) else 'rejected','reason':str(error),'speed_m_s':actual})
    return MotionChoice('cannot_move',';'.join(a['mode']+':'+a['reason'] for a in attempts),tuple(attempts))


def attach_to_scene(scene,endpoint,choice):
    """Add movement provenance without changing event clocks or truth values."""
    scene['sources'][endpoint]['motion_metadata']=choice.metadata()


def annotate_question_motion(questions,scene):
    """Carry provenance to every question, including questions with no target ID."""
    sources={ep:deepcopy(spec['motion_metadata']) for ep,spec in scene.get('sources',{}).items() if 'motion_metadata' in spec}
    if not sources:return questions
    out=deepcopy(questions)
    for question in out:
        question.setdefault('metadata',{})['source_motion']=deepcopy(sources)
    return out
