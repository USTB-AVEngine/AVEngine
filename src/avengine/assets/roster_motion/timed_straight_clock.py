"""Smooth slower turns outside the required straight speaking/silent windows."""
import math
import numpy as np

def timed_straight_clock(times,speed,*,straight_start_s=.7,straight_end_s=7.6,turn_speed_m_s=.4):
    times=np.asarray(times,float);straight=float(straight_end_s-straight_start_s)
    turn=10.-straight;ramp=min(.3,turn/3);slow=min(float(speed),float(turn_speed_m_s));extra=float(speed)-slow
    slow_begin=3.5-straight_start_s
    def leg(x):
        x=np.clip(x,0,straight);u=np.clip((x-slow_begin)/ramp,0,1)
        return speed*x-extra*ramp*(u**3-.5*u**4)-extra*np.maximum(x-slow_begin-ramp,0.)
    def arc(x):
        x=np.clip(x,0,turn)
        a=np.clip(x/ramp,0,1);b=np.clip((x-(turn-ramp))/ramp,0,1)
        return slow*x+extra*ramp*(b**3-.5*b**4)
    length=float(leg(straight));arc_length=float(arc(turn));half=length+arc_length
    def absolute(t):
        phase=(np.asarray(t)-straight_start_s)%10.;cycle=np.floor((np.asarray(t)-straight_start_s)/10.)
        return cycle*half+np.where(phase<=straight,leg(phase),length+arc(phase-straight))
    absolute_distance=absolute(times);origin=float(absolute(0.));distance=absolute_distance-origin
    phase=(times-straight_start_s)%10.;x=phase-straight
    a=np.clip(x/ramp,0,1);b=np.clip((x-(turn-ramp))/ramp,0,1)
    leg_u=np.clip((phase-slow_begin)/ramp,0,1)
    velocity=np.where(phase<=straight,speed-extra*leg_u**2*(3-2*leg_u),slow+extra*b*b*(3-2*b))
    return distance,velocity,{'straight_length_m':length,'turn_radius_m':arc_length/math.pi,
        'phase_distance_offset_m':origin,'straight_start_s':straight_start_s,'straight_end_s':straight_end_s,
        'turn_speed_m_s':slow,'silent_straight_slowdown_start_s':3.5,'clock_kind':'smooth slower turns outside straight required windows',
        'distance_integral_method':'analytic smoothstep integral; source phase follows the same exact distance'}

def single_pass_clock(times,speed,quiet_speed=.4,stop_s=7.9):
    """Speak at scene speed, continue silently, then finish a complete step."""
    times=np.asarray(times,float);a=np.clip(times/.25,0,1);b=np.clip((times-3.5)/.3,0,1)
    c=np.clip((stop_s-times)/.3,0,1)
    velocity=(speed-(speed-quiet_speed)*b*b*(3-2*b))*a*a*(3-2*a)*c*c*(3-2*c)
    velocity[times>=stop_s]=0.
    distance=np.r_[0.,np.cumsum((velocity[:-1]+velocity[1:])*np.diff(times)/2)]
    return distance,velocity
