"""Production speed ranges checked against independently read-back body roots."""
import numpy as np

SPEED_RANGES={'human':(.8,1.4),'dog':(.5,1.2),'cat':(.3,.8),'slide':(.2,.5)}


def certify_moving_root_speed(positions, times, moving, kind):
    positions=np.asarray(positions,float);times=np.asarray(times,float);moving=np.asarray(moving,bool)
    if positions.shape!=(len(times),3) or moving.shape!=times.shape or len(times)<2:
        raise ValueError('measured root speed requires a complete matching native clock')
    dt=np.diff(times)
    if not np.isfinite(positions).all() or np.any(dt<=0):
        raise ValueError('measured root speed requires finite positions and increasing native times')
    speed=np.linalg.norm(np.diff(positions[:,[0,2]],axis=0),axis=1)/dt
    selected=(moving[:-1]|moving[1:]) & (speed>1e-6)
    if not selected.any():
        raise ValueError('requested movement has no measured translated native roots')
    median=float(np.median(speed[selected]));lo,hi=SPEED_RANGES[kind]
    if not lo-1e-6<=median<=hi+1e-6:
        raise ValueError('measured moving-root median speed %.9f outside %s range [%.3f, %.3f] m/s'%(median,kind,lo,hi))
    return {'status':'pass','kind':kind,'median_m_s':median,'allowed_range_m_s':[lo,hi],
            'moving_segment_count':int(selected.sum()),'minimum_m_s':float(speed[selected].min()),
            'maximum_m_s':float(speed[selected].max()),'measurement':'native body-to-world root readback; horizontal consecutive-frame distances / measured dt'}
