"""Own-process census sampling timeboxes; no production acceptance predicate."""
import signal


class CensusSamplingTimeout(TimeoutError):
    pass


def arm_sampling_timebox(seconds):
    seconds=float(seconds)
    if not 0<seconds<=2700:raise ValueError('invalid census wall-time budget')
    def expired(signum,frame):
        raise CensusSamplingTimeout('census_asset_sampling_timebox_'+str(int(seconds))+'s')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    return previous


def disarm_sampling_timebox(previous):
    signal.setitimer(signal.ITIMER_REAL,0)
    signal.signal(signal.SIGALRM,previous)
