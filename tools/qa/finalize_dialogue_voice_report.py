"""Build final dialogue report and record its CPU usage."""
from _dialogue_voice_runtime import *
from report_dialogue_voice_expansion import main as report_main

def main():
    tick = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    report_main()
    after = resource.getrusage(resource.RUSAGE_SELF)
    save(STAGE / 'evidence/final_report_cpu.json', {
        'cpu_user_seconds': after.ru_utime - before.ru_utime,
        'cpu_system_seconds': after.ru_stime - before.ru_stime,
        'wall_seconds': time.monotonic() - tick,
        'finished_utc': stamp(),
    })

if __name__ == '__main__':
    main()
