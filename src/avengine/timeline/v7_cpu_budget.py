"""Budget compute workers while preserving every process in memory accounting."""
from pathlib import Path


def compute_process_count(records):
    total=0
    for record in records:
        command=record.get('cmd','');words=command.split();state=record.get('state','')
        if state=='Z':continue
        if not words:continue
        program=Path(words[0]).name
        media=program in {'ffmpeg','ffprobe'}or 'Unreal'in program
        native=('avengine.timeline.v7_phase2_fresh_rir'in command and '--request'in words)
        worker=('run_v7_acceptance.py'in {Path(w).name for w in words} and '--worker'in words)
        automatic=('generate_v7_demand_scenes.py'in {Path(w).name for w in words} and '--worker'in words)
        if media or native or worker or automatic:
            total+=1
        elif state=='R' and program not in {'ssh','sshd','bash','zsh','sh','bwrap','timeout','sleep'}:
            total+=1
    return total
