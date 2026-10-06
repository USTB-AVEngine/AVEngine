"""Read-only P19 plan inspection across ranks; never claim, dispatch or stop work."""
from __future__ import annotations

import json
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def journal_tail(path, limit=512_000):
    """Skip an incomplete concurrent final JSONL write and bound log I/O."""
    with Path(path).open('rb') as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - limit))
        lines = stream.read().decode(errors='replace').splitlines()
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def summarize(progress, owner, job_id, ranks, age_seconds, stale_after=120):
    """A sibling worker's exit cannot override the actual coordinator's progress."""
    starts = [r for data in ranks.values() for r in data if r.get('event') == 'experiment_start']
    ends = [r for data in ranks.values() for r in data if r.get('event') == 'experiment_end']
    active = progress.get('active_parents', [])
    active_count = len(active) if isinstance(active, list) else int(active or 0)
    matching_owner = bool(job_id and owner and owner.get('job_id') == job_id
                          and owner.get('pid') == progress.get('pid')
                          and owner.get('hostname') == progress.get('hostname'))
    fresh = age_seconds is not None and 0 <= age_seconds <= stale_after
    if job_id and starts and matching_owner and fresh and progress.get('status') == 'running':
        verdict = 'generation_running' if active_count else 'coordinator_running_no_active_parents'
    elif job_id and starts and matching_owner and not fresh:
        verdict = 'progress_stale_not_proven_running'
    elif ends:
        verdict = 'experiment_ended_check_stage_state'
    elif job_id and starts:
        verdict = 'claimed_awaiting_current_coordinator'
    else:
        verdict = 'no_current_experiment_start'
    exited = [r for data in ranks.values() for r in data
              if r.get('event') == 'worker_done' and r.get('reason') == 'all_runnable_claimed']
    return {'verdict': verdict, 'claiming_ranks': sorted({r['rank'] for r in starts}),
            'sibling_all_runnable_claimed_ranks': sorted({r['rank'] for r in exited}),
            'active_parents': active_count, 'workers_limit': progress.get('workers_limit'),
            'formal_parents': progress.get('formal_parents'), 'formal_windows': progress.get('formal_windows'),
            'formal_questions': progress.get('formal_questions'),
            'pending_parents': progress.get('remaining_parents'),
            'failed_parent_receipts': len(progress.get('failures', [])),
            'progress_status': progress.get('status'), 'progress_age_seconds': age_seconds,
            'owner_matches_current_job_and_progress': matching_owner,
            'current_job_id': job_id, 'coordinator_pid': progress.get('pid'),
            'coordinator_hostname': progress.get('hostname'), 'GPU_started': progress.get('GPU_started'),
            'experiment_start_events': starts, 'experiment_end_events': ends}


def inspect_plan(plan_path, *, active_root=None, now=None):
    """Read this plan only. No media/house access, directory creation or queue edits."""
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    if not plan:
        raise ValueError('missing plan')
    profile_paths = {e['checkpoint'] for exp in plan['experiments'] for e in exp.get('evals', [])
                     if e.get('checkpoint_flag') == '--generator-profile'}
    if len(profile_paths) != 1:
        raise ValueError('expected one generator profile for this plan')
    profile_path = Path(next(iter(profile_paths)))
    profile = read_json(profile_path)
    task = Path(profile['root'])
    active_root = Path(active_root) if active_root else Path(profile['owner_root']) / 'hardbank/tmp/p19_active'
    active = active_root / plan['name']
    try:
        stamp = active.read_text().split()
    except FileNotFoundError:
        stamp = []
    job_id = stamp[0] if stamp else None
    run_root = Path(plan['run_root'])
    ranks = {}
    if job_id:
        for f in (run_root / 'workers' / job_id).glob('rank*.jsonl'):
            ranks[f.stem] = journal_tail(f)
    progress_path = task / 'PROGRESS.json'
    try:
        modified = progress_path.stat().st_mtime
        progress = read_json(progress_path) or {}
    except FileNotFoundError:
        modified, progress = None, {}
    owner = read_json(task / 'coordinator.lock/owner.json') or {}
    inspected = time.time() if now is None else now
    result = summarize(progress, owner, job_id, ranks,
                       None if modified is None else inspected - modified)
    claims = {}
    stage_receipts = {}
    for exp in plan['experiments']:
        f = run_root / 'claims' / (job_id or 'no-current-job') / exp['name']
        claims[exp['name']] = read_json(f)
        stage_receipts[exp['name']] = []
        exp_root = run_root / exp['name']
        for ev in exp.get('evals', []):
            for attempt in exp_root.glob('eval_' + ev['name'] + '_a*'):
                if attempt.is_dir():
                    stage_receipts[exp['name']].append({'attempt': str(attempt), 'run': read_json(attempt / 'run.json')})
    return {**result, 'checked_unix': inspected, 'plan': plan['name'], 'profile': str(profile_path),
            'claims': claims, 'stage_receipts': stage_receipts,
            'quota': read_json(task / 'owner_quota_monitor.json'),
            'sources': {'plan': str(plan_path), 'lease': str(active),
                        'progress': str(progress_path), 'owner': str(task / 'coordinator.lock/owner.json'),
                        'worker_logs': [str(run_root / 'workers' / (job_id or 'no-current-job') / (n + '.jsonl'))
                                        for n in ranks]},
            'interpretation': 'P19 ranks claim experiments; the coordinator dispatches CPU parents. A sibling worker_done is not plan failure.',
            'mutations': 0, 'house_access': 0}
