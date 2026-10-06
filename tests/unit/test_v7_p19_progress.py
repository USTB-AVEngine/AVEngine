"""Prevent sibling rank exit and stale pre-reboot progress from masquerading as status."""
import json
from avengine.timeline.v7_p19_progress import summarize, inspect_plan, journal_tail


def facts():
    p = {'status': 'running', 'pid': 105, 'hostname': 'p19', 'active_parents': list(range(40)),
         'formal_parents': 91, 'formal_windows': 242, 'formal_questions': 3388, 'workers_limit': 40}
    o = {'pid': 105, 'hostname': 'p19', 'job_id': 'job_new'}
    ranks = {'rank0': [{'event': 'worker_done', 'reason': 'all_runnable_claimed', 'rank': 0}],
             'rank5': [{'event': 'experiment_start', 'rank': 5}]}
    return p, o, ranks


def test_other_rank_claim_is_running():
    p, o, ranks = facts()
    r = summarize(p, o, 'job_new', ranks, 3)
    assert r['verdict'] == 'generation_running'
    assert r['claiming_ranks'] == [5]
    assert r['sibling_all_runnable_claimed_ranks'] == [0]
    assert r['active_parents'] == 40


def test_stale_progress_does_not_prove_live_dispatch():
    p, o, ranks = facts()
    assert summarize(p, o, 'job_new', ranks, 300)['verdict'] == 'progress_stale_not_proven_running'


def test_new_job_does_not_inherit_old_coordinator_status():
    p, o, ranks = facts()
    o['job_id'] = 'job_old'
    r = summarize(p, o, 'job_new', ranks, 3)
    assert r['verdict'] == 'claimed_awaiting_current_coordinator'
    assert not r['owner_matches_current_job_and_progress']


def test_no_current_lease_does_not_prove_running():
    p, o, ranks = facts()
    assert summarize(p, o, None, {}, 3)['verdict'] == 'no_current_experiment_start'


def test_zero_active_is_not_empty_pending():
    p, o, ranks = facts()
    p['active_parents'] = []
    p['remaining_parents'] = 279
    r = summarize(p, o, 'job_new', ranks, 3)
    assert r['verdict'] == 'coordinator_running_no_active_parents'
    assert r['pending_parents'] == 279


def test_read_only_inspection_uses_exact_profile_path(tmp_path):
    plan_path = tmp_path / 'plan.json'
    root = tmp_path / 'task'
    root.mkdir()
    profile_path = tmp_path / 'capture_by_path' / 'absolute' / 'data' / 'jzy' / 'profile.json'
    profile_path.parent.mkdir(parents=True)
    run = root / 'p19_run'
    active = tmp_path / 'active'
    active.mkdir()
    p, o, ranks = facts()
    (root / 'PROGRESS.json').write_text(json.dumps(p))
    (root / 'coordinator.lock').mkdir()
    (root / 'coordinator.lock/owner.json').write_text(json.dumps(o))
    (active / 'fixture').write_text('job_new p19 timestamp')
    profile_path.write_text(json.dumps({'root': str(root), 'owner_root': str(tmp_path)}))
    plan_path.write_text(json.dumps({'name': 'fixture', 'run_root': str(run), 'experiments': [
        {'name': 'exp', 'evals': [{'name': 'job', 'checkpoint_flag': '--generator-profile', 'checkpoint': str(profile_path)}]}]}))
    for name, rows in ranks.items():
        f = run / 'workers/job_new' / (name + '.jsonl')
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    before = {str(f): f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    result = inspect_plan(plan_path, active_root=active)
    after = {str(f): f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    assert before == after
    assert result['verdict'] == 'generation_running'
    assert result['mutations'] == result['house_access'] == 0
    assert result['profile'] == str(profile_path)


def test_concurrent_partial_journal_line_is_ignored(tmp_path):
    f = tmp_path / 'worker.jsonl'
    f.write_text('{"event":"experiment_start","rank":5}\n{"event":')
    assert journal_tail(f) == [{'event': 'experiment_start', 'rank': 5}]
