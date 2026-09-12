from avengine.dataset import production_runner as runner


def test_legacy_logical_marker_does_not_hide_exact_worker(monkeypatch):
    argv=["/python","-m","avengine.dataset.production_runner",
          "--execute-work-item","/run/workers/example_audio_01/task.json"]
    monkeypatch.setattr(runner,"process_identity",lambda pid:
                        {"pid":pid,"start_ticks":"123","cmdline":argv})
    assert runner.process_matches({"pid":42,"start_ticks":"123","argv":argv,
                                   "cmdline_marker":"example:audio:01"})


def test_another_task_or_recycled_pid_is_never_adopted(monkeypatch):
    monkeypatch.setattr(runner,"process_identity",lambda pid:
                        {"pid":pid,"start_ticks":"123","cmdline":["/python","other_task"]})
    assert not runner.process_matches({"pid":42,"start_ticks":"123","argv":["/python","our_task"]})
    assert not runner.process_matches({"pid":42,"start_ticks":"122","argv":["/python","other_task"]})
