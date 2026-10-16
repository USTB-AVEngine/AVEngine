"""Submit independent exact RIR requests before their ordered consumption."""
from __future__ import annotations
from concurrent.futures import Future
from contextlib import contextmanager
import os
import threading
import queue
from itertools import count
import numpy as np

ENV_KEY = 'AVENGINE_V7_RIR_PIPELINE'


class _PriorityExecutor:
    def __init__(self, workers):
        self.jobs = queue.PriorityQueue()
        self.order = count()
        self.threads = [threading.Thread(target=self._run, name='rir-submit-'+str(i)) for i in range(workers)]
        for thread in self.threads:
            thread.start()

    def submit(self, function, *args, priority=1):
        future = Future()
        self.jobs.put((priority, next(self.order), future, function, args))
        return future

    def _run(self):
        while True:
            _, _, future, function, args = self.jobs.get()
            if future is None:
                return
            if future.set_running_or_notify_cancel():
                try:future.set_result(function(*args))
                except BaseException as error:future.set_exception(error)

    def shutdown(self, wait=True, cancel_futures=False):
        if cancel_futures:
            while True:
                try:item=self.jobs.get_nowait()
                except queue.Empty:break
                if item[2] is not None:item[2].cancel()
        for thread in self.threads:
            self.jobs.put((float('inf'), next(self.order), None, None, ()))
        if wait:
            for thread in self.threads:thread.join()


class RIRSubmissionQueue:
    def __init__(self, renderer, workers):
        self.renderer = renderer
        self.executor = _PriorityExecutor(workers)
        self.futures = {}
        self.lock = threading.Lock()

    @staticmethod
    def key(position, role, trial):
        return (np.asarray(position, dtype=np.float64).tobytes(), role, trial)

    def submit(self, position, role='primary', trial=None):
        key = self.key(position, role, trial)
        with self.lock:
            if key not in self.futures:
                self.futures[key] = self.executor.submit(self.renderer._render_one, position, role, trial, priority=0 if role=='primary' else 1)
            return self.futures[key]

    def get(self, position, role, trial):
        return self.submit(position, role, trial).result()

    def primary(self, positions):
        for point in positions:
            self.submit(point)

    def references(self, positions):
        for point in positions:
            for trial in range(3):
                self.submit(point, 'independent_midpoint_reference', trial)


def submit_missing_primary(submission, cache, positions):
    """Inspect the ordinary cache on its owning thread; never move SQLite."""
    if submission is None:
        return
    import json
    for position in positions:
        point = np.asarray(position, dtype=float)
        key = json.dumps(point.tolist(), separators=(',', ':'))
        if key in cache.memory:
            continue
        row = cache.db.execute('SELECT id,completed FROM rirs WHERE pose=?', (key,)).fetchone()
        if row is not None and (cache.root / f'rir_{row[0]:06d}.npy').is_file():
            continue
        submission.primary([point])


@contextmanager
def submission_window(renderer):
    if os.environ.get(ENV_KEY) != '1' or not hasattr(renderer, '_render_one'):
        yield
        return
    existing = getattr(renderer, 'submission_queue', None)
    if existing is not None:
        yield
        return
    queue = RIRSubmissionQueue(renderer, renderer.parallel_workers)
    renderer.submission_queue = queue
    try:
        yield
    finally:
        renderer.submission_queue = None
        queue.executor.shutdown(wait=True, cancel_futures=True)
