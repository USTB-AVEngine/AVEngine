"""Submit independent exact RIR requests before their ordered consumption."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import threading
import numpy as np

ENV_KEY = 'AVENGINE_V7_RIR_PIPELINE'


class RIRSubmissionQueue:
    def __init__(self, renderer, workers):
        self.renderer = renderer
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.futures = {}
        self.lock = threading.Lock()

    @staticmethod
    def key(position, role, trial):
        return (np.asarray(position, dtype=np.float64).tobytes(), role, trial)

    def submit(self, position, role='primary', trial=None):
        key = self.key(position, role, trial)
        with self.lock:
            if key not in self.futures:
                self.futures[key] = self.executor.submit(self.renderer._render_one, position, role, trial)
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
