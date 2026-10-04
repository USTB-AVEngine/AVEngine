"""Stream actual modal/target-only masks into unchanged v6 pixel predicates."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import numpy as np
from avengine.qa.pixel_visibility import _frame_truth

class DenseNativeVisibility:
    """One observed frame at a time; no hand-authored visibility state is accepted."""
    def __init__(self, targets, registry, resolution_hw=(720, 1280)):
        self.targets = {str(k): int(v) for k, v in targets.items()}
        if not self.targets or len(set(self.targets.values())) != len(self.targets):
            raise ValueError('native targets need distinct semantic IDs')
        self.registry = deepcopy(registry)
        self.resolution_hw = tuple(resolution_hw)
        self.frames = {k: [] for k in self.targets}
        self.occlusion = []
        self.seen = set()
        self.semantic_to_instance = {}
        for key, entry in registry.items():
            sid = entry.get('semantic_id') if isinstance(entry, dict) else None
            if not isinstance(sid, int) or sid <= 0 or sid in self.semantic_to_instance:
                raise ValueError('occluder registry requires unique native semantic IDs')
            self.semantic_to_instance[sid] = str(key)

    def add(self, actor_id, frame_index, modal_mask, target_only_mask, *, native_receipt):
        actor = str(actor_id)
        if actor not in self.targets or not native_receipt:
            raise ValueError('unregistered target or missing native render receipt')
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError('native frame index must be a nonnegative integer')
        key = actor, frame_index
        if key in self.seen:
            raise ValueError('duplicate native target/frame')
        modal, target = np.asarray(modal_mask), np.asarray(target_only_mask)
        if modal.shape != self.resolution_hw or target.shape != modal.shape:
            raise ValueError('native mask resolution changed')
        if modal.dtype.kind not in 'iu' or target.dtype.kind not in 'iu':
            raise ValueError('native semantic masks must have integer labels')
        sid = self.targets[actor]
        row = _frame_truth(modal_mask=modal, target_only_mask=target,
                           target_semantic_id=sid, frame_index=frame_index)
        hidden = (target == sid) & (modal != sid)
        values, counts = np.unique(modal[hidden], return_counts=True)
        known, unknown = [], 0
        for value, count in zip(values, counts):
            instance = self.semantic_to_instance.get(int(value))
            if instance is None:
                unknown += int(count)
            else:
                known.append(instance)
        self.frames[actor].append(dict(row, native_receipt=native_receipt))
        self.occlusion.append({'target_instance_id': actor, 'frame_index': frame_index,
                               'occluder_instance_ids': sorted(known),
                               'unknown_occluder_pixels': unknown,
                               'occluded_pixel_count': int(hidden.sum()),
                               'native_receipt': native_receipt})
        self.seen.add(key)

    def finish(self, frame_count, *, frame_index_scope='scene'):
        expected = set(range(frame_count))
        for actor, rows in self.frames.items():
            if {row['frame_index'] for row in rows} != expected:
                raise ValueError('missing actual modal/target-only frame for target ' + actor)
        if frame_index_scope != 'scene':
            raise ValueError('dense producer uses the observed scene clock')
        # Unknown stage/background IDs do not establish a unique QA-10 object.
        # Keep exact measured unknown pixels in the sidecar; the adapter withholds
        # those frame identities rather than certifying a fabricated object.
        certifiable = [r for r in self.occlusion if r['unknown_occluder_pixels'] == 0]
        return {'pixel_visibility_truth': {
                    'schema': 'avengine_qa_pixel_visibility_truth_v1',
                    'status': 'computed_modal_target_only_v1',
                    'resolution_hw': list(self.resolution_hw),
                    'per_instance': {a: {'semantic_id': self.targets[a],
                        'frames': sorted(rows, key=lambda r: r['frame_index']),
                        'state_counts': dict(Counter(r['state'] for r in rows))}
                                     for a, rows in self.frames.items()}},
                'occluder_registry': deepcopy(self.registry),
                'occluder_evidence': {
                    'frame_index_scope': 'scene', 'frame_records': certifiable},
                'unattributed_occlusion_frames': [r for r in self.occlusion if r['unknown_occluder_pixels']],
                'automatic': True, 'human_viewing_required': False,
                'visibility_predicate_source': 'unchanged avengine.qa.pixel_visibility._frame_truth'}
