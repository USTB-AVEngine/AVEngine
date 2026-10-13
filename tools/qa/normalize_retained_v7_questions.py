#!/usr/bin/env python3
"""Export a format-only copy of a sealed v7 delivery; media remain read-only."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from avengine.qa.answer_format import normalize_export


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scoring_row(public, private):
    """Recover the declared open truth, without transforming model output."""
    truth = deepcopy(private['canonical_answer'])
    kind = public.get('answer_type') or private.get('authority_truth', {}).get('answer_type')
    qa = public['qa_id']
    if qa in {'QA-13', 'QA-25'} and kind != 'choice':
        kind = 'angle_deg'
    if kind is None:
        kind = 'closed_set'
    data = {'answer_type': kind, 'truth': truth}
    if kind == 'angle_deg':
        data['convention'] = 'left_positive'
    elif kind in {'count_single', 'count_pair'}:
        data['truth'] = truth if isinstance(truth, list) else [truth]
    elif kind == 'closed_set':
        label = str(truth)
        data['classes'] = {label: [label]}
    return {'question_id': public['question_id'], 'qa_id': qa, 'forms': {'open': data}}


def text(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)


def export(manifest_path, output, scorer_path, source_commit):
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output must be a new directory: {output}')
    manifest = read(manifest_path)
    spec = importlib.util.spec_from_file_location('retained_v7_existing_scorer', scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scorer
    spec.loader.exec_module(scorer)
    source_hashes = {str(manifest_path): sha(manifest_path), str(scorer_path): sha(scorer_path)}
    output.mkdir(parents=True)
    lineage, checks, views = [], [], []
    by_family = Counter()
    for parent in manifest['logical_parent_statuses']:
        pilot_path = Path(parent['source'])
        pilot = read(pilot_path)
        source_hashes[str(pilot_path)] = sha(pilot_path)
        seal = pilot_path.parent
        parent_out = output / 'parents' / parent['scene_id']
        for window in pilot['windows']:
            wid = window['window_id']
            old_qpath = seal / 'windows' / wid / 'questions.json'
            old_apath = seal / 'private' / (wid + '_answers.json')
            public, private = read(old_qpath), read(old_apath)
            source_hashes.update({str(p): sha(p) for p in (old_qpath, old_apath)})
            if any(q.get('form', 'open') != 'open' for q in public):
                raise ValueError('This copy tool expects the sealed open-only phase-one delivery')
            new_public, new_private = normalize_export(public, private)
            old_gold = {a['question_id']: a for a in private}
            new_gold = {a['question_id']: a for a in new_private}
            assert [q['question_id'] for q in public] == [q['question_id'] for q in new_public]
            assert list(old_gold) == list(new_gold)
            for q, nq in zip(public, new_public):
                old, new = old_gold[q['question_id']], new_gold[q['question_id']]
                for key in old:
                    if key not in {'answer', 'teacher_answer'}:
                        assert old[key] == new[key], (q['question_id'], key)
                for key in q:
                    if key not in {'question_en', 'question_zh'}:
                        assert q[key] == nq[key], (q['question_id'], key)
                item = scoring_row(q, old)
                before = scorer.score_answer(item, text(old['answer']), form='open')
                after = scorer.score_answer(item, new['teacher_answer'], form='open')
                assert before['score'] == after['score'] == 1.0, (q['question_id'], before, after)
                if item['forms']['open']['answer_type'] in {'angle_deg', 'count_pair', 'count_single', 'time_s', 'time_range_s'}:
                    assert before.get('parsed') == after.get('parsed'), (q['question_id'], before, after)
                checks.append({'question_id': q['question_id'], 'qa_id': q['qa_id'],
                               'old_answer': old['answer'], 'new_answer': new['teacher_answer'],
                               'canonical_answer': old['canonical_answer'], 'old_score': before,
                               'new_score': after, 'source_questions': str(old_qpath),
                               'source_answers': str(old_apath)})
                by_family[q['qa_id']] += 1
                original_media = deepcopy(q.get('media', {}))
                resolved_media = {key: str((old_qpath.parent / value).resolve())
                                  if isinstance(value, str) and value.endswith(('.mp4', '.wav')) else value
                                  for key, value in original_media.items()}
                views.append({'id': q['question_id'], 'question_id': q['question_id'], 'qa_id': q['qa_id'],
                              'form': 'open', 'prompt': nq.get('question_en', ''),
                              'prompt_zh': nq.get('question_zh', ''), 'teacher_answer': new['teacher_answer'],
                              'canonical_answer': new['canonical_answer'], 'media': resolved_media,
                              'source_media_directory': str(old_qpath.parent),
                              'source_questions': str(old_qpath), 'source_answers': str(old_apath)})
            qpath = parent_out / 'windows' / wid / 'questions.json'
            apath = parent_out / 'private' / (wid + '_answers.json')
            write(qpath, new_public)
            write(apath, new_private)
            write(qpath.parent / 'MEDIA_SOURCE.json', {'media_root': str(old_qpath.parent),
                  'reason': 'Question JSON retains its original relative media values; assembly resolves them from this read-only source.'})
            lineage.append({'scene_id': parent['scene_id'], 'window_id': wid,
                            'source_questions': str(old_qpath), 'source_answers': str(old_apath),
                            'questions': str(qpath), 'answers': str(apath), 'count': len(public)})
        write(parent_out / 'source_pilot_manifest.json', pilot)
    for path, digest in source_hashes.items():
        assert sha(path) == digest, ('Read-only source changed during export', path)
    with (output / 'training_views.jsonl').open('w') as fh:
        for view in views:
            fh.write(json.dumps(view, ensure_ascii=False) + '\n')
    write(output / 'row_comparisons.json', checks)
    result = {'status': 'PASS', 'source_manifest': str(manifest_path), 'source_code_commit': source_commit,
              'source_parent_count': len(manifest['logical_parent_statuses']), 'windows': len(lineage),
              'questions': len(views), 'by_family': dict(by_family),
              'scorer': str(scorer_path), 'scorer_sha256': sha(scorer_path),
              'scoring_projection': 'Original declared open type and canonical truth; exact canonical closed-set aliases; scorer default tolerances unchanged; outputs passed directly.',
              'ids_identical': True, 'canonical_truth_identical': True, 'private_evidence_identical': True,
              'scores_identical': True, 'all_scores': 1.0, 'source_hashes_after_match': True,
              'source_sha256': source_hashes, 'lineage': lineage,
              'media_and_facts': 'Read-only references to the sealed source; no rendering, sampling or seed change.'}
    write(output / 'COPY_MANIFEST.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scorer', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    result = export(args.manifest, args.output, args.scorer, args.source_commit)
    print(json.dumps({k: result[k] for k in ('status', 'source_parent_count', 'windows', 'questions', 'by_family')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
