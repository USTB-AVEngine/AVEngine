"""Retry automatic static layouts when their original native color proof is missing.

This is an upstream construction check. The original appearance interpreter,
question candidates, five-binding gate and answer/scoring rules are unchanged.
Explicit offscreen and occlusion profiles do not use this constructor.
"""
from __future__ import annotations


def native_binding_readiness(scene, visual, appearance, assets):
    profile=scene.get('production_scenario', {})
    if (profile.get('camera_policy') != 'binding_static'
            or not str(profile.get('recipe_revision', '')).startswith('kujiale-fixed-natural-construction-')):
        return {'status': 'not_requested', 'reason': 'not an automatic three-person binding layout'}
    expected={a['endpoint_id']: a['asset_id'] for a in assets}
    actors={a['endpoint_id']: a for a in visual['frames'][0]['actors']
            if a.get('endpoint_id') in expected}
    reviews=appearance.get('review', {}).get('actors', {})
    checks=[]
    for endpoint, asset in expected.items():
        actor=actors.get(endpoint)
        review=reviews.get(str(actor['actor_id']), {}) if actor else {}
        passed=bool(actor and actor.get('asset_id') == asset
                    and review.get('status') == 'reviewed' and review.get('frame_refs'))
        checks.append({'endpoint_id': endpoint, 'asset_id': asset,
                       'actor_id': actor.get('actor_id') if actor else None,
                       'status': 'pass' if passed else 'needs_new_layout',
                       'original_review_frame_refs': list(review.get('frame_refs', [])),
                       'original_review_method': review.get('review_method'),
                       'native_frame0_visible_pixels': actor.get('visible_pixels') if actor else None})
    return {'status': 'pass' if checks and all(c['status'] == 'pass' for c in checks) else 'needs_new_layout',
            'checks': checks, 'construction_profile': profile['recipe_revision'],
            'authority': 'unchanged complete native RGB/mask appearance review',
            'action_on_missing': 'fresh bounded same-room layout attempt; retain all required people and animals',
            'question_thresholds_changed': False, 'native_review_modified': False}
