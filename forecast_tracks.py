"""Versioned accumulated live recipes; original profiles stay reproducible."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'configs/live_tracks.json'
LIVE_PROFILES=('live','live-baseline','live-accumulated')
TRACK_PROFILES={'live-baseline':'baseline_v0','live-accumulated':'recommended'}

def apply_track(cfg,args):
    track=TRACK_PROFILES.get(args.profile)
    revision=getattr(args,'live_revision',None)
    if revision is not None and args.profile!='live-accumulated':raise ValueError('--live-revision requires --profile live-accumulated')
    if track is None:return cfg
    manifest=json.loads(MANIFEST.read_text('utf-8'))
    if track=='recommended':track=manifest['recommended']
    if revision is not None:
        if revision not in manifest['tracks'] or not revision.startswith('accumulated_'):raise ValueError('Unknown accumulated live revision: '+revision)
        track=revision
    recipe=manifest['tracks'][track]
    result=dict(cfg);result.update(recipe['parameters'])
    for key in ('center_ensemble_weights','center_ensemble_sha256'):result.pop(key,None)
    if not getattr(args,'detector_weights',None) and not getattr(args,'detector_sha256',None):
        result.update(detector_weights=str((ROOT/recipe['model']).resolve()),detector_sha256=recipe['sha256'])
    else:track+=':model_override'
    if getattr(args,'periodic_axis',False) and recipe.get('periodic_packet'):
        result['periodic_packet']=recipe['periodic_packet']
    args.live_recipe_id=track
    return result
