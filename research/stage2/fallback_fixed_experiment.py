"""Versioned rerun after a logical reset fix; no threshold/parameter tuning."""
from pathlib import Path
import argparse,datetime,json
import fallback_experiment as harness
from activity_fallback_fixed import ActivityTracker
from study import OUT,save,digest,verify

DEST=OUT/'activity_fallback_fixed'

def check():
    verify();p=json.loads((DEST/'protocol.json').read_text())
    for name,h in p['sources'].items():assert digest(Path(__file__).with_name(name))==h
    return p

def freeze():
    verify();assert (OUT/'activity_fallback/sequence2/scores.json').exists()
    if (DEST/'protocol.json').exists():raise RuntimeError('Fixed protocol exists')
    DEST.mkdir(exist_ok=True)
    names=['fallback_fixed_experiment.py','activity_fallback_fixed.py','activity_fallback.py','fallback_experiment.py','prepare_public_archive.py','test_validation.py']
    sources={name:digest(Path(__file__).with_name(name)) for name in names}
    for name in names:(DEST/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    save(DEST/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,
        sequences=['1','2','3'],duration_s=20,thresholds_and_filters_unchanged=True,no_label_input=True,
        change='On the .3s loss reset, set last_seen=None once, so pending confirmation can accumulate again.',
        initial_failure_retained=str(OUT/'activity_fallback'),
        roles={'1':'Inspected development','2':'Already observed for failed version; no longer a pristine blind evaluation','3':'Additional transfer check not inspected for fixed method before this freeze, official train not test'},
        comparison='Same three variants, scoring and original CV bank as frozen initial attempt.',
        no_new_algorithm_or_superiority_claimed=True))
    print('Reset fix frozen; initial failure retained',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','score']);parser.add_argument('--sequence',choices=['1','2','3']);a=parser.parse_args()
    harness.DEST=DEST;harness.ActivityTracker=ActivityTracker;harness.check=check
    if a.action=='freeze':freeze()
    else:getattr(harness,a.action)(a.sequence)
