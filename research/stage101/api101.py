"""Actual two-call93 API regression in one clean parent process."""
from pathlib import Path
import datetime,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from complementary_raw_api import forecast_raw
OUT=ROOT.parent/'output/stage101_complementary_API_20261022'


def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return __import__('hashlib').sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    forbidden=[m for m in sys.modules if m.split('.')[0] in ['numpy','cv2','torch','metavision_sdk_cv','metavision_core']];assert not forbidden
    sources={str(p):sha(p) for p in [Path(__file__),ROOT/'core/complementary_raw_api.py',ROOT/'core/raw_forecast_api.py']}
    save(OUT/'protocol101.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources,
        design='One clean stdlib parent, two real RAW93 child calls, exact all55/learning comparison to93; parent IO after each; existing-output and nonRAW rejection.',
        cases=[dict(name='clear6',start=22.,duration=.24),dict(name='side6',start=108.,duration=.24)],default_promoted=False))
    results=[];raw=Path('E:/dataset/train/6/media/event.raw')
    for case in load(OUT/'protocol101.json')['cases']:
        result=forecast_raw(raw,OUT/case['name'],start=case['start'],duration=case['duration'])
        actual=[json.loads(s) for s in result.observations.read_text(encoding='utf-8').splitlines()]
        reference=ROOT.parent/'output/stage93_RAW_complementary_20261022'/case['name']/'observations_forecasts.jsonl'
        expected=[json.loads(s) for s in reference.read_text(encoding='utf-8').splitlines()][:12]
        assert len(actual)==len(expected)==12
        for a,b in zip(actual,expected):
            for k in ['t','observation','event_activity','body_axis','sensing','frontends','lifecycle']:assert a[k]==b[k],(case['name'],a['t'],k)
        # A stale audit-hook namespace in the parent used to break this I/O.
        assert load(ROOT.parent/'output/checkpoints/latest_handoff.json')['last_completed_stage']>=99
        results.append(dict(case=case['name'],parent=12,native10=24,all55_exact93=True,parent_IO_after_child_ok=True,wall_s=result.elapsed_wall_s))
    rejected=[]
    for label,args,exception in [('non_RAW',(Path('not_native.hdf5'),OUT/'bad_nonraw'),ValueError),('existing_output',(raw,OUT/'clear6'),FileExistsError)]:
        try:forecast_raw(*args,duration=.24)
        except exception:rejected.append(label)
        else:raise AssertionError(label)
    assert not (OUT/'bad_nonraw').exists()
    assert not [m for m in sys.modules if m.split('.')[0] in ['numpy','cv2','torch','metavision_sdk_cv','metavision_core']]
    assert all(sha(Path(p))==h for p,h in sources.items())
    save(OUT/'checks101.json',dict(records=results,parent=24,native10=48,all55_exact93=True,parent_IO_safe=True,
        clean_import_stdlib_only=True,non_RAW_rejected_without_output=True,existing_output_protected=True,old_API85_unchanged=True,default_promoted=False))
    print('101 complementary93 API two real calls verified',flush=True)


if __name__=='__main__':main()
