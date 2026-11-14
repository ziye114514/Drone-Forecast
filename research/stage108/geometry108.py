"""Current20ms geometry availability control, unchanged104 geometry gates."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage104')]
from rigid104 import PlanarCenter,PARAMS
OUT=ROOT.parent/'output/stage108_current20_geometry_20261022'
INPUT=ROOT.parent/'output/stage106_continuous_screen_20261022/raw/observations_forecasts.jsonl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/stage104/rigid104.py',ROOT/'research/stage58/geometry58.py',ROOT/'research/stage49/rigid49.py']}
def freeze():
    assert not OUT.exists();save(OUT/'protocol108.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_sha256=sha(INPUT),parameters=PARAMS,
        design='Reuse actual current20ms geometry observations already computed in106 RAW. Same104 PlanarCenter and all conic/5-frame/conditioning gates, no threshold changes, no inferred missing motors. Compare availability with existing10ms+tracking source, not new end-to-end performance.',
        reason='106 current observations counted four candidates83/475 at20ms versus42/475 at10ms before this test; chosen by current availability, not future endpoint errors. Gates may still reject.',
        current_time='20ms support ends at exact current cutoff. No next frame, box, future target or frequency generated.',
        scope='Current-feature cache geometry diagnostic only. No newRAW, frequency or prediction accuracy claim.',default_promoted=False))
def audit():
    p=json.loads((OUT/'protocol108.json').read_text('utf-8'));assert p['source_hashes']==sources() and p['input_sha256']==sha(INPUT)
    model=PlanarCenter();records=[];modes=collections.Counter();reasons=collections.Counter()
    with INPUT.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);body=r['body_observation'];rotors=r['geometry20'].get('rotors',[])
            z,d=model.step(r['t'],body,rotors,r['body_axis']);modes[d['mode']]+=1;reasons[d['measurement']['reason']]+=1
            records.append(dict(t=r['t'],observation=z,diagnostic=d))
    save(OUT/'geometry108.json',dict(records=records,modes=dict(modes),reasons=dict(reasons),
        current=sum(r['diagnostic']['current'] for r in records),initializations=sum(r['diagnostic']['initialized_now'] for r in records),
        old10ms_current=0,parameters_unchanged=True,no_future_target_read=True,new_RAW=False))
    print(json.dumps(dict(modes=modes,reasons=reasons,current=sum(r['diagnostic']['current'] for r in records))),flush=True)
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('action',choices=['freeze','audit']);a=q.parse_args();globals()[a.action]()
