"""Causal hybrid pilot from fixed124 observations; explicitly cached input."""
from pathlib import Path
import argparse,collections,datetime,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage124'),str(ROOT/'research/stage125')]
from study124 import OUT as INPUT,CASES,rows,sha,save,stable
from hybrid125 import Hybrid,PARAMS,NAMES
OUT=ROOT.parent/'output/stage125_complementary_support_20261023'
def freeze():
    assert not OUT.exists()
    save(OUT/'protocol125.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),parameters=PARAMS,cases=list(CASES),
        sources={p.name:sha(p) for p in [Path(__file__),ROOT/'research/stage125/hybrid125.py']},source_protocol124=sha(INPUT/'protocol124.json'),
        scope='Cached current124 geometry/NN/axis pilot, not complete RAW deployment. Rule fixed before future-box scoring. First partial124 progress already inspected: development not blind.',
        causality='Replay only successive same-time current observations. Independently reset and stop at60 ticks for active prefix; later actual RAW deployment required.'))
def verify():
    p=json.loads((OUT/'protocol125.json').read_text('utf-8'));assert p['sources']=={x.name:sha(x) for x in [Path(__file__),ROOT/'research/stage125/hybrid125.py']};return p
def run():
    verify();table=[]
    for case in CASES:
        assert (INPUT/case/'checks124.json').exists();rr=rows(INPUT/case/'records.jsonl');model=Hybrid();output=[]
        for r in rr:
            co=r['outputs']['count_global'];oc=r['outputs']['occupancy_roi']
            # Actual full baseline measurements are serialized inside the
            # center only before fixed shape is used. Recompute original104
            # checks to avoid substituting fixed-pose or three-input fits.
            o=model.step(r['t'],r['body'],r['axis'],co['geometry'].get('rotors',[]),oc['geometry'].get('rotors',[]))
            assert o['forecast']['coefficients'].get(NAMES[0])==co['forecast']['coefficients'].get(NAMES[0])
            output.append(dict(t=r['t'],output=o))
        save(OUT/(case+'.json'),output)
        prefix=Hybrid()
        for r,o in zip(rr[:60],output[:60]):
            co=r['outputs']['count_global'];oc=r['outputs']['occupancy_roi'];a=prefix.step(r['t'],r['body'],r['axis'],co['geometry'].get('rotors',[]),oc['geometry'].get('rotors',[]));assert stable(a)==stable(o['output'])
        table.append(dict(case=case,frames=len(rr),chosen=dict(collections.Counter(o['output']['chosen'] for o in output)),
            first_structure=next((o['t'] for o in output if o['output']['center'].get('structure_ready')),None),current=sum(o['output']['center']['current'] for o in output),prefix_exact=True))
    save(OUT/'availability125.json',table);print(json.dumps(table),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
