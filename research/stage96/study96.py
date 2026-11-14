"""Fixed SDK-input ablation on the same95 visible interval, no retuning."""
from pathlib import Path
from types import FunctionType
import argparse,datetime,inspect,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage94'),str(ROOT/'research/stage95'),str(ROOT/'research/stage88')]
import study94 as old
from inventory88 import load,save,sha
import score88
from native_input96 import NativeActivity,NativePaired
OUT=ROOT.parent/'output/stage96_native_frequency_input_20261022'
PREV=ROOT.parent/'output/stage95_official_visible_interval_20261022'


def sources():return {str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('native_input96.py'),Path(old.__file__),ROOT/'research/stage88/paired88.py',ROOT/'research/stage72/increment72.py',ROOT/'research/stage79/native79.py']}


def freeze():
    assert not OUT.exists();p=load(PREV/'protocol_feature95.json')
    save(OUT/'protocol_feature96.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        start_s=p['start_s'],duration_s=p['duration_s'],step_us=20000,
        input='Installed SDK strict7 already passed79/89 synthetic controls; same16 unique pixels/mode_fraction.35/ring<.5 rule.',
        continuity='Unchanged72 tracker (half/double ambiguity,20% frequency step,160ms expiry). One qualified SDK candidate with mode_fraction weight; neutral phase weight1 is NOT a phase observation.',
        changed='Only relative brightness-frequency source; same direction, Gain/RLS, mode filters, feedback, constant/zero/lag/three shuffle controls and6400 history/router.',
        interpretation='194 is now a development recording:96 motivated by95 module coverage. No blind/SOTA/mechanical-RPM claim.',
        default_promoted=False))
    save(OUT/'download94.json',load(old.OUT/'download94.json'));(OUT/'input').mkdir(exist_ok=True)
    (OUT/'input/coordinates.txt').write_bytes((old.OUT/'input/coordinates.txt').read_bytes())
    print('96 fixed native frequency input ablation frozen',flush=True)


def verify():
    p=load(OUT/'protocol_feature96.json');assert p['source_hashes']==sources();return p


def run(prefix=False):
    verify();namespace=dict(old.run.__globals__,OUT=OUT,verify=verify,sources=sources,StreamingActivity=NativeActivity,Paired=NativePaired)
    return FunctionType(old.run.__code__,namespace)(prefix)


def score():
    p=verify();assert not (OUT/'scores96.json').exists();assert load(OUT/'fred/prefix194/completion.json')['actual_event_prefix_exact']
    (OUT/'annotations').mkdir(exist_ok=True);(OUT/'annotations/194.txt').write_bytes((OUT/'input/coordinates.txt').read_bytes())
    text=inspect.getsource(score88.fred_score);a='t<20.6-1e-8 or t>49.6+1e-8';b=f't<{p["start_s"]+.6}-1e-8 or t>{p["start_s"]+p["duration_s"]-.4}+1e-8'
    assert text.count(a)==1;ns=dict(score88.fred_score.__globals__,OUT=OUT,METHODS=old.METHODS)
    exec(compile(text.replace(a,b),'<96 fixed95 interval scoring>','exec'),ns)
    protocol=load(ROOT.parent/'output/stage88_causal_event_controls_20261022/protocol88.json');protocol['fred_cases']=[194]
    detail,audit=ns['fred_score'](protocol);table=score88.summarise(detail)
    save(OUT/'endpoint_details96.json',detail);save(OUT/'scores96.json',dict(table=table,audit=audit,
        same_interval_as95=True,parameters_unchanged=True,native_frequency_not_mechanical_RPM=True,default_promoted=False))
    print('96 SDK relative input scores',[(r['method'],r['valid'],r['mean_fde']) for r in table if r['horizon']==.4 and r['group']=='all'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','score']);a=p.parse_args()
    if a.action=='prefix':run(True)
    else:globals()[a.action]()
