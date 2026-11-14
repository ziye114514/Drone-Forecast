"""Fixed four-rotor input control on the same visible interval, no retuning."""
from pathlib import Path
from types import FunctionType
import argparse,datetime,inspect,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage94'),str(ROOT/'research/stage88')]
import study94 as old
from inventory88 import load,save,sha
import score88
from rotor_input99 import RotorActivity,RotorPaired
OUT=ROOT.parent/'output/stage99_four_rotor_force_input_20261022'
PREV=ROOT.parent/'output/stage95_official_visible_interval_20261022'


def sources():return {str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('rotor_input99.py'),Path(old.__file__),ROOT/'research/stage88/paired88.py',ROOT/'src/event_drone_forecast/rotor_sensing.py']}
def freeze():
    assert not OUT.exists();p=load(PREV/'protocol_feature95.json')
    save(OUT/'protocol_feature99.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        start_s=p['start_s'],duration_s=p['duration_s'],step_us=20000,
        reuse='Unchanged original RotorSensing phase/FFT from98; use sum(f_i squared) with four fresh references.',
        current='Require all four currently seen and each frequency_t exactly current t. No treating coasting frequency as a fresh motor measurement.',
        scalar='sqrt(mean(f_i squared)) to unchanged72 continuity tracker; reference-normalised square equals relative squared-sum. No calibrated thrust or known blade/harmonic truth.',
        controls='Same direction/RLS/10-update warmup/modes/fallback and constant/zero/lag/3past-shuffle. No threshold retuning.',
        scope='194 development feasibility; not blind benchmark or RAW-only deployment; no default promotion.'))
    save(OUT/'download94.json',load(old.OUT/'download94.json'));(OUT/'input').mkdir(exist_ok=True);(OUT/'input/coordinates.txt').write_bytes((old.OUT/'input/coordinates.txt').read_bytes())
    print('99 per-rotor squared-sum fixed protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol_feature99.json');assert p['source_hashes']==sources();return p
def run(prefix=False):
    verify();ns=dict(old.run.__globals__,OUT=OUT,verify=verify,sources=sources,StreamingActivity=RotorActivity,Paired=RotorPaired)
    return FunctionType(old.run.__code__,ns)(prefix)
def score():
    p=verify();assert not (OUT/'scores99.json').exists();assert load(OUT/'fred/prefix194/completion.json')['actual_event_prefix_exact']
    (OUT/'annotations').mkdir(exist_ok=True);(OUT/'annotations/194.txt').write_bytes((OUT/'input/coordinates.txt').read_bytes())
    text=inspect.getsource(score88.fred_score);a='t<20.6-1e-8 or t>49.6+1e-8';b=f't<{p["start_s"]+.6}-1e-8 or t>{p["start_s"]+p["duration_s"]-.4}+1e-8'
    assert text.count(a)==1;ns=dict(score88.fred_score.__globals__,OUT=OUT,METHODS=old.METHODS);exec(compile(text.replace(a,b),'<99 same95 interval scoring>','exec'),ns)
    protocol=load(ROOT.parent/'output/stage88_causal_event_controls_20261022/protocol88.json');protocol['fred_cases']=[194]
    details,audit=ns['fred_score'](protocol);table=score88.summarise(details)
    trace=score88.rows(OUT/'fred/sequence194/forecasts.jsonl')
    evidence=dict(four_frequencies_ticks=sum(r['diagnostics']['frequency']['four_frequencies'] for r in trace),
        four_measured_current_ticks=sum(r['diagnostics']['frequency']['all_four_measured_current'] for r in trace),
        relative_frequency_current_ticks=sum(r['diagnostics']['frequency']['current'] for r in trace),
        max_gain_updates=max(r['diagnostics']['event_gain_updates'] for r in trace),max_gain=max(r['diagnostics']['event_gain'] for r in trace))
    save(OUT/'endpoint_details99.json',details);save(OUT/'scores99.json',dict(table=table,audit=audit,input_evidence=evidence,default_promoted=False))
    print('99 four-rotor input',evidence,[(r['method'],r['mean_fde']) for r in table if r['horizon']==.4 and r['group']=='all'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','score']);a=p.parse_args()
    if a.action=='prefix':run(True)
    else:globals()[a.action]()
