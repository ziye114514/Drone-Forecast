"""New official195 longer visible native-event test; fixed99 physics/controls."""
from pathlib import Path
from types import FunctionType
import argparse,datetime,inspect,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage94'),str(ROOT/'research/stage99'),str(ROOT/'research/stage88')]
import download94,study94,score88
from inventory88 import save,load,sha,parse,unique_rows
from rotor_input99 import RotorActivity,RotorPaired
OUT=ROOT.parent/'output/stage100_official195_longer_20261022'
META=ROOT.parent/'output/stage91b_FRED_all_test_screen_20261022'


def sources():return {str(p):sha(p) for p in [Path(__file__),Path(download94.__file__),Path(study94.__file__),ROOT/'research/stage99/rotor_input99.py',ROOT/'research/stage88/paired88.py',ROOT/'src/event_drone_forecast/rotor_sensing.py']}


def freeze():
    assert not OUT.exists();meta=load(META/'inventory91b.json');selected=next(r for r in meta['records'] if r['sequence']==195)
    commit=load(META/'protocol91b.json')['official_revision'];annotation=META/'annotations/195.txt'
    labels,bad=parse(annotation.read_text(encoding='utf-8-sig'));assert not bad;labels=unique_rows(labels)
    runs=[];begin=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]['t']-labels[i-1]['t']>.101:
            runs.append(dict(start=labels[begin]['t'],end=labels[i-1]['t'],labels=i-begin));begin=i
    selected_run=max(runs,key=lambda r:(r['end']-r['start'],-r['start']))
    start=float(np.ceil(selected_run['start']*50)/50);duration=float(np.floor((selected_run['end']-start)*50)/50)
    assert duration>=10.
    save(OUT/'protocol100.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        official_revision=commit,selected=selected,all_annotation_runs=runs,selected_run=selected_run,start_s=start,duration_s=duration,
        selection='New195 recording, next-largest supplied-box metadata candidate; longest continuous provided-box interval, earliest tie; no195 events/prediction scores viewed before freeze.',
        fixed='Same99 four-current-phaseFFT squared-sum,10-update gate, RLS/geometry/modes/6400KF/router, constant/zero/lag/3past-shuffle. No threshold change.',
        interpretation='Visibility/size-conditioned new-recording transfer, not full benchmark. Metadata looked at, no blind or mechanicalRPM truth claim.',
        max_native_compressed_bytes=1250*1024*1024,default_promoted=False))
    # Existing94 downloader is reused byte-for-byte; its generic member path
    # comes from selected195. Its old sequence label is corrected afterwards.
    save(OUT/'protocol94.json',dict(source_sha256=sha(Path(download94.__file__)),official_revision=commit,selected=selected,
        annotation_path=str(annotation),native_event_download_max_compressed_bytes=1250*1024*1024))
    print('100 new longer195 frozen',start,start+duration,flush=True)


def verify():
    p=load(OUT/'protocol100.json');assert p['source_hashes']==sources();return p


def download():
    verify();fn=FunctionType(download94.download.__code__,dict(download94.download.__globals__,OUT=OUT))
    fn();m=load(OUT/'download94.json');assert m['member'].startswith('195/');m['sequence']=195;m['metadata_label_correction']='Generic94 downloader member is195; legacy label194 corrected, native SHA/CRC unchanged.'
    save(OUT/'download94.json',m)


def run(prefix=False):
    verify();source=inspect.getsource(study94.run);assert 'sequence194' in source
    source=source.replace('194','195')
    ns=dict(study94.run.__globals__,OUT=OUT,verify=verify,sources=sources,StreamingActivity=RotorActivity,Paired=RotorPaired)
    exec(compile(source,'<100 unmodified94 loop with195 paths and fixed99 input>','exec'),ns)
    return ns['run'](prefix)


def score():
    p=verify();assert not (OUT/'scores100.json').exists();assert load(OUT/'fred/prefix195/completion.json')['actual_event_prefix_exact']
    (OUT/'annotations').mkdir(exist_ok=True);(OUT/'annotations/195.txt').write_bytes((OUT/'input/coordinates.txt').read_bytes())
    text=inspect.getsource(score88.fred_score);a='t<20.6-1e-8 or t>49.6+1e-8';b=f't<{p["start_s"]+.6}-1e-8 or t>{p["start_s"]+p["duration_s"]-.4}+1e-8'
    assert text.count(a)==1;ns=dict(score88.fred_score.__globals__,OUT=OUT,METHODS=study94.METHODS);exec(compile(text.replace(a,b),'<100 same88 scoring, new195 fixed interval>','exec'),ns)
    protocol=load(ROOT.parent/'output/stage88_causal_event_controls_20261022/protocol88.json');protocol['fred_cases']=[195]
    details,audit=ns['fred_score'](protocol);table=score88.summarise(details)
    trace=score88.rows(OUT/'fred/sequence195/forecasts.jsonl')
    evidence=dict(four_frequencies_ticks=sum(r['diagnostics']['frequency']['four_frequencies'] for r in trace),
        four_measured_current_ticks=sum(r['diagnostics']['frequency']['all_four_measured_current'] for r in trace),
        relative_frequency_current_ticks=sum(r['diagnostics']['frequency']['current'] for r in trace),
        max_gain_updates=max(r['diagnostics']['event_gain_updates'] for r in trace),
        active_physical_ticks=sum(r['diagnostics']['fresh'] for r in trace))
    save(OUT/'endpoint_details100.json',details);save(OUT/'scores100.json',dict(table=table,audit=audit,input_evidence=evidence,
        fixed99_on_new_recording=True,default_promoted=False,mechanical_RPM_truth=False))
    print('100 new195 longer test',evidence,[(r['method'],r['valid'],r['mean_fde']) for r in table if r['horizon']==.4 and r['group']=='all'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','download','run','prefix','score']);a=p.parse_args()
    if a.action=='prefix':run(True)
    else:globals()[a.action]()
