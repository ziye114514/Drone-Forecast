from pathlib import Path
import argparse,copy,datetime,hashlib,importlib.util,json,os,sys
import numpy as np
from compose48 import *
sys.path.insert(0,str(ROOT/'research/stage45'));import graph45
OUT=ROOT.parent/'output/stage48_body_rotor_physics_20261019';POSITIONS=graph45.OUT/'sequence';SENSING=ROOT.parent/'output/stage34_slot_frequency_forecast_20261018'
CASES={'clear6':dict(start=22.,duration=2.,geometry='stage12_clear_22_27_20261017/observations.jsonl'),
    'side6':dict(start=108.,duration=5.,geometry='stage12_side_108_120_20261017/observations.jsonl'),
    'train3':dict(start=25.,duration=6.,geometry='stage31_cross_recording_physics_20261018/train3/observations_forecasts.jsonl'),
    'negative5':dict(start=0.,duration=1.,geometry=None)}
def load(p):return graph45.load(p)
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(map(json.loads,p.read_text(encoding='utf-8').splitlines()))
def sources():
    result=graph45.sources()
    for key,value in load(SENSING/'protocol.json')['source_hashes'].items():assert sha(ROOT/key)==value;result[key]=value
    for n in ['compose48.py','study48.py']:result['research/stage48/'+n]=sha(Path(__file__).with_name(n))
    return result
def inputs():
    result={}
    for c,cfg in CASES.items():
        paths=[POSITIONS/c/'observations.jsonl',POSITIONS/c/'forecasts.jsonl']
        if cfg['geometry']:paths += [SENSING/c/'sensing.jsonl',ROOT.parent/'output'/cfg['geometry']]
        result[c]={p.relative_to(ROOT.parent).as_posix():sha(p) for p in paths}
    return result
def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,parameters=PARAMS,input_hashes=inputs(),label_hashes=load(POSITIONS/'protocol.json')['label_hashes'],
        design='Three same-state expert comparisons on original34 legacy/slotfix sensor data +45 improved body positions:13s positive650 ticks +1s50known-empty, total700 native20. Frozen cached RAW outputs consumed at exact current cutoff only, no new decoder/FFT accuracy claim. Same six history coefficients/validity must match45 exactly on every branch. Save all15 forecasts first, score sparse true future endpoints after.',
        limits='Native20 position filter/axis update differs from older10ms stage34; do not compare conditional means across windows/method rates. Current side5s not full12s. Previous data seen, no blind claim. No independent physical pose/RPM or true mass/wind truth.',
        prior_art='https://arxiv.org/abs/2604.18289; execute local independent pixel adaptation of paper-style ellipse/frequency/thrust modules, not author released code or proof of author3D/RPM scores.'))
    print('Stage48 same-position body/rotor physics frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];assert inputs()==p['input_hashes'];return p
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name=='targets.json' or path.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()):raise RuntimeError('Physics replay truth forbidden: '+str(path))
def infer():
    sys.addaudithook(guard);protocol=verify()
    for case,cfg in CASES.items():
        position_rows=[r for r in rows(POSITIONS/case/'observations.jsonl') if cfg['start']<r['t']<=cfg['start']+cfg['duration']];forecast={round(r['t'],8):r['frontends']['body_first'] for r in rows(POSITIONS/case/'forecasts.jsonl')}
        if cfg['geometry']:
            geometry={round(r['t'],8):r['observation'] for r in rows(ROOT.parent/'output'/cfg['geometry'])};sensor={round(r['t'],8):r for r in rows(SENSING/case/'sensing.jsonl')}
        models={m:Predictor('canonical_static') for m in METHODS};selectors={m:Selector() for m in METHODS};epochs={m:0 for m in METHODS};stats={m:dict(physics_active=0,physics_resets=0,association_dropped=0,updates=0) for m in METHODS};folder=OUT/case;folder.mkdir(parents=True,exist_ok=False)
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for row in position_rows:
                t=row['t'];key=round(t,8);position=row['methods']['body_first']['observation'];g=geometry[key] if cfg['geometry'] else dict(detected=False);sr=sensor[key] if cfg['geometry'] else None;obs=observation(position,g);predictions={}
                for method in METHODS:
                    sensed=copy.deepcopy(sr['methods']['original' if method=='legacy_rotors' else 'replace_stale']) if sr else empty_sensing();audit=None
                    if method!='legacy_rotors' and sr:
                        epoch=sr['slot_audit']['calibration_epoch']
                        if epoch!=epochs[method]:models[method].reset_features();stats[method]['physics_resets']+=1;epochs[method]=epoch
                    if method=='associated_slotfix':
                        sensed,audit=associate(sensed,position);stats[method]['association_dropped']+=audit['dropped']
                    q,_=models[method].step(dict(t=t,observation=obs,sensing=sensed));q=selectors[method].step(q,obs);q['association_diagnostic']=audit
                    assert q['valid']==forecast[key]['valid'],(case,t,method,'validity differs')
                    for model in prior.robust.MODELS:assert q['coefficients'].get(model)==forecast[key]['coefficients'].get(model),(case,t,method,model,'history state differs')
                    stats[method]['physics_active']+=q['feature_diagnostic']['active'];predictions[method]=q
                f.write(json.dumps(dict(t=t,frontends=predictions),allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(position_rows),history_all_six_exact_to45_on_all_branches=True,methods=stats,labels_read=False,sources_unchanged=sources()==protocol['source_hashes']));print(case,json.dumps(stats),flush=True)
def score():
    verify();spec=importlib.util.spec_from_file_location('body_physics_sparse48',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=METHODS;m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate;m.label_path=lambda c:ROOT.parent/'output/raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');m.score()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score']);globals()[p.parse_args().action]()
