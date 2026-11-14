from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
from slots33 import ROOT,PARAMS,ReplaceStale,RotorSensing,PaperConfig,geometry_record
from checks33 import run as checks

OUT=ROOT.parent/'output/stage33_rotor_slot_association_20261018'
CASES={'clear6':dict(folder='stage12_clear_22_27_20261017',file='observations.jsonl'),
    'side6':dict(folder='stage12_side_108_120_20261017',file='observations.jsonl'),
    'train3':dict(folder='stage31_cross_recording_physics_20261018/train3',file='observations_forecasts.jsonl')}


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def input_path(c):return ROOT.parent/'output'/CASES[c]['folder']/CASES[c]['file']


def sources():
    result={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('slots33.py'),Path(__file__).with_name('checks33.py')]}
    for name,value in load(ROOT.parent/'output/stage31_cross_recording_physics_20261018/protocol.json')['source_hashes'].items():assert sha(ROOT/name)==value;result[name]=value
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,cases=CASES,
        input_hashes={c:sha(input_path(c)) for c in CASES},design='Replay only association using original observations, with EMPTY event histories so no frequency estimation. Original sensor geometry/IDs must exactly reproduce saved original sensing records at all ticks; replace-stale changes ONLY slot lifecycle. No inference labels.',
        prior_art='https://github.com/tryolabs/norfair/blob/master/norfair/tracker.py inspected for unmatched-track lifecycle. Existing local RotorSensing/SciPy reused, Norfair not installed/executed.',
        scope='Internal current-rotor representation and four-visible availability, not true ID or RPM accuracy. No actual new frequency/prediction claims. No G/yolov10-main.'))
    print('Stage33 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run():
    protocol=verify();table=[]
    for case in CASES:
        assert sha(input_path(case))==protocol['input_hashes'][case]
        rows=list(map(json.loads,input_path(case).read_text(encoding='utf-8').splitlines()));sensors={'original':RotorSensing(PaperConfig()),'replace_stale':ReplaceStale(PaperConfig())}
        counts={m:dict(four_visible=0,unrepresented_current=0,frontend_four=0,retired=0,new_ids=0) for m in sensors};records=[]
        for r in rows:
            result={m:s.step(r['observation'],[],r['t']) for m,s in sensors.items()}
            assert geometry_record(result['original'])==geometry_record(r['sensing']),(case,r['t'],'original ID geometry differs')
            record=dict(t=r['t'],methods={})
            for method,sensing in result.items():
                observed=sum(x['observed_now'] for x in sensing['rotors']);n=len(r['observation']['rotors']);audit=getattr(sensors[method],'audit',{})
                counts[method]['four_visible']+=sensing['four_visible'];counts[method]['unrepresented_current']+=n>observed;counts[method]['frontend_four']+=n==4;counts[method]['retired']+=len(audit.get('retired_ids',[]))
                record['methods'][method]=dict(rotors=geometry_record(sensing),four_visible=sensing['four_visible'],audit=audit)
            records.append(record)
        folder=OUT/case;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'association.jsonl').exists()
        with (folder/'association.jsonl').open('x',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
        for m,s in sensors.items():counts[m]['new_ids']=s.next_id;table.append(dict(case=case,method=m,frames=len(rows),**counts[m]))
        save(folder/'completion.json',dict(frames=len(rows),original_sensing_geometry_exact=len(rows),frequency_estimation_executed=False,sources_unchanged=sources()==protocol['source_hashes']))
    save(OUT/'analysis.json',dict(table=table));print(json.dumps(table,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
