"""Causal adequacy pilot on119 features; neither new RAW nor independent data."""
from pathlib import Path
import argparse,datetime,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage121'),str(ROOT/'research/stage119')]
from validation121 import ValidatedSelector,PARAMS
from study119 import OUT as INPUT,rows,save,sha,stable,NAMES,REFERENCE
OUT=ROOT.parent/'output/stage121_causal_model_validation_20261023'

def sources():return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/stage121/validation121.py',ROOT/'research/stage119/partial119.py',ROOT/'research/stage107/center107.py']}
def freeze():
    assert not OUT.exists();save(OUT/'protocol121.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,
        inputs={f'clear6_mask{i}':sha(INPUT/f'clear6_mask{i}'/'records.jsonl') for i in range(4)},synthetic_sha256=sha(INPUT/'synthetic119.json'),
        design='Follow-up to119 failure: conditional covariance/shape spread failed to detect real conic-vs-affine model mismatch. Freeze five-check/.2s causal validation before reading121 outcomes. No new threshold search, weights learned only from prior visible-four-frame consistency.',
        limitations='Same inspected developmental feature streams and ideal synthetic model, no blind efficacy/new RAW claim. Full-four geometric center can be biased. Basic leave-one-out/model-selection is established; research candidate is event-rotor model-adequacy gating.',default_promoted=False))
def verify():
    p=json.loads((OUT/'protocol121.json').read_text('utf-8'));assert p['sources']==sources();return p

def sequence(rr):
    model=ValidatedSelector();out=[]
    for r in rr:
        q=model.step(r);out.append(dict(t=r['t'],index=r['index'],body=r['body'],input_count=len(r['input_rotors']),intervention=r.get('intervention',r['removed_one_observation']),
            removed_one_observation=r['removed_one_observation'],truth=r.get('truth'),output=q))
    return out

def run():
    p=verify()
    for mask in range(4):
        name=f'clear6_mask{mask}';source=INPUT/name/'records.jsonl';assert sha(source)==p['inputs'][name]
        rr=rows(source);out=sequence(rr);prefix=sequence(rr[:100]);assert stable(prefix)==stable(out[:100])
        save(OUT/(name+'.json'),dict(records=out,independent_prefix100_exact=True));print(name,flush=True)
    assert sha(INPUT/'synthetic119.json')==p['synthetic_sha256'];ss=json.loads((INPUT/'synthetic119.json').read_text('utf-8'))
    #119 synthetic lacks axis in saved row; reconstruct the same known axis
    #condition, without changing observation/candidate/filter inputs.
    synthetic=[]
    for case in ss['cases']:
        rr=case['records']
        for r in rr:r['axis']=dict(valid=case['case']=='edge_on' and r['intervention'],angle_rad=0.,confidence=.9)
        out=sequence(rr);assert stable(sequence(rr[:75]))==stable(out[:75]);synthetic.append(dict(case=case['case'],records=out,independent_prefix75_exact=True))
        print(case['case'],flush=True)
    save(OUT/'synthetic121.json',dict(cases=synthetic))

def evaluate():
    verify();ref=rows(REFERENCE/'clear6/records.jsonl');real=[]
    for mask in range(4):
        rr=json.loads((OUT/f'clear6_mask{mask}.json').read_text('utf-8'))['records'];errors=[];future=[];choices={}
        for i,r in enumerate(rr):
            if not r['removed_one_observation']:continue
            q=r['output'];name=q['selection']['chosen'];choices[name]=choices.get(name,0)+1
            target=ref[i]['outputs']['fixed_structure']['center']
            if target['current']:errors.append(np.linalg.norm(np.array(q['center']['center'])-target['center']))
            c=q['forecast']['coefficients'].get(NAMES[2])
            if c and i+20<len(rr):
                v=ref[i+20]['outputs']['fixed_structure']['forecast']['coefficients'].get(NAMES[2])
                if v:future.append(np.linalg.norm(np.array(c['position'])+.4*np.array(c['velocity'])-v['position']))
        real.append(dict(case=f'clear_mask{mask}',mode='validated121',center_consistency_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),reference_fde04_mean_px=float(np.mean(future)),
            center_reference_pairs=len(errors),reference_forecast_pairs=len(future),choices=choices))
    synth=[]
    for case in json.loads((OUT/'synthetic121.json').read_text('utf-8'))['cases']:
        rr=case['records'];first=next(r for r in rr if r['output']['center']['current']);offset=np.array(first['output']['forecast']['coefficients'][NAMES[2]]['position'])-first['truth'];errors=[];future=[];choices={}
        for i,r in enumerate(rr):
            if not r['intervention']:continue
            q=r['output'];name=q['selection']['chosen'];choices[name]=choices.get(name,0)+1;c=q['forecast']['coefficients'].get(NAMES[2])
            if not c:continue
            errors.append(np.linalg.norm(np.array(c['position'])-r['truth']-offset))
            if i+20<len(rr):future.append(np.linalg.norm(np.array(c['position'])+.4*np.array(c['velocity'])-rr[i+20]['truth']-offset))
        synth.append(dict(case=case['case'],mode='validated121',relative_position_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),fde04_mean_px=float(np.mean(future)),choices=choices))
    save(OUT/'scores121.json',dict(real_consistency=real,synthetic=synth,real_COM_truth=False,new_RAW=False));print(json.dumps(dict(real=real,synthetic=synth)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','evaluate']);a=p.parse_args();globals()[a.action]()
