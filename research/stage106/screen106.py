"""Predeclared continuous RAW screening; selection uses current observability."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,collections
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage104')]
import run_planar_center_forecast as entry
OUT=ROOT.parent/'output/stage106_continuous_screen_20261022'
CASE=dict(raw='E:/dataset/train/6/media/event.raw',start=110.,duration=9.5)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]
def sources():
    ns=entry.build();r=ns['fingerprints']();ns.clear();r['research/stage106/screen106.py']=sha(__file__);return r
def freeze():
    assert not OUT.exists();save(OUT/'protocol106.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),case=CASE,sources=sources(),
        design='Fixed train6 110-119.5s chosen in105 from existing ON/OFF visual samples, before dense model/forecast outputs. Run unchanged104 RAW60 pipeline. Select no segments by future errors. Current geometry/body-axis admission only; all failures retained.',
        snapshot_stride_parent=25,step_s=.02,complete_transition_requirement='Qualified four-conic model lock, then side/partial >=100ms, then qualified geometry recovery in same run; no state imported from22s or old cached runs.',
        no_parameter_tuning=True,no_new_predictor=True,default_promoted=False))
def verify():
    p=load(OUT/'protocol106.json');assert p['sources']==sources();return p
def run():
    cfg=verify()['case']
    args=[sys.executable,str(ROOT/'run_planar_center_forecast.py'),'--raw',cfg['raw'],'--output',str(OUT/'raw'),
          '--start',str(cfg['start']),'--duration',str(cfg['duration']),'--preview','--verify-frames']
    with (OUT/'parent.log').open('x',encoding='utf-8') as f:
        r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,r.returncode;check()
def check():
    verify();rr=rows(OUT/'raw/observations_forecasts.jsonl');assert len(rr)==475
    modes=collections.Counter();reasons=collections.Counter();full=[];sides=[];snapshots=[];states=[]
    cap=cv2.VideoCapture(str(OUT/'raw/forecasts.mp4'));decoded=0
    for i,r in enumerate(rr):
        q=r['frontends']['delayed_feedback'];d=q['rigid_center_diagnostic'];modes[d['mode']]+=1;reasons[d['measurement']['reason']]+=1
        if d['current']:full.append(r['t'])
        if r['body_axis'].get('valid'):sides.append(r['t'])
        states.append(dict(t=r['t'],body_current=r['body_observation'].get('detected',False),
            rotor_count=sum(z['observed_now'] for z in r['sensing']['rotors']),geometry_current=d['current'],
            initialized=d['ready'],mode=d['mode'],reason=d['measurement']['reason'],
            line_valid=r['body_axis'].get('valid',False),line_ratio=r['body_axis'].get('ratio')))
        ok,image=cap.read();assert ok;decoded+=1
        if i%25==0:
            box=r['body_observation'].get('bbox')
            if box is not None:
                b=np.asarray(box);lo=np.maximum(np.int32(b[:2]-30),0);hi=np.minimum(np.int32(b[2:]+30),[1280,720]);crop=image[lo[1]:hi[1],lo[0]:hi[0]]
            else:crop=image
            tile=np.full((220,320,3),240,np.uint8);scale=min(320/max(crop.shape[1],1),185/max(crop.shape[0],1));small=cv2.resize(crop,None,fx=scale,fy=scale)
            hh,ww=small.shape[:2];tile[:hh,:ww]=small
            cv2.putText(tile,f"{r['t']:.2f}s four={states[-1]['rotor_count']} line={states[-1]['line_valid']}",(5,207),0,.42,(20,30,40),1);snapshots.append(tile)
    ok,_=cap.read();assert not ok;cap.release()
    grid=np.full((int(np.ceil(len(snapshots)/4))*220,1280,3),255,np.uint8)
    for i,im in enumerate(snapshots):grid[i//4*220:(i//4+1)*220,i%4*320:(i%4+1)*320]=im
    cv2.imwrite(str(OUT/'fixed_contact106.png'),grid)
    # Count actual lock -> >=100ms missing current geometry -> recovery, never
    # assume a visual resemblance constitutes a successful model lock.
    intervals=[];last_full=None;gap_begin=None
    for s in states:
        if s['geometry_current']:
            if gap_begin is not None and last_full is not None and s['t']-gap_begin>=.1-1e-8:
                intervals.append(dict(last_full=last_full,missing_begin=gap_begin,recovered=s['t'],duration=s['t']-gap_begin))
            last_full=s['t'];gap_begin=None
        elif last_full is not None and gap_begin is None:gap_begin=s['t']
    result=dict(parent=len(rr),native10=2*len(rr),decoded_frames=decoded,modes=dict(modes),reasons=dict(reasons),
        body_current=sum(s['body_current'] for s in states),line_current=len(sides),geometry_current=len(full),
        first_geometry=None if not full else full[0],last_geometry=None if not full else full[-1],complete_transitions=intervals,
        states=states,no_future_reference_read=True,no_claimed_prediction_accuracy=True,
        actual_RAW=True,source_unchanged=load(OUT/'raw/completion.json')['source_hashes_unchanged'])
    save(OUT/'screen106.json',result);print(json.dumps({k:v for k,v in result.items() if k!='states'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check']);a=p.parse_args();globals()[a.action]()
