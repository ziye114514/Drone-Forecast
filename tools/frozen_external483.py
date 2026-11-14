"""One-shot external acquisition: freeze both predictors before any frames.

The same pre-reserved10--16s interval is retained even if empty or difficult.
All predictions precede reference extraction. No model selection or code
changes are allowed based on these external frames/results in this batch.
"""
from pathlib import Path
import sys,os,json,hashlib,subprocess
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES

if __name__=='__main__':
    raise SystemExit('Archived unexecuted483 protocol: user explicitly retained blind video on2026-11-12. Prepare a NEW frozen protocol only after the30px development gate; do not run this old candidate comparison.')
    import numpy as np,cv2
    out=ROOT.parent/'output/frozen_external483_20261112';out.mkdir(exist_ok=False)
    raw='E:/dataset/recordings/rec_20261004_162506/event.raw'
    arms={'control473':'run_rotor_joint_only_candidate.py','candidate478':'run_rotor_symmetric_candidate.py'}
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCE_FILES}
    protocol=dict(raw=raw,start=10,duration=6,arms=arms,source_sha256=hashes,
        model_sha256=hashlib.sha256((ROOT/'models/native_response_plane_v1.json').read_bytes()).hexdigest(),
        candidate_selected_before_external_decode=True,all_predictions_before_reference_extraction=True,
        new_file_not_in19_experiment_acquisitions=True,prior_metadata_inventory_only=True,
        fixed25_origins='10.8:0.2:15.6',no_reselection_of_time_interval=True,
        no_tuning_after_external_results=True,reference_is_weak_native_outline_not_human_COM_GT=True,
        test_now_to_audit_generalization_at_user_latest_request=True,original30px_gate_not_achieved=True,
        FPS_work=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for arm,entry in arms.items():
        cmd=[sys.executable,str(ROOT/entry),'--raw',raw,'--start','10','--duration','6','--output',str(out/arm),
            '--gpu-python','C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',
            '--rotor-native-geometry-dll',str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')]
        print(arm,'RAW inference',flush=True)
        with (out/f'{arm}.log').open('w',encoding='utf8') as f:r=subprocess.run(cmd,cwd=ROOT.parent,stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0,(arm,r.returncode)
        print(arm,'predictions frozen',flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    (out/'both_predictions_frozen.json').write_text(json.dumps(dict(source_sha256=hashes),indent=2),'utf8')
    # Reference imagery sees ALL original event pixels and no predictor outputs.
    from live_decode import native_batches
    from reference_boxes181 import outline_box
    targets=np.round(np.arange(10.8,16.0001,.2),1);index=0;buffer=None;records={};previews=[]
    ref=out/'reference';ref.mkdir()
    for t,e in native_batches(Path(raw),10.,6.,False,33333,reader_backend='cooperative'):
        buffer=e.copy() if buffer is None else np.concatenate([buffer,e])
        while index<len(targets) and t>=targets[index]-1e-9:
            stamp=float(targets[index]);end=round(stamp*1e6);q=buffer[(buffer['t']>=end-20000)&(buffer['t']<end)]
            counts=np.bincount(q['y'].astype(np.int64)*1280+q['x'],minlength=1280*720).reshape(720,1280)
            b=outline_box(counts,8);w=outline_box(counts,4)
            records[f'{stamp:.1f}']=dict(t=stamp,state='present' if b is not None else 'absent',box=b,weaker_outline_box=w,
                threshold_center_disagreement_px=None if b is None or w is None else float(np.linalg.norm((np.array(b[:2])+b[2:]-np.array(w[:2])-w[2:])/2)),
                forecast_outputs_used=False,reference_is_independent_human_GT=False,assistant_visual_review_pending=True)
            image=np.uint8(255-np.clip(counts*20,0,255));cv2.imwrite(str(ref/f'{stamp:.1f}.png'),image)
            if index%5==0:previews.append((stamp,image.copy()))
            index+=1
        buffer=buffer[buffer['t']>=round(t*1e6)-250000]
    assert index==len(targets),(index,len(targets))
    (ref/'references_pending_review.json').write_text(json.dumps(dict(records=records,forecast_outputs_used=False),indent=2),'utf8')
    from PIL import Image,ImageDraw
    panel=Image.new('RGB',(1280,3*385),'white');draw=ImageDraw.Draw(panel)
    for i,(stamp,a) in enumerate(previews[:6]):
        x=(i%2)*640;y=(i//2)*385;panel.paste(Image.fromarray(a).resize((640,360)),(x,y+25));draw.text((x+5,y+4),f'{stamp:.1f}s native ON/OFF density; no forecast overlay',fill='black')
    panel.save(ref/'blind_contact.png')
    print('References extracted, visual review required; no scores yet.',flush=True)
