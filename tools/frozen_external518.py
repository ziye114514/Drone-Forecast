"""Gate-controlled single fixed external validation after actual mean<=30.

The reserved10--16s is used once for a frozen487 vs517 comparison. No interval,
model or coefficient may be changed after seeing its frames or its results.
This script only generates pending weak references, never auto-approves them.
"""
from pathlib import Path
import sys,os,json,hashlib,subprocess
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES


if __name__=='__main__':
    import numpy as np,cv2
    base=ROOT.parent;gate=base/'output/raw_bg_memory517_20261113'
    checked=json.loads((gate/'results.json').read_text('utf8'))
    assert checked['all6_actual_RAW'] and checked['original102_preserved'] and checked['source_unchanged_during_batch']
    assert checked['pooled']['background']['n']==102 and checked['pooled']['background']['mean']<=30.
    previous=json.loads((gate/'protocol.json').read_text('utf8'))['source_sha256']
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in previous.items())
    out=base/'output/frozen_external518_20261113';out.mkdir(exist_ok=False)
    raw=Path('E:/dataset/recordings/rec_20261004_162506/event.raw')
    arms={'baseline487':'run_rotor_local_contrast_candidate.py','candidate517':'run_rotor_background_memory_candidate.py'}
    files=sorted(set(SOURCE_FILES+['tools/frozen_external518.py','tools/reference_boxes181.py']))
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(raw=str(raw),start=10,duration=6,arms=arms,source_sha256=hashes,
        actual_development_gate=checked['pooled']['background'],gate_required_mean_px=30.,gate_passed=True,
        gate_margin_px=30.-checked['pooled']['background']['mean'],
        user_authorization='Retain blind until30px development gate, then fixed new-video validation before FPS.',
        one_fixed_validation_not_a_development_source=True,prior_only_metadata_not_frames=True,
        both_predictions_before_reference_extraction=True,fixed25_origins='10.8:0.2:15.6',
        no_reselection_of_interval=True,no_tuning_after_external_results=True,
        weak_native_outline_not_human_COM_truth=True,visual_review_required=True,FPS_work=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for arm,entry in arms.items():
        cmd=[sys.executable,str(ROOT/entry),'--raw',str(raw),'--start','10','--duration','6','--output',str(out/arm),
            '--gpu-python','C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',
            '--rotor-native-geometry-dll',str(base/'output/uncertainty216_20261108/geometry_kernels.dll')]
        print(arm,'fixed external RAW inference',flush=True)
        with (out/f'{arm}.log').open('w',encoding='utf8') as f:r=subprocess.run(cmd,cwd=base,stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0,(arm,r.returncode)
        summary=json.loads((out/arm/'summary.json').read_text('utf8'))
        assert summary['frames']==summary['native_packets_read']==181 and summary['skipped_observations']==0
        print(arm,'predictions frozen',flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (out/'both_predictions_frozen.json').write_text(json.dumps(dict(source_sha256=hashes),indent=2),'utf8')
    from live_decode import native_batches
    from reference_boxes181 import outline_box
    targets=np.round(np.arange(10.8,16.0001,.2),1);index=0;buffer=None;records={};previews=[]
    ref=out/'reference';ref.mkdir()
    for t,e in native_batches(raw,10.,6.,False,33333,reader_backend='cooperative'):
        buffer=e.copy() if buffer is None else np.concatenate([buffer,e])
        while index<len(targets) and t>=targets[index]-1e-9:
            stamp=float(targets[index]);end=round(stamp*1e6);q=buffer[(buffer['t']>=end-20000)&(buffer['t']<end)]
            counts=np.bincount(q['y'].astype(np.int64)*1280+q['x'],minlength=1280*720).reshape(720,1280)
            b=outline_box(counts,8);weak=outline_box(counts,4)
            records[f'{stamp:.1f}']=dict(t=stamp,state='present' if b is not None else 'absent',box=b,weaker_outline_box=weak,
                threshold_center_disagreement_px=None if b is None or weak is None else float(np.linalg.norm((np.array(b[:2])+b[2:]-np.array(weak[:2])-weak[2:])/2)),
                forecast_outputs_used=False,reference_is_independent_human_GT=False,assistant_visual_review_pending=True)
            image=np.uint8(255-np.clip(counts*20,0,255));cv2.imwrite(str(ref/f'{stamp:.1f}.png'),image)
            np.savez_compressed(ref/f'{stamp:.1f}_counts.npz',counts=counts)
            if index%5==0:previews.append((stamp,image.copy()))
            index+=1
        buffer=buffer[buffer['t']>=round(t*1e6)-250000]
    assert index==len(targets),(index,len(targets))
    (ref/'references_pending_review.json').write_text(json.dumps(dict(records=records,forecast_outputs_used=False,
        reference_is_weak_native_outline=True,review_required_before_scoring=True),indent=2),'utf8')
    from PIL import Image,ImageDraw
    panel=Image.new('RGB',(1280,3*385),'white');draw=ImageDraw.Draw(panel)
    for i,(stamp,a) in enumerate(previews[:6]):
        x=(i%2)*640;y=(i//2)*385;panel.paste(Image.fromarray(a).resize((640,360)),(x,y+25));draw.text((x+5,y+4),f'{stamp:.1f}s native event counts; no forecast overlay',fill='black')
    panel.save(ref/'blind_contact.png')
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    print('Frozen external predictions and pending references complete; visual review before scoring.',flush=True)
