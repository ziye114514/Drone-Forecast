"""Frozen local-unused FRED11 validation; fixed10--16s, no test tuning.

The two source-frozen algorithms run before reading full official references
or displaying frames. Only RAW header and first annotation metadata were seen
before this protocol. No NN, training or recording-specific inference rules.
"""
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile,shutil,re
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES

if __name__=='__main__':
    import numpy as np
    base=ROOT.parent;gate=base/'output/raw_bg_memory517_20261113'
    checked=json.loads((gate/'results.json').read_text('utf8'))
    assert checked['all6_actual_RAW'] and checked['pooled']['background']['n']==102 and checked['pooled']['background']['mean']<=30
    prior=json.loads((gate/'protocol.json').read_text('utf8'))['source_sha256']
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in prior.items())
    out=base/'output/frozen_fred522_20261113';out.mkdir(exist_ok=True)
    assert not any(p.is_file() for p in out.rglob('*')), 'Never repeat an external test with a saved protocol or inputs'
    archive=Path('E:/FRED/11.zip');raw=out/'input/events.raw';raw.parent.mkdir(exist_ok=True)
    arms={'baseline487':'run_rotor_local_contrast_candidate.py','candidate517':'run_rotor_background_memory_candidate.py'}
    files=sorted(set(SOURCE_FILES+['tools/frozen_fred522.py','tools/score_rotor164.py','tools/compare_final_pipeline.py']))
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    with zipfile.ZipFile(archive) as z:
        entry=z.getinfo('Event/events.raw')
        protocol=dict(archive=str(archive),archive_bytes=archive.stat().st_size,raw_member=entry.filename,
            raw_bytes=entry.file_size,raw_zip_CRC=entry.CRC,start=10,duration=6,arms=arms,source_sha256=hashes,
            prior_metadata_search='521 plus full local text search: no prior FRED11 use found; not absolute proof.',
            train1_to6_already_historically_exposed=True,gate_actual_mean_px=checked['pooled']['background']['mean'],
            both_inference_streams_before_full_official_reference_or_frame_review=True,
            fixed_origins='10.8:0.2:15.6',no_reselection_of_source_or_interval=True,no_test_tuning=True,
            primary_metric='0.4s source-time mean FDE on available official-positive queries; missing reported separately',
            predeclared_provisional_pass='mean<=30px AND available>=90% of fixed official-positive queries; native-frame alignment review required',
            reference_is_official_bbox_not_COM_or_rotor_only_outline=True,original102_kept_separate=True,FPS_work=False)
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
        with z.open(entry) as src,raw.open('wb') as dst:shutil.copyfileobj(src,dst)
    assert raw.stat().st_size==entry.file_size
    for arm,entry_name in arms.items():
        print(arm,'FRED11 fixed RAW beginning',flush=True)
        cmd=[sys.executable,str(ROOT/entry_name),'--raw',str(raw),'--start','10','--duration','6','--output',str(out/arm),
            '--gpu-python','C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',
            '--rotor-native-geometry-dll',str(base/'output/uncertainty216_20261108/geometry_kernels.dll')]
        with (out/f'{arm}.log').open('w',encoding='utf8') as log:rc=subprocess.run(cmd,cwd=base,stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0,(arm,rc)
        s=json.loads((out/arm/'summary.json').read_text('utf8'))
        assert s['frames']==s['native_packets_read']==181 and s['skipped_observations']==0
        print(arm,'predictions frozen',flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (out/'both_predictions_frozen.json').write_text(json.dumps(dict(source_sha256=hashes),indent=2),'utf8')
    with zipfile.ZipFile(archive) as z:text=z.read('interpolated_coordinates.txt').decode('utf8')
    (out/'input/official_interpolated_coordinates.txt').write_text(text,'utf8');records={}
    for line in text.splitlines():
        if not line.strip():continue
        stamp,value=line.split(':',1);t=float(stamp);a=[float(v.strip()) for v in value.split(',')]
        if 9.7<=t<=16.3:
            assert len(a)>=4 and np.isfinite(a[:4]).all()
            records[str(t)]=dict(t=t,state='present',box=a[:4],reference='official_FRED_interpolated_event_bbox',
                object_id=a[4] if len(a)>4 else None,forecast_used=False)
    ref=out/'official_references.json';ref.write_text(json.dumps(dict(records=records,
        official_reference_not_COM_or_rotor_outline=True,full_annotations_read_after_predictions=True),indent=2),'utf8')
    from score_rotor164 import score
    from compare_final_pipeline import reference_at
    rr=sorted(records.values(),key=lambda r:r['t']);origins=np.round(np.arange(10.8,15.6001,.2),1)
    positive=[float(t) for t in origins if reference_at(rr,t+.4) is not None]
    cases={arm:score(out/arm/'predictions.jsonl',ref,origins) for arm in arms}
    for c in cases.values():
        c.update(manual_boxes_are_development_references=False,new_temporal_extension_is_not_a_new_recording=False,
            official_FRED_new_local_source=True)
        c['fixed_positive_queries']=len(positive);c['positive_coverage']=c['available']/max(1,len(positive))
        c['provisional_pass_pending_native_review']=bool(c['error'].get('mean') is not None and c['error']['mean']<=30 and c['positive_coverage']>=.9)
    result=dict(cases=cases,fixed25_queries=True,positive_queries=len(positive),source_unchanged=True,
        no_test_tuning=True,official_reference_alignment_review_pending=True,original102_separate=True,FPS_work=False)
    (out/'results_pending_review.json').write_text(json.dumps(result,indent=2),'utf8')
    print('FRED11 raw official scores (alignment review pending):',json.dumps({a:{k:c[k] for k in ('available','error','positive_coverage')} for a,c in cases.items()}),flush=True)
    from live_decode import native_batches
    import cv2
    from PIL import Image,ImageDraw
    targets=np.round(np.arange(10.8,16.0001,.2),1);index=0;buffer=None;images=[];p=out/'native_review';p.mkdir()
    for t,e in native_batches(raw,10,6,False,33333,reader_backend='cooperative'):
        buffer=e.copy() if buffer is None else np.concatenate([buffer,e])
        while index<len(targets) and t>=targets[index]-1e-9:
            stamp=float(targets[index]);end=round(stamp*1e6);q=buffer[(buffer['t']>=end-20000)&(buffer['t']<end)]
            counts=np.bincount(q['y'].astype(np.int64)*1280+q['x'],minlength=1280*720).reshape(720,1280)
            im=np.uint8(255-np.clip(counts*20,0,255));cv2.imwrite(str(p/f'{stamp:.1f}.png'),im)
            label=reference_at(rr,stamp);images.append((stamp,im.copy(),label))
            index+=1
        buffer=buffer[buffer['t']>=round(t*1e6)-250000]
    assert index==27
    for page,start in enumerate(range(0,len(images),8)):
        panel=Image.new('RGB',(1280,1540),'white');draw=ImageDraw.Draw(panel)
        for i,(t,im,label) in enumerate(images[start:start+8]):
            x=i%2*640;y=i//2*385;panel.paste(Image.fromarray(im).resize((640,360)),(x,y+25))
            draw.text((x+5,y+4),f'{t:.1f}s native20ms; no forecast overlay',fill='black')
        panel.save(p/f'page{page}.png')
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    print('27 native frames ready for alignment review. No algorithm changes permitted from this test.',flush=True)
