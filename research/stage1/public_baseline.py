"""Stage2 public-data feasibility: frozen RAW-only runtime and annotation audit.

This is a TRAIN-sequence diagnostic, not an official FRED test result. Labels
are read for evaluation only after inference finishes. No new method is added.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
import cv2,numpy as np
from run_study import OUT,ROOT,save,csvsave
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.motion import extrapolate,KinematicFilter
from event_drone_forecast.config import Config
from event_drone_forecast.raw_io import open_raw,raw_chunks

def main():
    assert (OUT/'stage1_completion.json').exists(),'Finish Stage1 before Stage2 inference'
    dest=OUT/'public_truth_substitutes';folder=dest/'fred_sequence1';run=folder/'raw_only_baseline'
    freeze=dest/'public_protocol.json'
    if not freeze.exists():
        save(freeze,dict(stage1_complete_before_public_inference=True,sequence='1',official_split='train',start_s=0,duration_s=20,
            selection='Existing local sequence1 chosen for format/transfer feasibility, not prediction error.',
            inference_inputs=[str(folder/'events.raw')],estimator='Unchanged v0.3 run_paper.py',
            horizons_s=[.1,.2,.4],scoring='Actual provided coordinates timestamps; no interpolated GT or GT input to raw estimator.',
            additional_oracle='CV with past annotation centers, explicitly separate and not RAW-only.',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            no_official_benchmark_claim=True))
        (dest/'public_baseline_source.py').write_bytes(Path(__file__).read_bytes())
    if not (run/'summary.json').exists():
        subprocess.run([sys.executable,str(ROOT/'core/run_paper.py'),'--raw',str(folder/'events.raw'),
                        '--output',str(run),'--duration','20'],check=True)
    meta=json.loads((run/'run_manifest.json').read_text());assert meta['inference_external_inputs']==[str((folder/'events.raw').resolve())]
    # Evaluation-only process begins here.
    labels=[]
    for line in (folder/'coordinates.txt').read_text().splitlines():
        stamp,values=line.split(':',1);fields=[f.strip() for f in values.split(',')]
        box=np.array(list(map(float,fields[:4])))
        labels.append(dict(t=float(stamp),bbox=box.tolist(),center=((box[:2]+box[2:])/2).tolist()))
    traces=[json.loads(s) for s in (run/'forecasts.jsonl').read_text().splitlines()];times=np.array([r['t'] for r in traces])
    methods=['cv','paper_nominal','paper_adaptive','paper_fused'];details=[]
    for label in labels:
        if not 0<=label['t']<=20:continue
        for h in (.1,.2,.4):
            k=np.searchsorted(times,label['t']-h,side='right')-1
            if k<0:continue
            q=traces[k]
            for method in methods:
                row=dict(protocol='RAW-only automatic detector',target_time_s=label['t'],anchor_time_s=float(times[k]),
                    horizon_s=h,method=method,available=q['active'])
                if q['active']:
                    p=extrapolate(q['states'][method],label['t']-times[k]);row['error_px']=float(np.linalg.norm(p-label['center']))
                details.append(row)
    # Annotation-only CV: validates that this archive can support 2D endpoint
    # metrics. It is not an alternative truth for the automatic predictions.
    oracle=[];cv=KinematicFilter(Config(),2);last=None
    for label in sorted(labels,key=lambda r:r['t']):
        t=label['t']
        if t>20:break
        if last is not None and t-last>.1:cv=KinematicFilter(Config(),2)
        state=cv.update(t,label['center'],1.);oracle.append(dict(t=t,state=state));last=t
    otimes=np.array([r['t'] for r in oracle])
    for label in labels:
        if label['t']>20:continue
        for h in (.1,.2,.4):
            k=np.searchsorted(otimes,label['t']-h,side='right')-1
            if k<0:continue
            anchor=oracle[k];future=label['t']-anchor['t'];available=future<=h+.05
            row=dict(protocol='Past-annotation CV, NOT RAW-only',target_time_s=label['t'],anchor_time_s=anchor['t'],horizon_s=h,
                     method='oracle_past_boxes_cv',available=available)
            if available:
                prediction=anchor['state'][:2]+anchor['state'][2:]*future;row['error_px']=float(np.linalg.norm(prediction-label['center']))
            details.append(row)
    csvsave(dest/'public_endpoint_details.csv',details);metrics=[]
    for method in methods+['oracle_past_boxes_cv']:
        for h in (.1,.2,.4):
            subset=[r for r in details if r['method']==method and r['horizon_s']==h];ok=[r for r in subset if r['available']]
            metrics.append(dict(method=method,horizon_s=h,targets=len(subset),available=len(ok),coverage=len(ok)/max(1,len(subset)),
                mean_error_px=float(np.mean([r['error_px'] for r in ok])) if ok else None))
    # Visual alignment audit, with original author-rendered event images and
    # independently decoded RAW at the same indicated timestamp.
    panels=[];audit=[]
    with zipfile.ZipFile(r'E:\FRED\1.zip') as archive:
        frames=[p for p in archive.namelist() if p.startswith('Event/Frames/') and p.endswith('.png')]
        for desired in (5.1,10.,15.,18.):
            label=min(labels,key=lambda r:abs(r['t']-desired));stamp=label['t']
            member=min(frames,key=lambda p:abs(int(Path(p).stem.split('_')[-1])/1e6-stamp))
            provided=cv2.imdecode(np.frombuffer(archive.read(member),np.uint8),cv2.IMREAD_COLOR)
            reader=open_raw(folder/'events.raw');height,width=reader.get_size();batch=[]
            for t,e in raw_chunks(reader,max(0,stamp-.02),.02,10000):batch.append(e)
            e=np.concatenate(batch);pixels=e['y'].astype(int)*width+e['x'];density=np.bincount(pixels,minlength=width*height).reshape(height,width)
            decoded=cv2.cvtColor(np.uint8(np.clip(np.log1p(density)/4,0,1)*255),cv2.COLOR_GRAY2BGR)
            box=np.int32(label['bbox']);images=[]
            for name,im in [('Author event PNG',provided),('Decoded RAW 20ms',decoded)]:
                cv2.rectangle(im,tuple(box[:2]),tuple(box[2:]),(30,230,50),2)
                cv2.putText(im,f'{name} t={stamp:.6f}s | provided box',(15,30),0,.7,(30,230,50),2)
                images.append(cv2.resize(im,(768,432)))
            panels.append(np.hstack(images));x0,y0,x1,y1=box
            audit.append(dict(t=stamp,png=member,raw_dimensions=[width,height],png_dimensions=[provided.shape[1],provided.shape[0]],
                events_in_box=int(density[max(0,y0):min(height,y1),max(0,x0):min(width,x1)].sum()),total_events=len(e),
                note='Overlay requires visual audit; do not optimize temporal offsets from forecast errors.'))
    cv2.imwrite(str(dest/'public_alignment_audit.png'),np.vstack(panels))
    result=dict(stage='2: public-data feasibility and baseline only',sequence='FRED train1, first20s',
        automatic_run=json.loads((run/'summary.json').read_text()),metrics=metrics,alignment_audit=audit,
        label_note='Local archive legacy coordinates contain four box values without explicitID/class; no official canonical forecasting score claimed.',
        labels_used_in_raw_inference=False,independent_rpm_or_pose_truth=False,
        limitations=['Single public train sequence and short interval, not a full benchmark or unseen test.',
            'Past-annotation CV is an explicitly oracle-position input protocol, not deployment output.',
            'Bounding-box labels do not verify rotor RPM, thrust or attitude.'])
    save(dest/'public_baseline_summary.json',result);print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
