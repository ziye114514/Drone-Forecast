"""Existing background matcher on exact487 packet intervals, no reference read."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np,cv2,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from background_unknown_response import BackgroundMotion

SOURCES={'13':'phase185_20261107/cache_recording13','15':'motion181_20261107/cache_recording15',
    '16':'phase185_20261107/cache_recording16','11':'phase185_20261107/cache_recording11',
    '3':'frozen_tangent287_20261108/cache','4':'frozen324_20261109/cache'}

def packets(cache,rows):
    manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames'];index=0;prior=10000000
    for r in rows:
        end=round(r['t']*1e6);parts=[]
        while index<len(manifest) and round(manifest[index]['t']*1e6)<=prior:index+=1
        j=index
        while j<len(manifest):
            old_begin=10000000 if j==0 else round(manifest[j-1]['t']*1e6)
            if old_begin>=end:break
            a=np.load(cache/manifest[j]['file'],mmap_mode='r')
            lo=int(np.searchsorted(a['t'],prior));hi=int(np.searchsorted(a['t'],end))
            if hi>lo:parts.append(a[lo:hi])
            j+=1
        if not parts:raise ValueError(('No cached events',r['t']))
        n=sum(map(len,parts));assert n==r['counts']['events'],(r['t'],n,r['counts']['events'])
        count=np.zeros(1280*720,np.float32)
        for a in parts:count+=np.bincount(a['y'].astype(np.int64)*1280+a['x'],minlength=1280*720)
        yield count.reshape(720,1280);prior=end

def evidence(model,row,counts):
    body=row['body'];point=body.get('center')
    proxy=dict(t=row['t'],body=body,coefficient=None if point is None else dict(position=point))
    info=model.step(proxy,(counts,np.zeros_like(counts)));item=dict(t=row['t'],proof=info,pair=None)
    if model.transforms and abs(model.transforms[-1]['t']-row['t'])<1e-9:
        q=model.transforms[-1];M=np.asarray(q['M']);point=np.asarray(point if point is not None else [640.,360.],float)
        delta=point-np.linalg.solve(M[:,:2],point-M[:,2])
        item['pair']=dict(source_t=q['source'],t=q['t'],displacement_xy=delta.tolist(),
            covariance_px2=q['Q'].tolist(),affine=M.tolist(),local_background_point=point.tolist())
    return item

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--reference-support',action='store_true');parser.add_argument('--translation-only',action='store_true');args=parser.parse_args()
    out=ROOT.parent/('output/background_translation503_20261112' if args.translation_only else 'output/background_supported501_20261112' if args.reference_support else 'output/background_reference497_20261112');out.mkdir(exist_ok=False);cv2.setNumThreads(2)
    files=['background_unknown_response.py','native_plane_motion.py','tools/measure_background497.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files};progress={}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,exact_actual487_intervals=True,reference_support=args.reference_support,translation_only=args.translation_only,
        no_parameter_search=True,rotor_region_excluded=True,no_annotation_access=True,
        background_flow_is_not_independent_camera_pose=True,reserved_video_not_read=True),indent=2),'utf8')
    for n,p in SOURCES.items():
        rows=[json.loads(s) for s in (ROOT.parent/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl').read_text('utf8').splitlines()]
        model=BackgroundMotion(reference_support=args.reference_support,translation_only=args.translation_only);values=[]
        for r,c in zip(rows,packets(ROOT.parent/'output'/p,rows)):values.append(evidence(model,r,c))
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(q,allow_nan=False)+'\n' for q in values),'utf8')
        progress[n]=dict(frames=len(values),background_pairs=sum(q['pair'] is not None for q in values),
            qualified_three_pair_derivatives=sum(q['proof']['qualified'] for q in values),all_native_counts_match=True)
        (out/'progress.json').write_text(json.dumps(progress,indent=2),'utf8');print(n,progress[n],flush=True)
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
