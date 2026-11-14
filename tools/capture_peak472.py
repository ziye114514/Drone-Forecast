"""Two explicit peak-geometry mechanisms, six RAWs, fixed scoring cohort."""
from pathlib import Path
import os,subprocess,sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mean-only',action='store_true');parser.add_argument('--full-neighborhood',action='store_true');parser.add_argument('--multichannel',action='store_true');parser.add_argument('--symmetric',action='store_true');parser.add_argument('--affine-hull',action='store_true');parser.add_argument('--timestamped',action='store_true');parser.add_argument('--joint-cost',action='store_true');parser.add_argument('--complete-cost',action='store_true');parser.add_argument('--local-contrast',action='store_true');parser.add_argument('--contrast-symmetric',action='store_true');parser.add_argument('--contrast-plane',action='store_true');args=parser.parse_args()
    if args.contrast_plane:out=ROOT.parent/'output/raw_contrast_plane492_20261112'
    elif args.contrast_symmetric:out=ROOT.parent/'output/raw_contrast_symmetric488_20261112'
    elif args.local_contrast:out=ROOT.parent/'output/raw_local_contrast487_20261112'
    else:out=ROOT.parent/('output/raw_complete_cost485_20261112' if args.complete_cost else 'output/raw_joint_cost484_20261112' if args.joint_cost else 'output/raw_timestamped482_20261112' if args.timestamped else 'output/raw_supported_plane480_20261112' if args.affine_hull else 'output/raw_symmetric478_20261112' if args.symmetric else 'output/raw_multichannel475_20261111' if args.multichannel else 'output/raw_full_joint474_20261111' if args.full_neighborhood else 'output/raw_mean473_20261111' if args.mean_only else 'output/raw_peak472_20261111')
    out.mkdir(exist_ok=False)
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCE_FILES}
    arms={'complete_cost':'run_rotor_complete_surface_candidate.py'} if args.complete_cost else {'joint_cost':'run_rotor_symmetric_surface_candidate.py'} if args.joint_cost else {'timestamped':'run_rotor_timestamped_candidate.py'} if args.timestamped else {'supported_plane':'run_rotor_supported_plane_candidate.py'} if args.affine_hull else {'symmetric':'run_rotor_symmetric_candidate.py'} if args.symmetric else {'multichannel':'run_rotor_multichannel_candidate.py'} if args.multichannel else {'full_joint':'run_rotor_full_joint_candidate.py'} if args.full_neighborhood else {'joint_only':'run_rotor_joint_only_candidate.py'} if args.mean_only else {'aperture':'run_rotor_aperture_candidate.py','joint_peak':'run_rotor_joint_peak_candidate.py'}
    if args.local_contrast:arms={'local_contrast':'run_rotor_local_contrast_candidate.py'}
    if args.contrast_symmetric:arms={'contrast_symmetric':'run_rotor_contrast_symmetric_candidate.py'}
    if args.contrast_plane:arms={'contrast_plane':'run_rotor_contrast_plane_candidate.py'}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,arms=arms,
        old_four_px2_best_direction_floor=True,no_fit_or_noise_search=True,
        mechanism_switches=vars(args),all6_uniform=True,
        all_inference_before_references=True,original102_fixed=True,new_video_unread=True,FPS_work=False),indent=2),'utf8')
    ns=('13','15','16','11','3','4');progress={}
    for n in ns:
        for arm,entry in arms.items():
            folder=out/f'{n}_{arm}'
            command=[sys.executable,str(ROOT/entry),'--raw',f'E:/dataset/experiment/{n}/media/event.raw','--start','10','--duration','6','--output',str(folder),
                '--gpu-python','C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe','--rotor-native-geometry-dll',str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')]
            print(n,arm,'RAW beginning',flush=True)
            with (out/f'run{n}_{arm}.log').open('w',encoding='utf8') as log:result=subprocess.run(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode==0,(n,arm,result.returncode)
            progress[f'{n}_{arm}']=json.loads((folder/'summary.json').read_text('utf8'));(out/'progress.json').write_text(json.dumps(progress,indent=2),'utf8');print(n,arm,'inference frozen',flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('polarity470',*arms)};missing={a:[] for a in pool};evidence={}
    for n in ns:
        baseline=ROOT.parent/f'output/raw_polarity470_20261111/recording{n}/predictions.jsonl';old=[json.loads(s) for s in baseline.read_text('utf8').splitlines()]
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={};evidence[n]={}
        for a,p in [('polarity470',baseline),*[(a,out/f'{n}_{a}/predictions.jsonl') for a in arms]]:
            rows=[json.loads(s) for s in p.read_text('utf8').splitlines()];assert len(rows)==len(old)==181 and all(r['t']==q['t'] and r['counts']['events']==q['counts']['events'] for r,q in zip(rows,old))
            if a=='aperture':
                assert all(r['body']==q['body'] for r,q in zip(rows,old))
                assert all(all((r['motion'].get('estimate') or {}).get(k)==(q['motion'].get('estimate') or {}).get(k) for k in ('dx','dy')) for r,q in zip(rows,old))
            count=sum(bool((r['motion'].get('estimate') or {}).get('native_peak_shape',{}).get('available')) for r in rows);evidence[n][a]=dict(peak_shape_frames=count)
            s=score(p,ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
            missing[a].extend(dict(recording=n,t=t) for t in ids if not any(q['valid'] and q['t']==t for q in s['records']))
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within31=sum(x<=31 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,missing_original=missing,evidence=evidence,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,'missing',missing,flush=True)
