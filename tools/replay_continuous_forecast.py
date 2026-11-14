"""Causal saved-observation comparison; no RAW/GPU or labels in inference."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from continuous_five_circle import ContinuousFiveCircle
from probe_continuous_structure import geometry
from research.stage146.branches146 import BranchGeometry,plane_axis
from live_structure_bridge import QualifiedCenter
from run_fast_structure import PhysicalPositionState
from fast_force_response import FastForceResponse
from summarize_live_pilot import reference_at,stats


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);results={}
    for name in ['clear_release_v2','side_release']:
        source=ROOT.parent/'output/five_circle_fast30_20261026'/name
        rows=[json.loads(s) for s in (source/'predictions.jsonl').read_text('utf-8').splitlines()]
        model=ContinuousFiveCircle();bg=BranchGeometry()
        center={n:QualifiedCenter() for n in ['original','continuous']}
        state={n:PhysicalPositionState() for n in center};force={n:FastForceResponse() for n in center}
        ref={n:None for n in center};previous={n:None for n in center};output=[];exact=True
        for r in rows:
            b=r['body'];rotors=r['geometry'].get('rotors',[])
            g=model.step(r['t'],b,geometry(copy.deepcopy(rotors),r['t'],b,bg))
            qualified=[z for z in rotors if z.get('mode')=='ellipse' or z.get('curvature_validated') or z.get('arc_center_qualified')]
            visible=False
            if g.get('axis_xy') is None and len(qualified)>=2 and np.median([z['ratio'] for z in qualified])>.16:
                axis,agreement=plane_axis(qualified)
                if agreement>=.85:g.update(axis_xy=axis.tolist(),geometric_weight=.35);visible=True
            side=r['geometry'].get('side',{})
            if g['current_geometric_center'] is None and not visible and side.get('valid') and side.get('confidence',0)>=.45:
                mode='edge_on' if qualified and np.median([z['ratio'] for z in qualified])<=.16 else 'axis_only'
                if g['mode']!='partial_two':g['mode']=mode;model.core.mode=mode
                g.update(axis_xy=side['normal_axis_xy'],geometric_weight=.35)
                if g['mode']!='partial_two':g=model.observe_axis(g,side['normal_axis_xy'],side['confidence'],r['t'])
            answers={}
            for n,s in [('original',r['structure']),('continuous',g)]:
                adjusted,ci=center[n].step(b,s,rotors);state[n].kf.noise_scale=ci.get('noise_scale',1.)
                c,valid,_=state[n].step(r['t'],adjusted);u=None
                f2=r.get('squared_frequency_proxy');quality=0.
                if f2 is not None:
                    f=np.sqrt(f2);past=previous[n]
                    coherent=past is not None and r['t']-past[0]<.12 and abs(np.log(f/past[1]))<.08
                    previous[n]=(r['t'],f)
                    if coherent and s.get('axis_xy') is not None:
                        if ref[n] is None:ref[n]=f
                        q=(f/ref[n])**2;normal=np.asarray(s['axis_xy'],float)
                        if normal[1]>0:normal=-normal
                        if .2<=q<=5.:u=q*normal;quality=max(.35,s.get('geometric_weight',0.))
                pred,fd=force[n].step(r['t'],c,b.get('center') if b.get('detected') else None,u,quality)
                answers[n]=dict(structure=s,coefficient=c,valid=valid,physical=pred,force=fd,center_bridge=ci)
                if n=='original' and c is not None:
                    exact=exact and all(np.allclose(c[k],r['coefficient'][k],atol=1e-8,rtol=0) for k in ['position','velocity'])
                    exact=exact and all(np.allclose(pred[h],r['physical'][h],atol=1e-8,rtol=0) for h in pred)
            output.append(dict(t=r['t'],body=b,results=answers))
        assert exact,'Original causal reconstruction differs; do not score'
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(z,default=lambda v:v.tolist() if hasattr(v,'tolist') else v) for z in output),'utf-8')
        scores={}
        if name=='side_release':
            labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
            truth=sorted(json.loads(labels.read_text('utf-8'))['records'].values(),key=lambda z:z['t'])
            for H in [.1,.2,.4]:
                errors={n:[] for n in center};last=None
                for r in output:
                    slot=int(np.floor(r['t']*5+1e-8))
                    if slot==last or not all(a['valid'] for a in r['results'].values()):continue
                    last=slot;p=reference_at(truth,r['t']+H)
                    if p is None:continue
                    for n in center:errors[n].append(float(np.linalg.norm(np.asarray(r['results'][n]['physical'][str(H)])-p)))
                scores[str(H)]={n:stats(e) for n,e in errors.items()}
        results[name]=dict(original_forecasts_reconstructed_exactly=exact,scores=scores,
            original_CI=sum(z['results']['original']['center_bridge']['applied'] for z in output),
            continuous_CI=sum(z['results']['continuous']['center_bridge']['applied'] for z in output),
            partial_CI=sum(z['results']['continuous']['center_bridge'].get('reason')=='prior_conditioned_partial_center_CI' for z in output))
    (out/'summary.json').write_text(json.dumps(dict(results=results,new_RAW=False,labels_after_inference=True),indent=2),'utf-8');print(json.dumps(results))


if __name__=='__main__':run(sys.argv[1])
