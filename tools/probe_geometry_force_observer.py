"""Rank-aware two-force prototype on fixed current observations."""
from pathlib import Path
import collections,json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from geometry_force_observer import GeometryForceObserver,projected_fit
from compare_final_pipeline import score


def checks():
    d=np.array([12.,1000.]);rows=[]
    for t in np.linspace(0,.4,13):
        theta=np.radians(-35+70*t/.4);n=np.array([np.sin(theta),-np.cos(theta)])
        amplitude=800+100*np.sin(10*t)
        rows.append(dict(t=t,n=n,a=amplitude*n+d,se=np.array([2.,2.]),angle_sigma=np.radians(1.)))
    value,C,info=projected_fit(rows)
    assert value is not None and np.linalg.norm(value-d)<1e-6 and info['data_information_rank']==2
    straight=[dict(r,n=np.array([0.,-1.])) for r in rows]
    value,C,info=projected_fit(straight);assert value is None and info['data_information_rank']==1
    signed=[dict(r,n=r['n']*(-1 if i%2 else 1)) for i,r in enumerate(rows)]
    value,C,info=projected_fit(signed);assert np.linalg.norm(value-d)<1e-6
    return dict(controlled_rank_two_recovers_image_disturbance=True,
        parallel_planes_do_not_invent_full_force=True,headless_normal_sign_invariant=True,
        metric_or_real_event_force_accuracy_proved=False)


def main():
    irregular='--irregular' in sys.argv
    out=ROOT.parent/'output/fred_transfer_20261029'/('geometry_force_observer_irregular' if irregular else 'geometry_force_observer');out.mkdir(exist_ok=False)
    base=ROOT.parent/'output/fred_transfer_20261029/motion_observability_v3'
    paths={name:base/f'{name}_motion.jsonl' for name in ('side','other','fred0','fred1','fred2')}
    paths['clear']=ROOT.parent/'output/fred_transfer_20261029/RAW_clear_integrated2s/predictions.jsonl'
    summaries={};frozen={}
    for name,path in paths.items():
        original=[json.loads(x) for x in path.read_text().splitlines()];model=GeometryForceObserver(irregular=irregular);modified=[]
        for r in original:
            p,d=model.step(r['t'],r['coefficient'],r['physical'],r['force'],r['structure'],r['image_motion'],
                r.get('normal_turn_forecast'),r.get('adaptive_force_memory'))
            modified.append(dict(r,physical=p,geometry_force_observer=d))
        (out/f'{name}.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in modified)+'\n','utf-8')
        summaries[name]=dict(qualified=sum(r['geometry_force_observer']['qualified'] for r in modified),
            applied=sum(r['geometry_force_observer']['applied'] for r in modified),
            reasons=dict(collections.Counter(r['geometry_force_observer']['reason'] for r in modified)))
        frozen[name]=(original,modified)
    for name in ('side','other'):
        label=ROOT.parent/('output/dense_labels_side_110_1195/labels.json' if name=='side' else 'output/dense_labels_train2_40_46/labels.json')
        truths=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        summaries[name]['error']={k:score(rows,truths)['physical_future_error_px']['0.4'] for k,rows in zip(('control','geometry_inferred'),frozen[name])}
    result=dict(controlled_checks=checks(),sequences=summaries,default_not_enabled=True,
        current_motion_not_independent_true_force_truth=True,annotations_only_used_after_freezing=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
