"""Bounded real-candidate missing-observation consistency test.

Current withheld hub is used only as the scoring reference. Past full layouts
initialize the existing model; no current full center or NN body enters it.
This is controlled missing geometry, NOT real occlusion or COM ground truth.
"""
from pathlib import Path
import sys,json,copy,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from five_circle_model import FiveCircleModel,diagonal_center
from rotor_evidence import RotorEvidence,VisibilityLayout
OUT=ROOT.parent/'output/decision169_20261105'


def full_record(row):
    rotors=row['geometry'].get('rotors',[])
    if len(rotors)!=4 or any(r.get('fov_clipped') or not r.get('current_observation',True) or r.get('ratio',0)<=.16 or not(r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated')) for r in rotors):return None
    p=np.array([[r['cx'],r['cy']] for r in rotors]);center=diagonal_center(p)
    if center is None or not np.isfinite(center).all():return None
    return dict(t=row['t'],rotors=rotors,center=center)


def geometry(rotors,full):
    return dict(observed_rotors=rotors,state='four' if full else 'insufficient',confidence=.8 if full else .6,axis_xy=[0.,-1.])


if __name__=='__main__':
    source=ROOT.parent/'output/nn_rotor166_20261105/clear_observations.jsonl'
    rows=list(map(json.loads,source.read_text('utf-8').splitlines()));eligible=[a for r in rows if (a:=full_record(r)) is not None]
    output=[];history=[];skipped=[]
    for current in eligible:
        past=[r for r in history if current['t']-r['t']<=.12+1e-8]
        if len(past)<2:
            skipped.append(dict(t=current['t'],reason='need_two_recent_complete_observations'));history.append(current);continue
        first,last=past[-2:];velocity=(last['center']-first['center'])/(last['t']-first['t'])
        prior_center=last['center']+(current['t']-last['t'])*velocity
        for hidden in range(4):
            model=FiveCircleModel(3)
            api_evidence=RotorEvidence();api_layout=VisibilityLayout()
            for seed in [first,last]:
                model.step(seed['t'],dict(detected=True,center=seed['center'].tolist()),geometry(copy.deepcopy(seed['rotors']),True))
                packet=api_evidence.step(seed['t'],dict(detected=True,center=seed['center'].tolist()),dict(mode='unresolved'),seed['rotors'],[])
                api_layout.step(packet)
            visible=[r for j,r in enumerate(current['rotors']) if j!=hidden]
            expected=model._expected(prior_center).copy()
            out=model.step(current['t'],dict(detected=True,center=prior_center.tolist()),geometry(copy.deepcopy(visible),False))
            packet=api_evidence.step(current['t'],dict(detected=True,center=prior_center.tolist()),dict(mode='unresolved'),visible,[])
            api=api_layout.step(packet)
            baseline=np.mean([[r['cx'],r['cy']] for r in visible],axis=0)
            prediction=out.get('current_geometric_center');inferred=out.get('inferred_model_ids',[])
            record=dict(t=current['t'],hidden_index_in_reference=hidden,visible_centers=[[r['cx'],r['cy']] for r in visible],
                past_only_prior_center=prior_center.tolist(),full_reference_center=current['center'].tolist(),
                withheld_reference_hub=[current['rotors'][hidden]['cx'],current['rotors'][hidden]['cy']],
                independent_centroid=baseline.tolist(),model_center=prediction,model_mode=out['mode'],
                current_full_center_used_by_model=False,NN_body_used_by_model=False,
                independent_centroid_error_px=float(np.linalg.norm(baseline-current['center'])),
                history_only_error_px=float(np.linalg.norm(prior_center-current['center'])),
                model_center_error_px=None if prediction is None else float(np.linalg.norm(np.asarray(prediction)-current['center'])),
                api_model=api['model'],api_center=api['current_center'],
                api_center_error_px=None if api['current_center'] is None else float(np.linalg.norm(np.asarray(api['current_center'])-current['center'])),
                inferred_hub_error_px=None,hidden_hub_used_by_model=False)
            if prediction is not None and len(inferred)==1:
                ids=out['observed_model_ids'];delta=np.mean(np.asarray(record['visible_centers'])-expected[ids],axis=0)
                translation=prior_center+delta
                record['shared_translation_center']=translation.tolist()
                record['shared_translation_error_px']=float(np.linalg.norm(translation-current['center']))
                q=np.asarray(out['model_centers'])[inferred[0]]
                record['inferred_hub']=q.tolist();record['inferred_hub_error_px']=float(np.linalg.norm(q-record['withheld_reference_hub']))
            output.append(record)
        history.append(current)
    common=[r for r in output if r['model_center_error_px'] is not None]
    def stats(values):
        return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95))) if values else dict(n=0,mean=None,median=None,p95=None)
    summary=dict(reference='same-current full detected quad, not COM or independent human ground truth',
        test_type='controlled missing geometry; real video candidates; no pixel occlusion synthesis',
        full_current_center_used_for_scoring_only=True,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        eligible_complete_frames=len(eligible),query_frames=len(output)//4,skipped_frames=skipped,
        planned_mask_cases=len(output),model_outputs=len(common),
        baseline_all=stats([r['independent_centroid_error_px'] for r in output]),
        baseline_common=stats([r['independent_centroid_error_px'] for r in common]),
        model_common=stats([r['model_center_error_px'] for r in common]),
        history_only_common=stats([r['history_only_error_px'] for r in common]),
        shared_translation_common=stats([r['shared_translation_error_px'] for r in common if 'shared_translation_error_px' in r]),
        api_outputs=sum(r['api_center'] is not None for r in output),
        api_geometry_consistency=stats([r['api_center_error_px'] for r in output if r['api_center_error_px'] is not None]),
        withheld_hub_common=stats([r['inferred_hub_error_px'] for r in common if r['inferred_hub_error_px'] is not None]),
        parameter_search=False,original_model_reused=True,true_detection_or_pose_accuracy_measured=False)
    (OUT/'layout_cases.json').write_text(json.dumps(output,indent=2),'utf-8');(OUT/'layout_summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    print(json.dumps(summary),flush=True)
