"""Inspect saved current geometry and ROI boundaries, never future labels."""
from pathlib import Path
import argparse, copy, datetime, sys
import numpy as np
import study55 as current
ROOT=current.ROOT;sys.path.insert(0,str(ROOT/'research/stage49'))
import rigid49
OUT=current.OUT;CASES=['clear6','side6','train3'];PROTOCOL=OUT/'geometry_diagnostic_supplement.json'

def freeze():
    current.verify();assert not PROTOCOL.exists()
    inputs={f'{prefix}{case}/rotor_ticks.jsonl':current.sha(OUT/(prefix+case)/'rotor_ticks.jsonl') for case in CASES for prefix in ['off_','']}
    current.save(PROTOCOL,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),
        actual_reused49_sha256=current.sha(ROOT/'research/stage49/rigid49.py'),input_hashes=inputs,
        design='Remove only n_events when comparing current geometry semantics; inspect reference fitted ellipse perimeter against sent ROI. Reuse49 diagnostics without threshold changes for every exactly-four current candidate frame. Fixed128 perimeter samples and original49 Monte Carlo128. No labels, forecasts, new classifier or altered inference.',
        limits='Reference detections are not true rotors. Ellipse perimeter includes unobserved extrapolated arcs; boundary overlap cannot prove real rotor clipping. Convexity and assumed localization noise do not certify identity or pose.'))
    print('Saved-current geometry diagnostic frozen',flush=True)

def analyze():
    current.verify();p=current.load(PROTOCOL);assert p['helper_sha256']==current.sha(Path(__file__)) and p['actual_reused49_sha256']==current.sha(ROOT/'research/stage49/rigid49.py')
    for n,h in p['input_hashes'].items():assert current.sha(OUT/n)==h
    summary=[];details=[];rng=np.random.default_rng(49);phase=np.linspace(0,2*np.pi,128,endpoint=False)
    for case in CASES:
        off=current.rows(OUT/('off_'+case)/'rotor_ticks.jsonl');on=current.rows(OUT/case/'rotor_ticks.jsonl')
        item=dict(case=case,native10=len(on),geometry_changed_excluding_event_count=0,sensing_changed=0,roi_ticks=0,
            reference_current_candidates_under_roi=0,reference_centers_outside_roi=0,reference_ellipses_with_perimeter_outside_roi=0,
            four_to_less_under_roi=0,less_to_four_under_roi=0)
        conditions={'off':[],'body':[]};four={'off':0,'body':0};convex={'off':0,'body':0}
        for a,b in zip(off,on):
            assert a['t']==b['t'];oa=copy.deepcopy(a['observation']);ob=copy.deepcopy(b['observation']);oa.pop('n_events',None);ob.pop('n_events',None)
            item['geometry_changed_excluding_event_count']+=oa!=ob;item['sensing_changed']+=a['sensing']!=b['sensing']
            for mode,row in [('off',a),('body',b)]:
                rotors=row['observation'].get('rotors',[])
                if len(rotors)==4:
                    four[mode]+=1;d=rigid49.diagnostics([[r['cx'],r['cy']] for r in rotors],[r['center_sigma_px'] for r in rotors],rng)
                    convex[mode]+=bool(d['accepted_geometry'])
                    if d['accepted_geometry']:conditions[mode].append(d['hub_axis_ratio'])
                    details.append(dict(case=case,t=row['t'],mode=mode,rigid_geometry=d))
            diagnostic=b['roi_diagnostic']
            if diagnostic['mode']!='body_roi':continue
            item['roi_ticks']+=1;lo=np.asarray(diagnostic['roi'][:2]);hi=np.asarray(diagnostic['roi'][2:]);old=oa.get('rotors',[]);new=ob.get('rotors',[])
            item['four_to_less_under_roi']+=len(old)==4 and len(new)<4;item['less_to_four_under_roi']+=len(old)<4 and len(new)==4
            for rotor in old:
                center=np.array([rotor['cx'],rotor['cy']]);item['reference_current_candidates_under_roi']+=1
                outside=bool(np.any(center<lo) or np.any(center>=hi));item['reference_centers_outside_roi']+=outside
                if rotor.get('mode')!='ellipse':continue
                angle=np.deg2rad(rotor['angle']);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                points=np.c_[rotor['major']/2*np.cos(phase),rotor['minor']/2*np.sin(phase)]@rotation.T+center
                fraction=float(np.mean(np.any((points<lo)|(points>=hi),axis=1)));item['reference_ellipses_with_perimeter_outside_roi']+=fraction>0
                if fraction>0:details.append(dict(case=case,t=a['t'],mode='reference_perimeter_overlap',center=center.tolist(),roi=diagnostic['roi'],perimeter_outside_fraction=fraction,center_outside=outside))
        for mode in ['off','body']:
            item[mode+'_exact_four_current']=four[mode];item[mode+'_convex_four_hypotheses']=convex[mode]
            item[mode+'_median_hub_axis_ratio']=float(np.median(conditions[mode])) if conditions[mode] else None
        summary.append(item)
    current.save(OUT/'geometry_current_diagnostic.json',dict(table=summary,details=details,labels_or_future_read=False,limits=p['limits']))
    for item in summary:print(item,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','analyze']);globals()[parser.parse_args().action]()
