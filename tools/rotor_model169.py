"""Export evidence + visibility-aware geometry from existing native outputs.

Usage: python tools/rotor_model169.py --input <predictions.jsonl> --output <newdir>
No weights or coordinate annotations are loaded by this standalone exporter.
"""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rotor_evidence import RotorEvidence,VisibilityLayout
from local_period_tracks import LocalPeriodTracks
from run_rotor_only import serial

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);evidence=RotorEvidence();layout=VisibilityLayout();periods=LocalPeriodTracks();summary={};count=0;side_count=0;upstreamNN=False
    with Path(a.input).open(encoding='utf-8') as source,(out/'rotor_model.jsonl').open('w',encoding='utf-8') as dest:
        for line in source:
            r=json.loads(line);packet=evidence.step(r['t'],r['body'],r['structure'],r.get('geometry',{}).get('rotors',[]),r.get('rates',[]))
            upstreamNN|=bool(r.get('NN_used_for_localization_and_ROI') or r.get('whole_body_NN_used'))
            packet['local_periodic_change_proxy']=r.get('local_source_tracking') or periods.step(r['t'],r.get('local_temporal_phase') or {})
            side_count+=packet['local_periodic_change_proxy'].get('qualified',False)
            g=layout.step(packet);count+=1;summary[g['model']]=summary.get(g['model'],0)+1
            dest.write(json.dumps(dict(evidence=packet,layout=g),default=serial,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(frames=count,models=summary,local_periodic_proxy_frames=side_count,NN_loaded=False,
        upstream_NN_localization_recorded=upstreamNN,
        input_is_already_extracted_native_candidates=True,physical_truth_claimed=False),indent=2),'utf-8')
    print(json.dumps(summary))
