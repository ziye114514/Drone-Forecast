"""Merge support regions into possible rotor-disc regions; report the whole curve.

Merging is a spatial grouping, not blade identity. The gap threshold is a
declared physical choice (a blade sweep spans roughly the disc minor axis), and
the count-vs-gap curve is published so nobody can pretend the number is
threshold-free. The frozen rotor conics are used only as a diagnostic reference.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from rotor_support_front import merge_region_clusters

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
GAPS=[4,8,12,16,24,32,48]
DEFAULT_GAP=24


def qualified(frame):return [track for track in frame['tracks'] if track['qualified']]


def coverage(clusters,rotors):
    """How many frozen rotors are explained by some merged cluster."""
    explained=0
    for rotor in rotors:
        cx,cy=float(rotor['cx']),float(rotor['cy'])
        semi=max(float(rotor.get('major',1))/2,1e-6)
        for cluster in clusters:
            x0,y0,x1,y1=cluster['bbox'];px=(x0+x1)/2;py=(y0+y1)/2
            if np.hypot(px-cx,py-cy)<=1.5*semi:explained+=1;break
    return explained


def main():
    tracks=json.loads((OUT/'tracks.json').read_text('utf-8'))
    result=dict(gap_curve={},default_gap_px=DEFAULT_GAP,
        default_gap_rationale='blade sweep spans roughly the disc minor axis (~28px in the frozen '
            'clear conics); the full curve is reported so the number is not presented as threshold-free',
        rotor_count_claimed=False,blade_identity_claimed=False)
    for name,entry in tracks['sequences'].items():
        curve={}
        for gap in GAPS:
            counts=[merge_region_clusters(qualified(frame),gap)['groups_after'] for frame in entry['frames']]
            curve[str(gap)]=dict(counts=counts,median=float(np.median(counts)),
                minimum=int(min(counts)),maximum=int(max(counts)))
        result['gap_curve'][name]=curve
        frames=[frame for frame in entry['frames'] if frame['t']>=entry['frames'][0]['t']]
        midpoint=0.5*(entry['frames'][0]['t']+entry['frames'][-1]['t'])
        at_target=min(frames,key=lambda frame:abs(frame['t']-midpoint))
        merged=merge_region_clusters(qualified(at_target),DEFAULT_GAP)
        detail=None
        predictions=entry['spec'].get('predictions')
        if predictions:
            rows=[json.loads(line) for line in (BASE/predictions).read_text('utf-8').splitlines()]
            row=min(rows,key=lambda item:abs(item['t']-at_target['t']))
            rotors=[r for r in row['geometry'].get('rotors',[]) if r.get('current_observation',True)]
            if rotors:
                detail=dict(t=at_target['t'],frozen_mode=row['structure'].get('mode'),
                    frozen_rotors=len(rotors),groups=merged['groups_after'],
                    regions_before=merged['regions_before'],
                    rotors_explained=coverage(merged['clusters'],rotors),
                    clusters=merged['clusters'])
        result.setdefault('at_midpoint',{})[name]=detail or dict(t=at_target['t'],groups=merged['groups_after'],
            regions_before=merged['regions_before'],clusters=merged['clusters'])
        print(name,{gap:curve[str(gap)]['median'] for gap in GAPS},
            'at_target',merged['regions_before'],'->',merged['groups_after'],flush=True)
    (OUT/'merge.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'merge.json')


if __name__=='__main__':main()
