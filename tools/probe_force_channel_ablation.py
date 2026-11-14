"""Does the frequency-derived thrust channel move the scored origins at all?

Frozen inputs only. Every variant replays the SAME rows and differs only in
``force.input_u``:

    as_is            frozen input (current default behaviour)
    no_thrust_input  input_u=None on every tick (the whole frequency->thrust
                     channel removed, position/curvature/flow branches untouched)
    variance_input   clear clip only: relative f2 taken from the independently
                     attributed multi-peak set; None where four slots are not
                     corroborated
    variance_hybrid  same, but falls back to the frozen input when incomplete

No predictor code is modified, no label enters a decision, and the legacy
replay must first reproduce the frozen predictions before anything is compared.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from probe_same_window_motion import replay
from compare_final_pipeline import score
from visibility_period import force_input

BASE=ROOT.parent/'output'
OUT=BASE/'peak_ownership_20261029'
CLIPS=[('side','motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl',
        'dense_labels_side_110_1195'),
       ('other','thrust_rate_review_20261029/experiment/other_baseline.jsonl','dense_labels_train2_40_46'),
       ('clear','motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl',None)]
OPTIONS=dict(curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,weak_support=True)


class VarianceInput:
    """Relative f2 from the independently attributed four-rotor peak set."""
    gap_s=.12

    def __init__(self,records):
        self.records=sorted(records,key=lambda r:r['t'])
        self.times=np.array([r['t'] for r in self.records])
        self.reference=None;self.previous=None;self.pairs=[]

    def peaks(self,t):
        k=int(np.searchsorted(self.times,t))
        near=[j for j in (k-1,k) if 0<=j<len(self.records)]
        if not near:return None
        j=min(near,key=lambda i:abs(self.times[i]-t))
        if abs(self.times[j]-t)>1/60:return None
        record=self.records[j]
        if len(record.get('claimed_slots',[]))!=4 or len(record.get('corroborated_slots',[]))!=4:
            return None
        frequencies={a['owner_model_id']:a['peak']['frequency_hz'] for a in record['attribution']
            if a.get('independent') and a.get('owner_model_id') is not None}
        if len(frequencies)!=4:return None
        return frequencies,record

    def step(self,t,structure,sdk_relative):
        found=self.peaks(t)
        if found is None:return None,dict(available=False)
        frequencies,_=found
        total=float(sum(f*f for f in frequencies.values()))
        if self.reference is None:self.reference=total
        relative=total/self.reference
        coherent=False
        if self.previous is not None and t-self.previous[0]<self.gap_s:
            coherent=abs(.5*np.log(relative/self.previous[1]))<.08
        self.previous=(t,relative)
        u,quality=force_input(structure,relative,coherent)
        self.pairs.append(dict(t=t,relative_f2=relative,sdk_relative_f2=sdk_relative,
            relative_difference=None if sdk_relative is None else relative/sdk_relative-1.,
            peak_sum_hz2=total,coherent=bool(coherent)))
        return u,dict(available=True,relative_f2=relative,coherent=bool(coherent),
            peaks_hz={str(k):float(v) for k,v in frequencies.items()},quality=quality)


def build(rows,mode,variance=None):
    out=[]
    for r in rows:
        current=r['force'].get('input_u')
        replacement=None;info=dict(available=False)
        if mode in ('variance_input','variance_hybrid') and variance is not None:
            sdk=r.get('period_selection',{}).get('relative_f2')
            replacement,info=variance.step(r['t'],r['structure'],sdk)
        if mode=='no_thrust_input':current=None
        elif mode=='variance_input' and variance is not None:current=replacement
        elif mode=='variance_hybrid' and replacement is not None:current=replacement
        out.append(dict(r,force=dict(r['force'],input_u=current),variance_input=info))
    return out


def difference(rows_a,rows_b):
    values=[]
    for a,b in zip(rows_a,rows_b):
        for h in a['physical']:
            if h in b['physical']:
                values.append(float(np.linalg.norm(np.asarray(a['physical'][h])-np.asarray(b['physical'][h]))))
    return dict(n=len(values),median=float(np.median(values)) if values else None,
        p95=float(np.percentile(values,95)) if values else None,
        max=float(max(values)) if values else None)


def partition(rows,truth):
    """0.4s error split by whether the thrust channel had input at that origin."""
    from compare_final_pipeline import reference_at
    groups={True:[],False:[]};last=None
    for r in rows:
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last:continue
        last=slot
        target=reference_at(truth,r['t']+.4)
        if target is None or '0.4' not in r['physical']:continue
        groups[bool(r['force'].get('active'))].append(
            float(np.linalg.norm(np.asarray(r['physical']['0.4'])-target)))
    return {('with_input' if k else 'without_input'):dict(n=len(v),
        mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None,
        p95=float(np.percentile(v,95)) if v else None) for k,v in groups.items()}


def near_segment():
    """The only densely labelled clear-view segment available; not a four-rotor view."""
    run=BASE/'peak_ownership_20261029/RAW_clear_20_21p4'
    if not (run/'predictions.jsonl').exists():return None
    rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text('utf-8').splitlines()]
    records=json.loads((BASE/'dense_labels_train6_20_35/labels.json').read_text('utf-8'))['records']
    truth=sorted(records.values() if isinstance(records,dict) else records,key=lambda r:r['t'])
    scored=score(rows,truth)
    summary=json.loads((run/'summary.json').read_text('utf-8-sig'))
    return dict(raw='E:/dataset/train/6/media/event.raw',start=20.0,duration=1.4,
        labels='dense_labels_train6_20_35',label_rate_hz=20.0,
        label_span_s=list(sorted(r['t'] for r in truth))[::len(truth)-1] if len(truth)>1 else None,
        modes=summary['modes'],steady_capacity_fps=summary['steady_capacity_fps'],
        current_force_inputs=summary['current_force_inputs'],
        manual_anchor_count=scored['manual_anchor_count'],detected_anchors=scored['detected_anchors'],
        box_center_error_px=scored['box_center_error_px'],
        future_error_px=scored['physical_future_error_px'],
        four_rotor_ticks=summary['modes'].get('four',0),
        note='edge-clipped close view: insufficient/axis_only only, so no rotor slots and no variance '
             'attribution; the segment cannot test the thrust channel')


def sparse_clear_localization():
    """The sparse archived development boxes on the same 26-28s clear clip."""
    import zipfile
    archive=ROOT.parent/'archive/output_history_20261028.zip'
    if not archive.exists():return None
    with zipfile.ZipFile(archive) as handle:
        data=json.loads(handle.read('paper_reproduction_v030/evaluation_data/train6.json').decode('utf-8'))
    rows=[json.loads(x) for x in
        (BASE/'motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl').read_text('utf-8').splitlines()]
    anchors=[]
    for label in sorted(data['labels'],key=lambda r:r['t']):
        if not 26.0<=label['t']<=28.0:continue
        if label.get('state')!='present' or not label.get('bbox'):continue
        row=min(rows,key=lambda r:abs(r['t']-label['t']))
        if abs(row['t']-label['t'])>1/60:continue
        entry=dict(t=label['t'],source_t=row['t'],detected=bool(row['body'].get('detected')))
        if entry['detected']:
            box=np.asarray(label['bbox']);center=(box[:2]+box[2:])/2
            entry['error_px']=float(np.linalg.norm(np.asarray(row['body']['center'])-center))
        anchors.append(entry)
    errors=[a['error_px'] for a in anchors if a.get('error_px') is not None]
    return dict(archive_entry='archive/output_history_20261028.zip!paper_reproduction_v030/evaluation_data/train6.json',
        label_rate_hz=1.5,anchors=anchors,detected=len(errors),anchor_count=len(anchors),
        mean_error_px=float(np.mean(errors)) if errors else None,
        note='sparse development boxes; spacing exceeds the150ms interpolation gate so they cannot score '
             'the0.4s target, only current localization')


def main():
    result=dict(clips={},default_enabled=False,forecaster_modified=False,annotations_used_for_decisions=False,
        labeled_near_segment=near_segment(),sparse_clear_localization=sparse_clear_localization())
    for name,rel,labels in CLIPS:
        rows=[json.loads(x) for x in (BASE/rel).read_text('utf-8').splitlines()]
        truth=None
        if labels:
            records=json.loads((BASE/labels/'labels.json').read_text('utf-8'))['records']
            truth=sorted(records.values() if isinstance(records,dict) else records,key=lambda r:r['t'])
        audit_path=OUT/f'audit_{name}.jsonl'
        variance=None
        if name=='clear' and audit_path.exists():
            variance=VarianceInput([json.loads(x) for x in audit_path.read_text('utf-8').splitlines()])
        modes=['as_is','no_thrust_input']+(['variance_input','variance_hybrid'] if variance else [])
        replayed={};entry=dict(modes={},input_pairs=None)
        for mode in modes:
            variant=build(rows,mode,variance)
            values,cost=replay(variant,False,**OPTIONS)
            replayed[mode]=values
            active=sum(bool(r['force'].get('active')) for r in values)
            mode_entry=dict(force_active_ticks=active,mean_compute_ms=float(np.mean(cost)))
            if truth:
                mode_entry['error_0.4']=score(values,truth)['physical_future_error_px']['0.4']
                mode_entry['error_by_input_presence']=partition(values,truth)
            entry['modes'][mode]=mode_entry
            if mode!='as_is':
                entry['difference_'+mode]=difference(replayed['as_is'],values)
        frozen=[{'physical':r['physical']} for r in rows]
        reproduced=max(float(np.max(np.abs(np.asarray(a['physical'][h])-np.asarray(b['physical'][h]))))
            for a,b in zip(frozen,replayed['as_is']) for h in a['physical'])
        entry['as_is_reproduces_frozen_predictions']=bool(reproduced<1e-6)
        entry['as_is_max_absolute_reproduction_difference']=reproduced
        if variance is not None and variance.pairs:
            diffs=[abs(p['relative_difference']) for p in variance.pairs if p['relative_difference'] is not None]
            entry['input_pairs']=dict(n=len(variance.pairs),median_relative_difference=float(np.median(diffs)) if diffs else None,
                max_relative_difference=float(max(diffs)) if diffs else None,
                coherent_ticks=sum(p['coherent'] for p in variance.pairs),
                pairs=variance.pairs)
        result['clips'][name]=entry
        print(name,json.dumps({k:v for k,v in entry.items() if k!='input_pairs'},ensure_ascii=False),flush=True)
    (OUT/'force_channel_ablation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print(json.dumps({k:{m:v['error_0.4']['mean'] if v.get('error_0.4') else None
        for m,v in c['modes'].items()} for k,c in result['clips'].items()},ensure_ascii=False))


if __name__=='__main__':main()
