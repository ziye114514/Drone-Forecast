"""Instrument frozen side replay; diagnostic shadows never update detector state."""
from pathlib import Path
import json,sys,collections,time
import numpy as np
from study import OUT,ROOT,verify,counts,open_raw,raw_chunks,save
from polarity import PolarityFrontend

DEST=ROOT.parent/'output/stage8_side_diagnosis_20261017'


def plain(r):
    return dict(cx=float(r['center'][0]),cy=float(r['center'][1]),
                major=float(r['major']),minor=float(r['minor']),mode=r['mode'])


class Inspect(PolarityFrontend):
    def _group(self,candidates,t):
        self.debug=dict(candidate_count=len(candidates),candidates=[plain(r) for r in candidates],
            center=None if self.center is None else self.center.tolist(),velocity=self.velocity.tolist(),
            last_t=self.last_t,last_seen=self.last_seen,last_extent=self.last_extent,
            constellation=self.constellation is not None,pending_streak=self.pending_streak)
        selected=super()._group(candidates,t)
        # _group is read-only. Disable only its predicted-center gate/penalty in a
        # diagnostic call, then restore state before actual detect() continues.
        original_center=self.center
        try:
            self.center=None
            shadow=super()._group(candidates,t)
        finally:
            self.center=original_center
        self.debug.update(selected_count=0 if selected is None else len(selected),
            selected=[] if selected is None else [plain(r) for r in selected],
            selected_without_motion_gate=0 if shadow is None else len(shadow),
            shadow=[] if shadow is None else [plain(r) for r in shadow])
        return selected


def run():
    protocol=verify();cfg=protocol['cases']['side'];reader=open_raw(cfg['raw']);h,w=reader.get_size()
    fronts={m:Inspect(w,h,m) for m in ['strict10','neighbor10']}
    old={round(r['t'],8):r for r in map(json.loads,(OUT/'side/observations.jsonl').read_text().splitlines())}
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'completion.json').exists()
    t0=time.perf_counter();records=[];matched=0
    with (DEST/'trace.jsonl').open('w',encoding='utf-8') as file:
        for t,e in raw_chunks(reader,cfg['start']-.03,cfg['duration']+.03,10000):
            prepared=counts(e,w,h);r=dict(t=t,methods={})
            for m,f in fronts.items():
                obs,_=f.detect_prepared(e,t,prepared)
                r['methods'][m]=dict(observation=obs,group=f.debug)
            if round(t,8) in old:
                for m in fronts:assert r['methods'][m]['observation']==old[round(t,8)]['methods'][m]['observation'],(t,m)
                matched+=1;records.append(r)
            file.write(json.dumps(r,allow_nan=False)+'\n')
    groups={}
    for label,subset in [('all',records),('strict_only',[r for r in records if r['methods']['strict10']['observation']['detected'] and not r['methods']['neighbor10']['observation']['detected']])]:
        groups[label]={}
        for m in fronts:
            reason=collections.Counter()
            for r in subset:
                x=r['methods'][m];g=x['group'];o=x['observation']
                if o['detected']:reason['published']+=1
                elif g['selected_count']:
                    reason['group_found_but_confirmation_not_met']+=1
                elif g['selected_without_motion_gate']:
                    reason['all_geometric_groups_rejected_by_motion_gate']+=1
                elif g['candidate_count']==0:reason['no_candidates']+=1
                else:reason['no_geometric_group_even_without_motion_gate']+=1
            groups[label][m]=dict(reason)
    save(DEST/'completion.json',dict(matched_published_frames=matched,wall_s=time.perf_counter()-t0,
        both_methods_exactly_equal_original=True,no_inference_parameters_changed=True,
        diagnostic_shadow_used_for_state_updates=False,counts=groups,
        limits='No target or per-rotor ground truth. A hypothetical valid group is not necessarily correct. Instrumentation identifies code rejection, not full counterfactual long-term performance.'))
    print(json.dumps(groups),flush=True)


if __name__=='__main__':run()
