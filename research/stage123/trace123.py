"""Read-only extraction funnel trace on a failed extension, exact replay."""
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage123'),str(ROOT/'research/stage110'),str(ROOT/'research/stage109'),str(ROOT/'research/stage21')]
from study123 import OUT,CASES,rows,save,sha,verify
from merge110 import MergeRecovery
from windows21 import Observer
from audit109 import audit_frame
from event_drone_forecast.raw_io import open_raw,raw_chunks

class Trace(MergeRecovery):
    def begin(self,capture):self.capture=capture;self.calls=[];self.grouped=[];self.selected=[]
    def feature(self,on,off):
        f=super().feature(on,off)
        if self.capture:self.feature_image=f.copy()
        return f
    def _candidates(self,contours,activity,t,b,thin_only=False):
        result=super()._candidates(contours,activity,t,b,thin_only)
        if self.capture:self.calls.append(dict(thin_only=thin_only,contours=[c.copy() for c in contours],activity=activity.copy(),accepted=copy.deepcopy(result)))
        return result
    def _group(self,candidates,t):
        result=super()._group(candidates,t)
        if self.capture:self.grouped=copy.deepcopy(candidates);self.selected=[] if result is None else copy.deepcopy(result)
        return result

def main():
    verify();folder=OUT/'extraction_trace2';folder.mkdir(exist_ok=False)
    cfg=CASES['extension2'];save(folder/'protocol.json',dict(case=cfg,snapshot_times=[43.5,45.],sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/stage110/merge110.py',ROOT/'research/stage109/audit109.py']},
        design='Post-efficacy diagnosis only. Re-read same300 RAW ticks, same110 native20 geometry, instrumentation only; compare actual rotor output exactly to122 worker. Existing NN boxes used only for crop/trace. No changed threshold, contour, grouping or prediction.'))
    ref=rows(OUT/'raw/extension2/observations_forecasts.jsonl');lookup={round(r['t'],8):r for r in ref};reader=open_raw(cfg['raw']);h,w=reader.get_size();front=Observer(w,h,'native20');front.geometry=Trace(w,h,'merge_safe');records=[];snapshots=[];begin=cfg['start']-.1
    for t,e in raw_chunks(reader,begin,cfg['duration']+.1,20000):
        if t-begin<.1-1e-8:continue
        capture=any(abs(t-v)<1e-8 for v in [43.5,45.]);g=front.geometry;g.begin(capture);obs=front.step(e,t)['observation']
        if t<=cfg['start']+1e-8:continue
        r=lookup[round(t,8)];expected=r['frontends']['delayed_feedback']['learned_structure118']['input_rotors'];assert obs.get('rotors',[])==expected,(t,'instrumented geometry mismatch')
        records.append(dict(t=t,selected=len(expected),candidates=len(g.candidates),group=copy.deepcopy(g.group_info),preline=copy.deepcopy(g.preline_audit),merge=copy.deepcopy(g.merge_audit)))
        if capture:
            audit_frame(g,r['body_observation'],e,t,folder);snapshots.append(dict(t=t,accepted_candidates=len(g.grouped),selected=len(g.selected),candidate_modes=[z['mode'] for z in g.grouped],candidate_sizes=[[z['major'],z['minor']] for z in g.grouped],group=copy.deepcopy(g.group_info)))
    assert len(records)==300;save(folder/'trace123.json',dict(frames=300,all_rotors_exact122=True,records=records,snapshots=snapshots,no_forecasts_generated=True,parameters_changed=False))
    print(json.dumps(dict(frames=300,all_rotors_exact122=True,snapshots=snapshots)),flush=True)
if __name__=='__main__':main()
