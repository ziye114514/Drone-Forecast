"""Supplement inactive extension prefix with an activated natural-model prefix."""
from pathlib import Path
import datetime,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage123')]
from study123 import OUT,entry,rows,save,stable,verify
def main():
    verify();protocol=OUT/'positive_prefix_protocol123.json';assert not protocol.exists()
    save(protocol,dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),raw='E:/dataset/train/6/media/event.raw',start=22.,duration=1.2,
        reason='Original extension1 40..42 actual prefix contained0 detections/valid forecasts. Keep it, add first60 ticks of already completed positive natural control to test learned/active selector state. No new efficacy targets, no retuning; after primary scoring.',
        unchanged_algorithm_stage=122,checks='All available coefficients/predictions and internal worker states exactly equal first60 natural6; union69 initialized branches, at least one current geometry and validation update.'))
    dest=OUT/'raw/prefix_natural6';entry.run('E:/dataset/train/6/media/event.raw',dest,22.,1.2,False,True,-1)
    rr=rows(dest/'observations_forecasts.jsonl');full=rows(OUT/'raw/natural6/observations_forecasts.jsonl')[:60];assert len(rr)==len(full)==60;names=set();current=validation=0
    for a,b in zip(rr,full):
        qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback']
        for key in ['coefficients','predictions','learned_structure118']:assert stable(qa[key])==stable(qb[key]),(a['t'],key)
        assert a['event_fields_sha256']==b['event_fields_sha256'];names.update(qa['coefficients'])
        v=qa['learned_structure118']['validated122'];current+=int(v['center']['current']);validation+=int(v['selection']['validation'] is not None)
    assert len(names)==69 and current>0 and validation>0
    save(dest/'checks123.json',dict(actual_RAW=True,frames=60,independent_positive_prefix=True,all_available69_and_model_states_exact=True,
        initialized_branch_union=len(names),current_geometry=current,validation_updates=validation,labels_read=False,no_parameter_changes=True))
    print(json.dumps(dict(positive_prefix=60,branch_union=len(names),current_geometry=current,validation_updates=validation)),flush=True)
if __name__=='__main__':main()
