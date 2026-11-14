"""Shorter real RAW replay must match the frozen full-run prefix."""
import json
from study35 import ROOT,OUT,METHODS,EventImages,BodyTracker,robust23,study34,verify,save,sha


def main():
    verify();case='train3';cfg=study34.CASES[case]
    observed=list(map(json.loads,(ROOT.parent/'output'/cfg['observation']).read_text(encoding='utf-8').splitlines()))
    expected=list(map(json.loads,(OUT/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()))
    forecasts=list(map(json.loads,(OUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines()))
    reader=study34.open_raw(cfg['raw']);h,w=reader.get_size();images=EventImages(w,h);trackers={m:BodyTracker(w,h,m) for m in METHODS if m!='baseline'};models={m:robust23.Predictor() for m in METHODS};n=0
    for i,(t,e) in enumerate(study34.raw_chunks(reader,25.,1.9,10000)):
        o=observed[i]['observation'];image,current,strong=images.step(e,t);result={'baseline':dict(observation=o,audit=dict(source='geometry' if o['detected'] else 'none'))}
        for m,p in trackers.items():q,a=p.step(image,current,strong,t,o,len(e));result[m]=dict(observation=q,audit=a)
        assert dict(t=t,methods=result)==expected[i],(t,'prefix observation differs')
        ff={}
        for m,p in models.items():q=result[m]['observation'];ff[m]=p.step(t,q['center'] if q['detected'] else None)[0]
        assert dict(t=t,frontends=ff)==forecasts[i],(t,'prefix forecast differs')
        n+=1
    assert n>=189
    save(OUT/'real_prefix_check.json',dict(frames=n,shorter_raw_duration_s=1.9,full_duration_s=6.,all_four_observations_and_six_forecasts_exact=True,
        includes_actual_flow_measurements=True,helper_sha256=sha(__import__('pathlib').Path(__file__))))
    print('Stage35 real RAW prefix exact: '+str(n),flush=True)


if __name__=='__main__':main()
