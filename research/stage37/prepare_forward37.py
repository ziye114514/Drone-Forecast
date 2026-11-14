"""SDK forward-only bridge: read largest past window once, derive smaller windows."""
from pathlib import Path
import datetime,json
import cv2,numpy as np
from common37 import ROOT,OUT,CASES,PARAMS,load,save,sha,verify
from study37 import samples,open_raw
from render37 import past,render


def main():
    p=verify();assert not (OUT/'inputs.json').exists() and not (OUT/'predictions.jsonl').exists()
    images=OUT/'images';preserved=OUT/'partial_preparation_v2_failed'
    if images.exists():
        assert not preserved.exists() and images.resolve().parent==OUT.resolve() and preserved.resolve().parent==OUT.resolve()
        preserved.mkdir();target=preserved/'images';assert target.resolve().parent.parent==OUT.resolve();images.rename(target)
    save(OUT/'preparation_bridge.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'),
        reason='Frozen preparation tried10ms then50ms at one cutoff; SDK RawReader forbids backward seeks. Preserve its partial images, read maximal50ms once and subset10ms. Algorithm, samples, renderings and thresholds unchanged. No detector inference has run.',
        truth_separation='GT read for sampling/preparation only; input manifest has no states/boxes. Detector access guard remains enabled.'))
    truth={s['id']:s for s in samples()};inputs=[];targets=[];checks=[]
    for case,cfg in CASES.items():
        reader=open_raw(cfg['raw']);h,w=reader.get_size();assert (h,w)==(720,1280)
        if 'baseline' in cfg:
            baseline=list(map(json.loads,(ROOT.parent/'output'/cfg['baseline']).read_text(encoding='utf-8').splitlines()));bt=np.array([r['t'] for r in baseline])
        for s in sorted([s for s in p['sample_selection'] if s['case']==case],key=lambda s:s['cutoff_us']):
            target=dict(truth[s['id']]);assert all(target[k]==s[k] for k in s);expected_count=None
            if 'baseline' in cfg:
                i=int(np.searchsorted(bt,s['cutoff_us']/1e6+1e-8,side='right')-1)
                if i>=0 and s['cutoff_us']/1e6-bt[i]<=.025:
                    target['baseline_geometry']=baseline[i]['observation']
                    if abs(bt[i]-s['cutoff_us']/1e6)<1e-8:expected_count=baseline[i]['event_count']
                else:target['baseline_geometry']=None
            targets.append(target);maximum=max(PARAMS['windows_us']);begin=s['cutoff_us']-maximum
            if reader.current_time>begin:reader=open_raw(cfg['raw'])
            reader.seek_time(begin);whole=past(reader.load_delta_t(maximum),s['cutoff_us'],maximum)
            for window in PARAMS['windows_us']:
                events=past(whole,s['cutoff_us'],window);assert not len(events) or int(events['t'].max())<s['cutoff_us']
                if window==10000 and expected_count is not None:assert len(events)==expected_count,(s['id'],len(events),expected_count)
                checks.append(dict(id=s['id'],window_us=window,events=len(events),min_t_us=int(events['t'].min()) if len(events) else None,max_t_us=int(events['t'].max()) if len(events) else None,
                    original10ms_event_count_exact=bool(len(events)==expected_count) if window==10000 and expected_count is not None else None))
                for kind in PARAMS['formats']:
                    image=render(events,w,h,kind);name=f'images/{s["id"]}_{window}_{kind}.png';path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
                    cv2.imencode('.png',image)[1].tofile(str(path));inputs.append(dict(id=s['id'],case=case,cutoff_us=s['cutoff_us'],window_us=window,format=kind,image=name,image_sha256=sha(path),events=len(events),width=w,height=h))
        print('Forward-only inputs prepared: '+case,flush=True)
    save(OUT/'inputs.json',dict(images=inputs,scope='No boxes/states/future labels. Native event renderings only.'))
    save(OUT/'targets.json',dict(targets=targets,scope='Evaluation only; detector must not open this file.'))
    save(OUT/'input_window_checks.json',checks)
    save(OUT/'preparation.json',dict(images=len(inputs),targets=len(targets),inputs_sha256=sha(OUT/'inputs.json'),targets_sha256=sha(OUT/'targets.json'),
        preparation_helper_sha256=sha(Path(__file__)),bridge_supplement_sha256=sha(OUT/'preparation_bridge.json'),sources_unchanged=verify()['source_hashes']==p['source_hashes'],
        original10ms_windows_exact=sum(r['original10ms_event_count_exact'] is True for r in checks)))
    print('Prepared causal detector inputs: '+str(len(inputs)),flush=True)


if __name__=='__main__':main()
