"""Small RAW shadow test of local periodicity without centers or ellipse masks."""
from pathlib import Path
from collections import deque,Counter
import argparse,datetime,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage64')]
from periodicity66 import *
import study64 as previous
from compensation64 import PastVelocity
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=ROOT.parent/'output/stage66_center_free_periodicity_20261021'
load=previous.load;save=previous.save;sha=previous.sha;rows=previous.rows
CASES={c:dict(previous.CASES[c]) for c in previous.MAIN}
CASES['clear6'].update(start=22.,duration=.8);CASES['side6'].update(start=112.,duration=1.)
CASES['train3'].update(start=28.,duration=1.);CASES['negative5'].update(start=0.,duration=.4)


def sources():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),Path(__file__).with_name('periodicity66.py'),
        ROOT/'research/stage64/compensation64.py',ROOT/'src/event_drone_forecast/rotor_sensing.py',
        ROOT/'src/event_drone_forecast/paper_config.py',ROOT/'src/event_drone_forecast/raw_io.py']}


def freeze():
    previous.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        cases=CASES,parameters=PARAMS,current_input_hashes={c:sha(previous.prior.REF/f'off_{c}/observations_forecasts.jsonl') for c in CASES},
        raw_metadata=load(previous.OUT/'protocol.json')['raw_metadata'],
        design='Original paper local ROI temporal bins/FFT without center/ellipse mask or '
               'phase-based mechanical RPM selection.6x3 image tiles are not four motor '
               'labels. Original160ms/100us/polarity Hann periodograms/SNR8/range retained. '
               'Compare raw and past-CV-compensated spatial assignment, plus half-window '
               'diagnostic. Uniform whole-buffer cap450000 like original sensing.',
        truth='No future labels/pose/RPM truth. Known synthetic rate signals plus Poisson, '
              'single-burst and step controls; tiles never converted to total thrust.',
        prior_art=['https://arxiv.org/html/2604.18289v1','https://github.com/ros-event-camera/frequency_cam'],
        actual_reuse='Original local polarity FFT mathematics and64 past velocity. '
                     'FrequencyCam inspected, not executed; no brightness/G reconstruction. '
                     'Cached current NN boxes: shadow observation experiment, not joint deployment.'))
    save(OUT/'controlled66.json',controlled());print('66 local periodicity protocol and controls saved',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for c,h in p['current_input_hashes'].items():assert h==sha(previous.prior.REF/f'off_{c}/observations_forecasts.jsonl')
    for n,m in p['raw_metadata'].items():
        s=Path(n).stat();assert m==dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return p


def run(case):
    verify();sys.addaudithook(previous.prior.previous.guard);cfg=CASES[case];folder=OUT/case;folder.mkdir(exist_ok=False)
    reference=rows(previous.prior.REF/f'off_{case}/observations_forecasts.jsonl');lookup={round(r['t'],8):r for r in reference}
    past=PastVelocity();committed=-1;buffer=deque();result=[];stats={m:Counter() for m in ['raw','aligned']};native=0
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size()
    fixed=round((cfg['start']+cfg['duration']*.5)*50)/50
    for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
        begin=t-.01
        while committed+1<len(reference) and reference[committed+1]['t']<=begin+1e-8:
            committed+=1;past.commit(reference[committed])
        v,va=past.before(begin);buffer.append((t,e.copy()));native+=1
        while buffer and buffer[0][0]<t-PARAMS['window_us']/1e6-.010001:buffer.popleft()
        if (i+1)%2:continue
        r=lookup[round(t,8)];body=r['body_observation'];methods={}
        events=np.concatenate([x[1] for x in buffer]);events=events[(events['t']>=round(t*1e6)-PARAMS['window_us'])&(events['t']<round(t*1e6))]
        original_count=len(events);stride=max(1,int(np.ceil(len(events)/PARAMS['max_buffer_events'])));events=events[::stride]
        # A fixed background-only ROI on the original negative control is
        # intentional: do not make this check trivially pass via absent NN.
        box=body.get('bbox') if body.get('detected') and body.get('confidence',0)>=.45 else None
        if case=='negative5':box=[400.,250.,900.,450.]
        for method,velocity in [('raw',np.zeros(2)),('aligned',v)]:
            m=tile_measurement(events,box,t,velocity) if box is not None else dict(valid=False,reason='no_current_NN_identity',tiles=[])
            methods[method]=m;stats[method]['ROI_frames']+=m['valid'];stats[method]['frames_with_spectral_candidate']+=any(q['valid'] for q in m['tiles']);stats[method]['frames_with_half_stable_candidate']+=any(q['stable'] for q in m['tiles'])
            stats[method]['spectral_tiles']+=sum(q['valid'] for q in m['tiles']);stats[method]['half_stable_tiles']+=sum(q['stable'] for q in m['tiles'])
            if abs(t-fixed)<1e-8 and m['valid']:
                freq=np.full((3,6),np.nan);snr=np.full((3,6),np.nan)
                for tile in m['tiles']:
                    if tile['valid']:freq[tile['row'],tile['column']]=tile['peak_hz'];snr[tile['row'],tile['column']]=tile['peak_to_noise']
                save(folder/f'{method}_fixed_map.json',dict(t=t,bbox=box,frequencies_hz=np.where(np.isfinite(freq),freq,-1).tolist(),snr=np.where(np.isfinite(snr),snr,0).tolist(),tiles=m['tiles']))
        records=r['sensing']['rotors'];result.append(dict(t=t,methods=methods,velocity=v.tolist(),velocity_audit=va,
             raw_window_events=original_count,retained_events=len(events),uniform_stride=stride,
             baseline_any_frequency=any(q.get('frequency_valid') for q in records),baseline_four_frequencies=r['sensing']['four_frequencies'],
             negative_control_fixed_ROI=case=='negative5'))
    assert native==round(cfg['duration']*100) and len(result)==native//2
    with (folder/'observations.jsonl').open('x',encoding='utf-8') as f:
        for r in result:f.write(previous.prior.json.dumps(r,allow_nan=False)+'\n')
    summary=dict(case=case,native10=native,parent=len(result),methods={m:dict(d) for m,d in stats.items()},
                 baseline_any_frequency_frames=sum(r['baseline_any_frequency'] for r in result),
                 baseline_four_frequency_frames=sum(r['baseline_four_frequencies'] for r in result),
                 mechanical_RPM_or_thrust_estimated=False,labels_read=False)
    save(folder/'completion.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='run':run(a.case)
    else:freeze()
