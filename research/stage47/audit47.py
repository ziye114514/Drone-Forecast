"""Native pixels, real prefix, old geometry, new model and decoded video checks."""
import hashlib,json
import cv2,numpy as np
import study47 as s
def rows(p):return list(map(json.loads,p.read_text(encoding='utf-8').splitlines()))
def main():
    s.verify();extra=s.load(s.OUT/'pipe_proxy_supplement.json')
    for name,value in extra['source_hashes'].items():assert s.sha(s.ROOT/name)==value
    original=s.ROOT.parent/'output/stage41_continuous_body_forecast_20261018';manifest=s.load(original/'inputs.json')['images'];images={(r['case'],round(r['t'],8)):r for r in manifest};reference={precision:{round(r['t'],8):r for r in rows(s.ROOT.parent/f'output/stage{stage}_{folder}_20261019/sequence/train3/observations.jsonl')} for precision,stage,folder in [(32,45,'cuda_graph'),(16,46,'fp16_graph')]}
    checks=[];table=[];loaded={};pixel_count=0
    for case,cfg in s.CASES.items():
        rr=rows(s.OUT/case/'observations_forecasts.jsonl');loaded[case]=rr;assert len(rr)==round(cfg['duration']/.02);completion=s.load(s.OUT/case/'completion.json');protocol=s.load(s.OUT/case/'protocol.json');assert completion['source_hashes_unchanged']
        for name,value in protocol['source_hashes'].items():assert s.sha(s.ROOT/name)==value,name
        matches=0;max_box=0.;geometry_exact=0
        for r in rr:
            image=images['negative5' if case.startswith('negative') else 'train3',round(r['t'],8)];native=cv2.imdecode(np.fromfile(original/image['image'],np.uint8),1);assert hashlib.sha256(native.tobytes()).hexdigest()==r['frame_sha256'];pixel_count+=1
            if not case.startswith('negative'):
                q=reference[cfg['precision']][round(r['t'],8)];o=q['methods']['body_only']['observation'];new=r['observations']['body_only'];assert o['detected']==new['detected']
                if new['detected']:max_box=max(max_box,float(abs(np.array(o['bbox'])-new['bbox']).max()))
                matches+=o==new
                if not cfg['body_only']:assert r['observations']['geometry']==q['methods']['geometry']['observation'];geometry_exact+=1
            else:
                assert all(not o['detected'] for o in r['observations'].values());assert all(not q['valid'] for q in r['forecasts'].values())
        video_frames=0
        if cfg['preview']:
            cap=cv2.VideoCapture(str(s.OUT/case/'forecasts.mp4'))
            while True:
                ok,im=cap.read()
                if not ok:break
                assert im.shape==(720,1280,3);video_frames+=1
            cap.release();assert video_frames==len(rr)
        checks.append(dict(case=case,frames=len(rr),native_pixels_exact=len(rr),body_observation_exact_to45_or46=matches,body_max_bbox_diff_px=max_box,geometry_exact_to41=geometry_exact,decoded_video_frames=video_frames))
        time=np.array([r['processing_ms'] for r in rr]);table.append(dict(case=case,precision=cfg['precision'],geometry_enabled=not cfg['body_only'],frames=len(rr),processing_median_ms=float(np.median(time)),processing_p95_ms=float(np.percentile(time,95)),fraction_over20ms=float(np.mean(time>20)),fraction_over100ms=float(np.mean(time>100)),worker_median_ms=float(np.median([r['gpu_inference_ms'] for r in rr])),wall_s=completion['wall_including_model_load_decode_video_s']))
    prefix=loaded['prefix_body16'];full=loaded['train3_body16'];exact_obs=exact_q=0;max_prediction=0.
    for a,b in zip(full,prefix):
        assert a['t']==b['t'] and a['frame_sha256']==b['frame_sha256'];exact_obs+=a['observations']['body_only']==b['observations']['body_only'];exact_q+=a['forecasts']['body_only']==b['forecasts']['body_only'];qa=a['forecasts']['body_only'];qb=b['forecasts']['body_only'];assert qa['valid']==qb['valid']
        if qa['valid']:
            for method in qa['predictions']:
                for h in qa['predictions'][method]:max_prediction=max(max_prediction,float(abs(np.array(qa['predictions'][method][h])-qb['predictions'][method][h]).max()))
    s.save(s.OUT/'regression.json',dict(native_pixels_exact=pixel_count,cases=checks,prefix=dict(frames=len(prefix),body_observation_exact=exact_obs,all_six_forecasts_exact=exact_q,max_prediction_diff_px=max_prediction),negative_all_current_forecasts_zero=True,sources_unchanged=True,scope='Inference-only new shared-memory RAW entry. No new label/pose/RPM accuracy claim.'))
    s.save(s.OUT/'analysis.json',dict(table=table,scope='Includes full pixel SHA in parent and worker, geometry only when enabled, excludes RAW next/write/preview from processing timer; wall includes cold start/decode/video. Different modes intentionally have different scope, no full-system speedup claim from body-only.'))
    print(json.dumps(dict(pixel_frames=pixel_count,prefix_exact=exact_q,table=table)),flush=True)
if __name__=='__main__':main()
