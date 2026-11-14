"""Deliverable checks and cross-chat handoff checkpoint; algorithms untouched."""
from pathlib import Path
import hashlib,json,re,sys
import cv2
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output'


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def checks():
    smoke()
    reports={21:'stage21_native_windows',22:'stage22_sparse_endpoint',23:'stage23_robust_acceleration',24:'stage24_raw_pipeline',25:'stage25_controlled_occlusion',26:'stage26_natural_layout',27:'stage27_same_state_physics',28:'stage28_axis_propagation',29:'stage29_delayed_model_weights',30:'stage30_raw_adaptive_physics'}
    if (OUT/'stage31_cross_recording_physics_20261018/第三十一阶段跨录像物理预测报告.html').exists():reports[31]='stage31_cross_recording_physics'
    reports.update({32:'stage32_cross_failure_audit',33:'stage33_rotor_slot_association',34:'stage34_slot_frequency_forecast',35:'stage35_body_feature_flow',36:'stage36_full_side_body_support'})
    reports.update({37:'stage37_upstream_detector',38:'stage38_noise_scale_detector',39:'stage39_sdk_noise_geometry'})
    reports.update({40:'stage40_record_split_finetune',41:'stage41_continuous_body_forecast',42:'stage42_raw_body_bridge'})
    for stage,name in reports.items():
        folder=OUT/(name+'_20261018');p=load(folder/'protocol.json');mismatch=[]
        for name,value in p['source_hashes'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:mismatch.append(name)
        assert not mismatch,mismatch
        bad=[];html=list(folder.glob('*.html'));assert html
        for page in html:
            for ref in re.findall(r'(?:href|src)="([^"]+)"',page.read_text(encoding='utf-8')):
                if '://' in ref or ref.startswith('#'):continue
                if not (page.parent/ref).exists():bad.append(ref)
        assert not bad,bad
        save(folder/'delivery_checks.json',dict(current_executed_source_hashes_unchanged=True,html_local_links_valid=True,html_files=[p.name for p in html],scope='Cached historical source mismatch, when present, remains separately declared in stage27 protocol.'))


def smoke():
    # New CLI smoke must reproduce both full observation and all model output.
    folder=OUT/'stage26_natural_layout_20261018';reference={round(r['t'],8):r for r in map(json.loads,(folder/'clear6/observations.jsonl').read_text(encoding='utf-8').splitlines())}
    forecasts={round(r['t'],8):r for r in map(json.loads,(folder/'clear6/forecasts.jsonl').read_text(encoding='utf-8').splitlines())}
    rows=list(map(json.loads,(folder/'cli_smoke/observations_forecasts.jsonl').read_text(encoding='utf-8').splitlines()))
    assert len(rows)==100
    for r in rows:
        assert r['observation']==reference[round(r['t'],8)]['methods']['layout20']['observation']
        assert r['forecasting']==forecasts[round(r['t'],8)]['frontends']['layout20']
    cap=cv2.VideoCapture(str(folder/'cli_smoke/forecasts.mp4'));frames=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(720,1280);frames+=1
    cap.release();assert frames==100
    save(folder/'cli_smoke/regression.json',dict(frames=100,observations_exact=True,all_forecasts_exact=True,decoded_mp4_frames=frames))
    print('RAW-layout smoke verified',flush=True)


if __name__=='__main__':checks();print('Latest completed deliverables verified',flush=True)
