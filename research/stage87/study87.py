"""One parent imports API and performs two genuine sequential RAW runs."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage85')]
import study85 as prior
from raw_forecast_api import forecast_raw
load=prior.load;save=prior.save;sha=prior.sha;rows=prior.rows
OUT=ROOT.parent/'output/stage87_process_safe_api_20261022'
CASES=dict(clear6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=.24),
           side6=dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=.24))
def sources():
    result=prior.sources();result['raw_forecast_api.py']=sha(ROOT/'core/raw_forecast_api.py');result['research/stage87/study87.py']=sha(__file__);return result
def freeze():
    prior.verify();assert not OUT.exists() and (prior.OUT/'checks85.json').is_file()
    p=load(prior.OUT/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        raw_metadata=p['raw_metadata'],reference_hashes={c:sha(prior.OUT/c/'observations_forecasts.jsonl') for c in CASES},
        design='One parent process imports only stdlib API, then sequentially invokes '
            'two genuine85 RAW subprocesses. Read project JSON after each, all '
            'current/full49 exact same85 prefix. Test existing output protection '
            'and invalid RAW refusal. No new trajectory accuracy/realtime claim.',default_promoted=False))
    print('87 process boundary protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for f,m in p['raw_metadata'].items():
        st=Path(f).stat();assert m==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p
def run():
    p=verify();assert not (OUT/'checks87.json').exists()
    # This Python parent has SDK imports from experiment verification, so use
    # an additional independent stdlib-only import probe for the API contract.
    import subprocess
    probe=subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,"+repr(str(ROOT))+");import raw_forecast_api;assert not any(x.startswith(('metavision','torch','cv2','numpy')) for x in sys.modules);print('stdlib-only API import passed')"],cwd=ROOT,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert probe.returncode==0,probe.stderr
    results=[]
    for case,cfg in CASES.items():
        result=forecast_raw(cfg['raw'],OUT/case,start=cfg['start'],duration=cfg['duration'],verify_frames=True)
        # Access metadata elsewhere in the SAME parent. Old in-process run()
        # audit namespace cleanup broke this operation after inference ended.
        assert load(prior.OUT/'checks85.json')['all49_exact78']
        assert sha(prior.OUT/case/'observations_forecasts.jsonl')==p['reference_hashes'][case]
        actual=rows(result.observations);expected=rows(prior.OUT/case/'observations_forecasts.jsonl')[:12]
        assert len(actual)==len(expected)==12
        keys=['t','event_activity','body_axis','event_measurement_audit','geometry20','body_observation','observation','sensing','slot_audit','frontends','lifecycle','frame_sha256','event_fields_sha256']
        for r,e in zip(actual,expected):
            for k in keys:assert r[k]==e[k],(case,r['t'],k)
        refused=False
        try:forecast_raw(cfg['raw'],OUT/case,start=cfg['start'],duration=cfg['duration'])
        except FileExistsError:refused=True
        assert refused
        results.append(dict(case=case,parent=12,native10=24,all_current49_exact85=True,parent_IO_after_child_ok=True,
            existing_output_refused=True,wall_including_cold_start_s=result.elapsed_wall_s,process_log=str(result.process_log)))
        print(case,'87 same parent call and IO passed',flush=True)
    refused=False
    try:forecast_raw(ROOT/'README.md',OUT/'invalid_raw')
    except ValueError:refused=True
    assert refused and not (OUT/'invalid_raw').exists()
    save(OUT/'checks87.json',dict(table=results,parent=24,native10=48,stdlib_only_API_import=True,
        two_calls_same_parent=True,parent_file_IO_safe=True,invalid_RAW_refused_without_output=True,
        all49_exact85=True,algorithm_changed=False,asynchronous_pipeline=False,default_promoted=False))
    print('87 real sequential API tests complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
