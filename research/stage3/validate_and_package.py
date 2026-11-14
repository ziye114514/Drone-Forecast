"""Frozen-source audit, prefix replay, baseline equality and reviewed evidence."""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
from html.parser import HTMLParser
import cv2,numpy as np
from experiments import ROOT,OUT,verify,save,digest
from analyze import traces
from raw_pipeline import ConsensusBank
from event_drone_forecast.paper_bank import PaperForecastBank
from event_drone_forecast.paper_config import PaperConfig

def main():
    protocol=verify()
    for rel,h in protocol['sources'].items():assert digest(OUT/'source_snapshot'/rel)==h
    raw_protocol=json.loads((OUT/'raw_v2/protocol.json').read_text())
    for rel,h in raw_protocol['source_hashes'].items():assert digest(ROOT/rel)==digest(OUT/'raw_v2/source_snapshot'/rel)==h
    first=json.loads((OUT.parent/'stage1_observation_errors_20261016/protocol.json').read_text())
    for rel,h in first['sources'].items():assert digest(ROOT/rel)==h
    runtime=hashlib.sha256()
    for p in sorted((ROOT/'src/event_drone_forecast').glob('*.py')):runtime.update(p.read_bytes())
    assert runtime.hexdigest()=='71b3eb3eb3e4c38acf438a29a757d651a7a7949a42f66b1c95acc64995a615b0'
    folder=OUT/'raw_v2/train6';obs=traces(folder/'observations.jsonl');cfg=PaperConfig()
    banks={'frozen':PaperForecastBank(cfg),'consensus':ConsensusBank(cfg,False),'consensus_audit':ConsensusBank(cfg,True)}
    published={name:traces(folder/name/'forecasts.jsonl') for name in banks};compared=0
    for k,r in enumerate(obs[:2600]):
        for name,bank in banks.items():
            q=bank.step(r['observation'],r['sensing'])
            if q is None:q=dict(t=r['t'],active=False,states={},physics_active=False)
            previous=published[name][k]
            assert q['active']==previous['active'] and q['physics_active']==previous['physics_active'],(name,k)
            for method,state in q['states'].items():assert state==previous['states'][method],(name,k,method)
        compared+=1
    save(OUT/'observation_prefix_validation.json',dict(frames=compared,through_time_s=26.,arms=3,prediction_states_exact=True,
        interpretation='Replay of truncated emitted observation logs, not a new native-SDK RAW-prefix decode. Local geometry audit uses current contour and past detector state only.'))
    counts={}
    for case in raw_protocol['cases']:
        current=OUT/'raw_v2'/case;summary=json.loads((current/'summary.json').read_text());observations=traces(current/'observations.jsonl')
        counts[case]=summary['frames'];assert len(observations)==summary['frames']
        cv_reference=None
        for arm in ('frozen','consensus','consensus_audit'):
            records=traces(current/arm/'forecasts.jsonl');assert len(records)==summary['frames']
            cv=[q['states'].get('cv') for q in records]
            if cv_reference is None:cv_reference=cv
            else:assert cv==cv_reference
            for q in records:
                for state in q['states'].values():
                    for key in ('position','velocity','acceleration'):assert np.isfinite(state[key]).all()
                if arm!='frozen' and q.get('normal_measurements') and not all(r['accepted'] for r in q['normal_measurements'].values()):
                    assert not q['physics_active'];assert not q['physical_parameters']['paper_adaptive']['update_allowed']
        assert json.loads((current/'run_manifest.json').read_text())['camera_matrix_supplied'] is False
    assert sum(counts.values())==6600
    raw=json.loads((OUT/'raw_analysis.json').read_text());assert raw['frozen_replays_exact_for']==['train6','train2']
    for case in ('train6','side6'):
        folder=OUT/'raw_v2'/case;cap=cv2.VideoCapture(str(folder/'actual_future_comparison.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,frame=cap.read();cap.release()
        assert ok and n>100;assert (folder/'comparison_frame39.png').exists()
    class LocalLinks(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src','poster') and value and not value.startswith(('https:','http:','#')):assert (OUT/value).exists(),value
    LocalLinks().feed((OUT/'第三阶段实验报告.html').read_text(encoding='utf-8'))
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent),'-p','test_*.py','-v'],capture_output=True,text=True)
    (OUT/'unit_tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8');assert result.returncode==0
    failed=[]
    for case in ('train6','side6','train2'):
        path=OUT/f'raw/{case}/observations.jsonl'
        failed.append(dict(case=case,observations_before_failure=len(path.read_text().splitlines()) if path.exists() else 0))
    save(OUT/'raw/failed_attempt.json',dict(exception='AttributeError: ConsensusAxis has no attribute differential_gain',
        scope='Reporting adapter interface, not geometry formula or tuned parameter.',attempts=failed,excluded_from_accuracy_scores=True))
    save(OUT/'delivery_validation.json',dict(frozen_sources_verified=True,original_runtime_sha256=runtime.hexdigest(),
        raw_frames=counts,total_raw_frames=6600,cv_shared_exact=True,observation_prefix_replay_exact_frames=2600,
        module_prefix_exact_frames=60,new_tests=9,raw_degenerate_measurements_freeze_physics=True,
        fabricated_intrinsics=False,old_failed_runs_retained=True))
    source=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.txt','.toml','.ps1') and not {'__pycache__','.git','.venv'}.intersection(p.relative_to(ROOT).parts)]
    evidence=[p for p in OUT.rglob('*') if p.is_file() and p.suffix.lower() not in ('.raw','.pyc','.npy') and '__pycache__' not in p.parts and p.name!='package_manifest.json']
    target=OUT.parent/'event_drone_forecast_stage3_evidence_20261016.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in source:z.write(p,Path('event_drone_forecast')/p.relative_to(ROOT))
        for p in evidence:z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    data=dict(path=str(target),bytes=target.stat().st_size,sha256=digest(target),source_files=len(source),evidence_files=len(evidence),excluded='Sensor RAW and SDK/dependencies. Input paths and source fingerprints retained.')
    save(OUT/'package_manifest.json',data);print(json.dumps(data),flush=True)

if __name__=='__main__':main()
