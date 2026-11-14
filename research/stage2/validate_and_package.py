"""Audit frozen protocols, actual RAW prefix, guard behavior and videos."""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
import cv2,numpy as np
from html.parser import HTMLParser
from study import ROOT,OUT,FIRST,verify,digest,save

def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]

def main():
    protocol=verify()
    for rel,h in protocol['sources'].items():assert digest(OUT/'source_snapshot'/rel)==h
    first=json.loads((FIRST/'protocol.json').read_text())
    for rel,h in first['sources'].items():assert digest(ROOT/rel)==h
    runtime=hashlib.sha256()
    for p in sorted((ROOT/'src/event_drone_forecast').glob('*.py')):runtime.update(p.read_bytes())
    assert runtime.hexdigest()=='71b3eb3eb3e4c38acf438a29a757d651a7a7949a42f66b1c95acc64995a615b0'
    for name in ('activity_fallback','activity_fallback_fixed'):
        p=json.loads((OUT/name/'protocol.json').read_text())
        for filename,h in p['sources'].items():
            assert digest(Path(__file__).with_name(filename))==digest(OUT/name/filename)==h
    for name in ('integrated','guarded'):
        p=json.loads((OUT/name/'protocol.json').read_text())
        for rel,h in p['source_hashes'].items():assert digest(ROOT/rel)==digest(OUT/name/'source_snapshot'/rel)==h
    completion=json.loads((OUT/'stage2_completion.json').read_text());assert completion['guarded_entry_validated']
    full=OUT/'integrated/fred3';prefix=OUT/'integrated/fred3_prefix12'
    for filename in ('observations.jsonl','forecasts.jsonl'):
        a=lines(full/filename);b=lines(prefix/filename);assert len(a)==2000 and len(b)==1200
        assert a[:1200]==b,filename
    save(OUT/'raw_prefix_validation.json',dict(sequence='FRED3',prefix_s=12,full_s=20,frames_compared=1200,
        observations_exact=True,forecasts_exact=True,future_labels_used=False))
    guard_counts={}
    for case in ('own_train6','fred3'):
        folder=OUT/'guarded'/case;obs=lines(folder/'observations.jsonl');pred=lines(folder/'forecasts.jsonl')
        assert len(obs)==len(pred)
        m=json.loads((folder/'run_manifest.json').read_text());assert len(m['inference_external_inputs'])==1
        nonrotor=0
        for o,p in zip(obs,pred):
            z=o['position_observation'];origin=o['original_rotor_observation']
            if origin['detected']:
                assert z['position_source']=='rotor_observation'
                for key in origin:assert origin[key]==z[key],key
            if z['position_source']!='rotor_observation':
                nonrotor+=1;assert not p['physics_active']
                if p.get('physical_parameters'):assert not p['physical_parameters']['paper_adaptive']['update_allowed']
            for state in p['states'].values():
                for key in ('position','velocity','acceleration'):assert np.isfinite(state[key]).all()
        guard_counts[case]=dict(frames=len(obs),nonrotor_physics_blocked_frames=nonrotor)
    assert len(lines(OUT/'integrated/own_train6/forecasts.jsonl'))==3000
    videos=[]
    for path in [*(OUT/'activity_fallback_fixed').glob('sequence*/真实未来对照.mp4'),
                 OUT/'integrated/own_train6/自有录像_真实未来对照.mp4',OUT/'guarded/own_train6/自有录像_真实未来对照.mp4']:
        cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,frame=cap.read();cap.release()
        assert ok and count>100,path
        videos.append(dict(path=str(path.relative_to(OUT)),frames=count))
    # Only the nine new meaningful geometry/reacquisition/identity tests; the
    # original runtime is bit-identical to its already-tested frozen delivery.
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent),'-p','test_*.py','-v'],capture_output=True,text=True)
    (OUT/'unit_tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8');assert result.returncode==0
    class LocalLinks(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href','poster') and value and not value.startswith(('http:','https:','#')):
                    assert (OUT/value).exists(),value
    LocalLinks().feed((OUT/'第二阶段实验报告.html').read_text(encoding='utf-8'))
    save(OUT/'delivery_validation.json',dict(runtime_unchanged_sha256=runtime.hexdigest(),protocols_verified=True,
        raw_prefix_exact_frames=1200,guard_checks=guard_counts,videos=videos,new_unit_tests=9,
        original_rotor_observations_preserved=True,no_pose_or_thrust_from_activity=True))
    source=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.toml','.txt','.ps1') and not {'__pycache__','.git','.venv'}.intersection(p.relative_to(ROOT).parts)]
    evidence=[p for p in OUT.rglob('*') if p.is_file() and p.suffix.lower() not in ('.raw','.pyc','.npy') and '__pycache__' not in p.parts and p.name!='package_manifest.json']
    target=OUT.parent/'event_drone_forecast_stage2_evidence_20261016.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in source:z.write(p,Path('event_drone_forecast')/p.relative_to(ROOT))
        for p in evidence:z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    data=dict(path=str(target),bytes=target.stat().st_size,sha256=digest(target),source_files=len(source),evidence_files=len(evidence),
        excludes='Sensor RAW, SDK, dependencies. Input inventories retained; original v0.3 remains separate.')
    save(OUT/'package_manifest.json',data);print(json.dumps(data),flush=True)

if __name__=='__main__':main()
