"""Verify frozen evidence, run relevant checks and package without sensor RAW."""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
import cv2,numpy as np
from run_study import OUT,ROOT,save,verify

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=verify()
    for rel,h in p['sources'].items():assert digest(OUT/'source_snapshot'/rel)==h,rel
    runtime=hashlib.sha256()
    for path in sorted((ROOT/'src/event_drone_forecast').glob('*.py')):runtime.update(path.read_bytes())
    assert runtime.hexdigest()=='71b3eb3eb3e4c38acf438a29a757d651a7a7949a42f66b1c95acc64995a615b0'
    complete=json.loads((OUT/'controlled/completion.json').read_text())
    assert complete['protocol_sha256']==digest(OUT/'protocol.json')
    folders=sorted((OUT/'controlled').glob('*/*/seed_*'));trials=0
    for folder in folders:
        with np.load(folder/'clean.npz') as clean:
            prefix=clean['times']<5
            for scenario in p['scenarios']:
                with np.load(folder/f'{scenario}.npz') as a:
                    assert a['prediction'].shape==(701,6,3,2)
                    for key in a.files:assert np.isfinite(a[key]).all(),(folder,scenario,key)
                    np.testing.assert_array_equal(a['prediction'][prefix],clean['prediction'][prefix])
                    np.testing.assert_array_equal(a['parameters'][prefix],clean['parameters'][prefix])
                    if scenario=='id_permutation_only':np.testing.assert_array_equal(a['prediction'],clean['prediction'])
                    if scenario=='clean':np.testing.assert_array_equal(a['prediction'][:,2],a['prediction'][:,5])
                trials+=1
    assert trials==complete['trials']==378
    raw=OUT/'raw_interventions';f=json.loads((raw/'supplemental_code_freeze.json').read_text())
    assert f['raw_script_sha256']==digest(Path(__file__).parent/'raw_interventions.py')==digest(raw/'raw_interventions.py')
    assert f['controlled_protocol_sha256']==digest(OUT/'protocol.json')
    rc=json.loads((raw/'completion.json').read_text());assert rc['future_labels_used'] is False
    for name in rc['conditions']:
        obs=[json.loads(s) for s in (raw/name/'observations.jsonl').read_text().splitlines()]
        assert len(obs)==600 and all(r['removed_events']==0 for r in obs if r['t']>25+1e-8)
        assert sum(r['removed_events'] for r in obs)==rc['conditions'][name]['deleted_events']
        with np.load(raw/name/'forecasts.npz') as a:
            assert a['prediction'].shape==(600,6,3,2)
            for key in a.files:assert np.isfinite(a[key]).all(),(name,key)
    video=cv2.VideoCapture(str(raw/'paired_raw.mp4'));frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT));video.release()
    assert frames==120
    public=OUT/'public_truth_substitutes';pp=json.loads((public/'public_protocol.json').read_text())
    assert pp['script_sha256']==digest(public/'public_baseline_source.py')==digest(Path(__file__).parent/'public_baseline.py')
    run=public/'fred_sequence1/raw_only_baseline';manifest=json.loads((run/'run_manifest.json').read_text())
    assert manifest['source_sha256']==runtime.hexdigest()
    assert len(manifest['inference_external_inputs'])==1 and manifest['input_raw'].endswith('events.raw')
    assert json.loads((public/'public_baseline_summary.json').read_text())['labels_used_in_raw_inference'] is False
    logs=[]
    for folder,pattern,label in [(ROOT/'tests','test_*.py','runtime_tests'),(Path(__file__).parent,'test_mechanism.py','mechanism_tests')]:
        command=[sys.executable,'-m','unittest','discover','-s',str(folder),'-p',pattern,'-v']
        result=subprocess.run(command,capture_output=True,text=True)
        (OUT/f'{label}.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
        assert result.returncode==0,label
        logs.append(dict(group=label,returncode=result.returncode))
    save(OUT/'delivery_validation.json',dict(frozen_sources_verified=True,runtime_sha256=runtime.hexdigest(),
        controlled_trials=trials,raw_conditions=4,raw_video_frames=frames,all_arrays_finite=True,
        no_pre_fault_prediction_difference=True,id_permutation_null_verified=True,raw_restored_after_25s=True,
        tests=logs,public_protocol_verified=True))
    # Store a complete source copy, including the exact report/analysis scripts.
    excluded={'__pycache__','.git','.venv'}
    source_files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.toml','.txt','.ps1') and not excluded.intersection(p.relative_to(ROOT).parts)]
    evidence=[p for p in OUT.rglob('*') if p.is_file() and p.suffix.lower() not in ('.raw','.npy','.pyc') and '__pycache__' not in p.parts and p.name!='package_manifest.json']
    target=OUT.parent/'event_drone_forecast_stage1_evidence_20261016.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in source_files:z.write(path,Path('event_drone_forecast')/path.relative_to(ROOT))
        for path in evidence:z.write(path,Path(OUT.name)/path.relative_to(OUT))
    with zipfile.ZipFile(target,'r') as z:assert z.testzip() is None
    result=dict(path=str(target),bytes=target.stat().st_size,sha256=digest(target),source_files=len(source_files),evidence_files=len(evidence),
        excluded='Sensor RAW, decoded event NPY, SDK and third-party dependency packages. Input paths and fingerprints retained.')
    save(OUT/'package_manifest.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
