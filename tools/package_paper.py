"""Package project plus final reproducible evidence, excluding RAW and SDKs."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/paper_reproduction_v030'
archive=OUT.parent/'event_drone_forecast_v0.3.0.zip'
validation=json.loads((OUT/'delivery_validation.json').read_text());assert validation['records_checked']==35996
sources=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
folders=['runs','evaluation_data','source_snapshot','visuals','figures','controlled_flight','causality_prefix_check']
evidence=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md','.html','.txt')]
for folder in folders:
    evidence.extend(p for p in (OUT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
evidence.extend(p for p in (OUT/'eeppr_benchmark').iterdir() if p.is_file() and (p.suffix=='.json' or p.name.endswith('_window_estimates.npy')))
evidence=[p for p in evidence if p.name!='package_manifest.json']
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(sources):z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    for p in sorted(set(evidence)):z.write(p,Path('output/paper_reproduction_v030')/p.relative_to(OUT))
    z.writestr('START_HERE.txt','Extract to a workspace folder. Open output/paper_reproduction_v030/查看论文复现实验.html.\n'
        'Source: event_drone_forecast/run_paper.py and PAPER_REPRODUCTION.md.\n'
        'Includes all final real-RAW predictions/observations/scores and videos; excludes input RAW, virtual environment, SDK and third-party repositories.\n'
        'Local RAW paths must exist to rerun inference. Past observations and all scores can be inspected without RAW.\n'
        'Optional EEPPR benchmark requires the separately downloaded GPL-3.0 original repository and public data.\n')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    entries=z.namelist()
    assert not any(name.lower().endswith('.raw') for name in entries)
record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    source_files=len(sources),evidence_files=len(set(evidence)),includes_final_full_forecasts=True,
    includes_final_full_observations=True,includes_comparison_videos=True,includes_input_raw=False,
    includes_sdk=False,includes_third_party_source=False,zip_integrity_verified=True)
(OUT/'package_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
