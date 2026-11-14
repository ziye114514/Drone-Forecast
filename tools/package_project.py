from pathlib import Path
import zipfile,json,hashlib

project=Path(__file__).resolve().parents[1]
out=project.parent/'output'
archive=out/'event_drone_forecast_v0.1.0.zip'
files=[p for p in project.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files):z.write(path,Path(project.name)/path.relative_to(project))
    for name in ('frozen_config.json','experiment_protocol.json','validation.json','aggregate_metrics.csv','实验结果与创新方向.md','causality_validation.json','single_process_timing.json','raw_fingerprints.json'):
        path=out/'raw_only_project'/name
        if path.exists():z.write(path,Path(project.name)/'experiment_evidence'/name)
with zipfile.ZipFile(archive) as z:
    bad=z.testzip()
    assert bad is None
record=dict(archive=str(archive),source_file_count=len(files),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            includes_raw_data=False,includes_venv=False,sdk_requirement='Install the compatible Metavision SDK separately.')
(out/'raw_only_project/package_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))
