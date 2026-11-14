from pathlib import Path
import argparse,hashlib,json,zipfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();root=Path(__file__).resolve().parents[1]
    archive=out.parent/'event_drone_forecast_v0.2.0.zip'
    sources=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    evidence=['experiment_protocol.json','frozen_config.json','validation.json','unit_test_result.json',
              'aggregate_metrics.csv','version_comparison.json','causality_validation.json','single_process_timing.json',
              '整合结果报告.md','visuals/geometry_synthetic.json','environment.json','unit_test_output.txt']
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(sources):z.write(path,Path(root.name)/path.relative_to(root))
        for name in evidence:
            path=out/name
            if path.exists():z.write(path,Path(root.name)/'experiment_evidence'/name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Corrupt source archive')
    record=dict(archive=str(archive),source_files=len(sources),bytes=archive.stat().st_size,
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                includes_raw=False,includes_venv=False,includes_full_videos=False,
                requirements='Compatible Metavision SDK + project Python dependencies; local data paths are examples.')
    (out/'package_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
