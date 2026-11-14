"""Check final source provenance, causal results, finite states and media."""
from pathlib import Path
import argparse,hashlib,json,re
import cv2,numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();root=Path(__file__).resolve().parents[1]
    protocol=json.loads((out/'experiment_protocol.json').read_text(encoding='utf-8'))
    source={name:hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in protocol['source_files'].items()}
    runs={};total=0
    def invalid(value):raise ValueError('Non-finite JSON number: '+value)
    for sid in (6,2,3,4):
        folder=out/'runs'/f'train{sid}'
        metadata=json.loads((folder/'run_manifest.json').read_text())
        traces=[json.loads(s,parse_constant=invalid) for s in (folder/'forecasts.jsonl').read_text().splitlines()]
        observations=[json.loads(s,parse_constant=invalid) for s in (folder/'observations.jsonl').read_text().splitlines()]
        active=[r for r in traces if r['active']]
        finite=all(np.isfinite([v for state in r['states'].values() for values in state.values() for v in values]).all() for r in active)
        evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
        matched=all(len({r['available'] for r in evaluation['metrics'] if r['horizon_s']==h})==1 for h in (.1,.2,.4))
        source_hash=hashlib.sha256()
        for path in sorted((root/'src/event_drone_forecast').glob('*.py')):source_hash.update(path.read_bytes())
        axis_only_honest=all(o.get('tilt_proxy_deg') is None for o in observations if o['detected'] and all(r['axis_only'] for r in o['rotors']))
        runs[f'train{sid}']=dict(frames=len(traces),source_matches=metadata['source_sha256']==source_hash.hexdigest(),
                                config_matches=(folder/'config.json').read_bytes()==(out/'frozen_config.json').read_bytes(),
                                inference_raw_only=metadata['inference_external_inputs']==[metadata['input_raw']],
                                finite=finite,common_method_coverage=matched,line_mode_no_false_tilt=axis_only_honest,
                                frames_aligned=len(traces)==len(observations))
        total+=len(traces)
    media={}
    for name in ('train6_close','train3','train6'):
        folder=out/'visuals'/name;cap=cv2.VideoCapture(str(folder/'current_vs_future.mp4'))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks=[]
        for index in (0,count//2,max(0,count-1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();checks.append(bool(ok and frame.shape==(600,1600,3)))
        cap.release()
        with Image.open(folder/'current_vs_future.gif') as gif:gif_count=gif.n_frames
        media[name]=dict(video_frames=count,keyframes_decoded=checks,gif_frames=gif_count)
    page=(out/'查看整合结果.html').read_text(encoding='utf-8')
    links=re.findall(r'(?:src|href)="([^"#]+)"',page)
    missing=[p for p in links if p!='validation.json' and not p.startswith(('http:','https:')) and not (out/p).exists()]
    causality=json.loads((out/'causality_validation.json').read_text())
    unit=json.loads((out/'unit_test_result.json').read_text())
    test_hashes_match=all(hashlib.sha256((root.parent/name).read_bytes()).hexdigest()==digest for name,digest in unit['test_file_sha256'].items())
    unit_matches=test_hashes_match and unit['passed']==16 and unit['failed']==0 and len(unit['test_names'])==16
    result=dict(source_matches=source,runs=runs,published_frames=total,media=media,missing_report_links=missing,causality=causality,
                unit_tests_record=dict(passed=unit['passed'],failed=unit['failed'],test_sources_match=test_hashes_match))
    result['passed']=(all(source.values()) and not missing and all(all(value for key,value in r.items() if key!='frames') for r in runs.values())
                      and all(all(r['keyframes_decoded']) and r['gif_frames']>10 for r in media.values())
                      and causality['observations_exactly_equal'] and causality['forecasts_exactly_equal'] and unit_matches)
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit('Final validation failed')


if __name__=='__main__':main()
