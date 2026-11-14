"""Validate frozen provenance, every published record and delivered media."""
from pathlib import Path
from itertools import zip_longest
from html.parser import HTMLParser
import hashlib,json,re,sys
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/paper_reproduction_v030'
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.motion import extrapolate
from paper_experiment import source_hash
protocol=json.loads((OUT/'experiment_protocol.json').read_text(encoding='utf-8'))
assert source_hash()==protocol['source_sha256'],'Runtime changed since freeze'
for rel,digest in protocol['source_files'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    assert hashlib.sha256((OUT/'source_snapshot'/rel).read_bytes()).hexdigest()==digest,rel
records=0;runs=[]
for run in sorted((OUT/'runs').iterdir()):
    meta=json.loads((run/'run_manifest.json').read_text());summary=json.loads((run/'summary.json').read_text())
    assert meta['source_sha256']==protocol['source_sha256']
    assert meta['inference_external_inputs']==[meta['input_raw']]
    count=0;previous=-1
    with (run/'observations.jsonl').open() as obs,(run/'forecasts.jsonl').open() as forecasts:
        for oline,qline in zip_longest(obs,forecasts):
            assert oline and qline
            o=json.loads(oline);q=json.loads(qline);count+=1
            assert o['t']==q['t'] and o['t']>previous;previous=o['t']
            for rotor in o.get('rotors',[]):
                if rotor['frequency_valid']:assert 60<=rotor['blade_pass_hz']<=1200
            if q.get('active'):
                assert len(q['states'])==8
                for state in q['states'].values():
                    assert np.isfinite(np.r_[state['position'],state['velocity'],state['acceleration']]).all()
                    if 'rollout_positions' in state:
                        assert np.isfinite(state['rollout_positions']).all()
                        np.testing.assert_allclose(extrapolate(state,0),state['position'],atol=1e-8)
                for n in q.get('thrust_axis',{}).values():
                    if n is not None:assert abs(np.linalg.norm(n)-1)<1e-8
    assert count==summary['frames'];records+=count;runs.append(dict(sequence=run.name,frames=count))
causal=json.loads((OUT/'causality_validation.json').read_text());assert all(v['exactly_equal'] for v in causal.values())
io=json.loads((OUT/'eeppr_benchmark/sdk_bridge_validation.json').read_text());assert all(v['iterator_exactly_equal'] and v['roi_exactly_equal'] for v in io)
testlog=(OUT/'unit_tests.txt').read_text(encoding='utf-8');assert re.search(r'Ran 25 tests',testlog) and '\nOK' in testlog
media=[]
for folder in ('four_rotors','side_view'):
    path=OUT/f'visuals/{folder}';info=json.loads((path/'visualization_manifest.json').read_text())
    capture=cv2.VideoCapture(str(path/'current_vs_future.mp4'));assert capture.isOpened()
    frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));assert frames==info['video_frames'];capture.release()
    with Image.open(path/'current_vs_future.gif') as im:assert im.n_frames==info['gif_frames']
    media.append(dict(clip=folder,video_frames=frames,gif_frames=info['gif_frames']))
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src','poster') and not value.startswith(('https:','http:','#')):self.links.append(value)
parser=Links();parser.feed((OUT/'查看论文复现实验.html').read_text(encoding='utf-8'))
for link in parser.links:
    if link.endswith('.zip'):continue # Created after this first validation.
    assert (OUT/link).is_file(),link
result=dict(runtime_sha256=source_hash(),source_snapshot_matches=True,records_checked=records,runs=runs,
            all_record_timestamps_monotonic=True,all_active_states_finite=True,unit_tests=25,
            raw_prefix_exact=True,eeppr_bridge_exact=True,media=media,local_report_links_checked=len(parser.links))
(OUT/'delivery_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
