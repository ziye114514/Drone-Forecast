"""Publish fixed-view fused detections in a truth-inaccessible process before scoring."""
from pathlib import Path
import argparse,collections,datetime,json,os,sys
from common38 import ROOT,OUT,load,save,sha,verify


def freeze():
    verify();assert not (OUT/'predictions.jsonl').exists() and not (OUT/'fusion_bridge.json').exists()
    save(OUT/'fusion_bridge.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),core_score_sha256=sha(Path(__file__).with_name('score38.py')),
        protocol_sha256=sha(OUT/'protocol.json'),purpose='Execute EXACT frozen score38.fuse with predictions alone BEFORE any target read. All40 fused output records later must match scorer recomputation. No algorithm/config change.'))


def run():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.name in ['targets.json','control_targets.json'] or p.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()) or 'freeze_v1_before_inference' in p.parts:
                raise RuntimeError('Truth cannot be read by detector fusion')
    sys.addaudithook(guard)
    verify();bridge=load(OUT/'fusion_bridge.json');assert sha(Path(__file__))==bridge['helper_sha256'] and sha(Path(__file__).with_name('score38.py'))==bridge['core_score_sha256']
    from score38 import fuse
    assert not (OUT/'fused_predictions.jsonl').exists();assert sha(OUT/'predictions.jsonl')==load(OUT/'inference.json')['predictions_sha256']
    groups=collections.defaultdict(list);cases={}
    for r in map(json.loads,(OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()):groups[r['id']].extend(r['boxes']);cases[r['id']]=r['case']
    with (OUT/'fused_predictions.jsonl').open('x',encoding='utf-8') as f:
        for ident,bb in groups.items():f.write(json.dumps(dict(id=ident,case=cases[ident],method='fused_all',boxes=fuse(bb)),allow_nan=False)+'\n')
    save(OUT/'fusion_completion.json',dict(frames=len(groups),fused_sha256=sha(OUT/'fused_predictions.jsonl'),ground_truth_access_guard=True,labels_read=False))
    print('Fused all fixed views before scoring: '+str(len(groups)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
