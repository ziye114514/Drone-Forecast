"""Preserve failed setup; constrain artifact substitution to constructor only."""
import argparse,datetime,importlib.util,traceback
from backend46 import *
from scoped_backend46 import make_backend
def supplement():
    path=OUT/'path_scope_supplement.json'
    hashes={n:sha(Path(__file__).with_name(n)) for n in ['pilot_repair46.py','scoped_backend46.py']}
    if not path.exists():save(path,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_hashes=hashes,reason='Initial46 benchmark recorded fp16_graph setup failure before inference: broad model-path text occurs in both original45 verification and constructor, so unique-anchor assertion stopped before loading FP16. Preserve benchmark.json and frozen sources. Substitute only unique str(path) constructor argument; same model and graph options. Write new benchmark_scoped_path.json, no threshold/precision tuning.'))
    else:assert load(path)['helper_hashes']==hashes
def benchmark():
    verify();supplement();assert not (OUT/'benchmark_scoped_path.json').exists();source=(ROOT/'research/stage45/benchmark45.py').read_text(encoding='utf-8')
    changes=[("for kind in ['run','binding','graph']:","for kind in ['fp32_graph','fp16_graph']:"),("OUT/'benchmark.json'","OUT/'benchmark_scoped_path.json'"),("failures[kind]=str(e)","failures[kind]=repr(e)+'\\n'+traceback.format_exc()")]
    for a,b in changes:assert source.count(a)==1;source=source.replace(a,b)
    namespace={'__name__':'scoped_precision46','__file__':str(ROOT/'research/stage45/benchmark45.py'),'traceback':traceback};exec(compile(source,namespace['__file__'],'exec'),namespace);namespace.update(OUT=OUT,verify=verify,Backend=lambda kind:graph45.Backend('graph') if kind=='fp32_graph' else make_backend('graph'),boxes=finite_boxes)
    import torch
    with torch.inference_mode():namespace['main']()
def sequence_action(action):
    supplement();path=Path(__file__).with_name('study46.py');spec=importlib.util.spec_from_file_location('scoped_sequence46',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.fp16_backend=make_backend
    if action=='freeze_sequence':
        module.freeze_sequence();p=load(OUT/'sequence/protocol.json');p['source_hashes'].update({'research/stage46/'+n:v for n,v in load(OUT/'path_scope_supplement.json')['helper_hashes'].items()});p['artifact_factory']='scoped_backend46.make_backend, unique constructor only';save(OUT/'sequence/protocol.json',p)
    else:module.sequence_action(action)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['benchmark','freeze_sequence','infer','replay','score']);a=p.parse_args()
    if a.action=='benchmark':benchmark()
    else:sequence_action(a.action)
