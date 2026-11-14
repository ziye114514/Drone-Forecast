"""Same RAW research runner, bounded layout is the sole frontend adaptation."""
from pathlib import Path
import argparse,hashlib,importlib.util,types

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

pipeline=module('unchanged_raw_runner24',ROOT/'core/run_research_forecast.py')
structure=module('structured_front26',ROOT/'research/stage26/structured26.py')
original_fingerprints=pipeline.fingerprints
pipeline.windows=types.SimpleNamespace(METHODS={'neighbor20':None,'layout20':None},Observer=structure.Observer)


def fingerprints():
    result=original_fingerprints()
    for p in [Path(__file__),ROOT/'research/stage26/structured26.py',ROOT/'research/stage19/layout19.py',ROOT/'research/stage11/identity.py']:
        result[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


pipeline.fingerprints=fingerprints


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--frontend',choices=['neighbor20','layout20'],default='layout20')
    p.add_argument('--primary',choices=pipeline.robust.MODELS,default='trusted_acc');p.add_argument('--rotor-features',action='store_true');p.add_argument('--preview',action='store_true');a=p.parse_args()
    pipeline.run(a.raw,a.output,a.start,a.duration,a.frontend,a.primary,a.rotor_features,a.preview)
