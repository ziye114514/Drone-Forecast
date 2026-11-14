"""Lightweight source/CLI contracts; never imports SDK/CUDA or reads RAW.

This is extraction and configuration verification, not a benchmark or an
end-to-end inference test. It is safe to run while workloads are paused.
"""
from pathlib import Path
import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))


def node(module,name):
    return next(n for n in module.body if getattr(n,'name',None)==name)


def dump(value):
    return ast.dump(value,include_attributes=False)


def backend_contract():
    old=tree(ROOT/'research/stage45/graph45.py')
    new=tree(ROOT/'src/event_drone_forecast/detector_backend.py')
    for name in ('preprocess','boxes'):
        assert dump(node(old,name))==dump(node(new,name)),name
    old_params=next(n for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMS' for t in n.targets))
    new_params=next(n for n in new.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMS' for t in n.targets))
    assert dump(old_params)==dump(new_params)
    original=copy.deepcopy(node(old,'Backend'))
    extracted=node(new,'Backend')
    constructor=node(original,'__init__')
    assert [a.arg for a in constructor.args.args]==['self','kind','model_path']
    assert len(constructor.args.defaults)==1 and isinstance(constructor.args.defaults[0],ast.Constant) and constructor.args.defaults[0].value is None
    constructor.args.defaults=[]
    replacements=[]
    class RequireExplicitModel(ast.NodeTransformer):
        def visit_IfExp(self,n):
            if ast.unparse(n)=="model_path if model_path is not None else common44.OUT / 'weights/best.onnx'":
                replacements.append(True)
                return ast.copy_location(ast.Name(id='model_path',ctx=ast.Load()),n)
            return self.generic_visit(n)
    RequireExplicitModel().visit(original)
    assert len(replacements)==1
    assert dump(original)==dump(extracted),'Backend changed beyond removal of implicit historical weights'
    assert 'common44' not in (ROOT/'src/event_drone_forecast/detector_backend.py').read_text('utf-8')
    return dict(preprocess_AST_exact=True,NMS_AST_exact=True,ORT_backend_AST_exact_except_required_model=True,parameters_exact=True)


def worker_contract():
    old=node(tree(ROOT/'research/stage47/worker47.py'),'main')
    new=node(tree(ROOT/'live_worker.py'),'main')
    def request_loop(function):
        return next(n for n in ast.walk(function) if isinstance(n,ast.For)
                    and isinstance(n.iter,ast.Attribute) and ast.unparse(n.iter)=='sys.stdin')
    loop=copy.deepcopy(request_loop(old))
    class RenameBackend(ast.NodeTransformer):
        def visit_Name(self,n):
            if n.id=='graph45':n.id='backend_module'
            return n
    RenameBackend().visit(loop)
    assert dump(loop)==dump(request_loop(new)),'GPU request/NMS/association protocol changed'
    imports=[]
    for n in ast.walk(tree(ROOT/'live_worker.py')):
        if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
        if isinstance(n,ast.ImportFrom):imports.append(n.module)
    assert not any(n and n.startswith(('graph45','common44','common41')) for n in imports)
    return dict(request_loop_AST_exact_except_module_alias=True,historical_benchmark_imports=False)


def launcher_contract(root):
    # All imports and subprocess calls are standard-library only. A zero-byte
    # local marker checks file-path validation; its contents are never opened.
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('lightweight_forecast_contract',root/'forecast.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='edf-cli-contract-') as tmp:
        d=Path(tmp);raw=d/'marker.raw';raw.write_bytes(b'')
        config=d/'config.json';config.write_text('{}',encoding='utf-8')
        base=['run','--raw',str(raw),'--output',str(d/'not-created'),'--config',str(config),'--dry-run']
        runtime=dict(gpu_python=sys.executable,detector_weights=str(d/'not-read.onnx'),detector_sha256='0'*64,cache_dir=str(d/'cache'))
        def invoke(extra=(),cfg=None,restricted=False):
            config.write_text(json.dumps(cfg or {}),encoding='utf-8')
            stdout=io.StringIO();stderr=io.StringIO()
            with patch.object(launcher,'resolve',return_value=runtime),patch.object(launcher.subprocess,'call',side_effect=AssertionError('Subprocess forbidden')),contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                code=launcher.main(base+list(extra),**(dict(allowed_profiles=('live',),default_profile='live',require_config=True) if restricted else {}))
            return code,stdout.getvalue(),stderr.getvalue()
        code,out,_=invoke(['--profile','live'])
        assert code==0 and json.loads(out)['raw_events_read'] is False
        cmd=json.loads(out)['argv'];assert cmd[3]==str(root/'run_live_forecast.py')
        cases.append('live dry-run starts no subprocess')
        code,out,_=invoke(restricted=True)
        assert code==0 and json.loads(out)['argv'][3]==str(root/'run_live_forecast.py')
        cases.append('restricted launcher defaults to live')
        invalid=[(['--profile','live','--producer','thread','--frame-transport','frames'],{},'requires --producer process'),
                 (['--profile','live','--tracker-hz','51'],{},'<= 50'),
                 (['--profile','live','--tracker-hz','5','--publish-hz','10'],{},'<= 50'),
                 (['--profile','live','--tracker-hz','nan'],{},'positive finite'),
                 (['--profile','live'],dict(producer='thread',frame_transport='frames'),'requires --producer process'),
                 (['--profile','live','--producer','thread','--startup-sync','reader-ready'],{},'requires --producer process')]
        for extra,cfg,message in invalid:
            code,_,err=invoke(extra,cfg)
            assert code==2 and message in err,(extra,err)
            cases.append('rejected '+repr(extra or cfg))
        code,_,err=invoke(cfg=dict(profile='research'),restricted=True)
        assert code==2 and 'Profile not available' in err
        cases.append('restricted config cannot select missing research entry')
        assert not (d/'not-created').exists() and not (d/'cache').exists()
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    backend=backend_contract()
    worker=worker_contract()
    cases=launcher_contract(args.runtime_root.resolve())
    prohibited=[m for m in sys.modules if m.split('.')[0] in ('numpy','torch','cv2','scipy','onnxruntime','ultralytics','metavision_core','metavision_sdk_core','metavision_sdk_cv')]
    assert not prohibited,prohibited
    report=dict(backend=backend,worker=worker,launcher_cases=cases,heavy_modules_imported=prohibited,
        RAW_events_read=False,GPU_started=False,timing_measured=False,
        full_RAW_inference_tested=False)
    if args.output:args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
