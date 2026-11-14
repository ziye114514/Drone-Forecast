"""Standard-library startup state checks; no processes, SDK, RAW or benchmarks."""
from pathlib import Path
from threading import Condition
import argparse
import ast
import hashlib
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_startup import wait_for_release
from live_process import ProcessReplay


class Flag:
    def __init__(self,value=False):self.value=value
    def is_set(self):return self.value
    def set(self):self.value=True


class Connection:
    def __init__(self):self.messages=[]
    def send(self,message):self.messages.append(message)


class Signal(Flag):
    def __init__(self,stop,cancel=False):super().__init__();self.calls=0;self.stop=stop;self.cancel=cancel
    def wait(self,timeout):
        assert timeout==.05
        self.calls+=1
        if self.cancel:self.stop.set();return False
        if self.calls==2:self.set()
        return self.is_set()


def expect(kind,function):
    try:function()
    except kind:return
    raise AssertionError('Expected '+kind.__name__)


def check_states():
    connection=Connection();stop=Flag()
    assert wait_for_release(connection,None,stop) and not connection.messages
    signal=Signal(stop)
    assert wait_for_release(connection,signal,stop)
    assert connection.messages==[('prepared',None)] and signal.calls==2
    connection=Connection();stop=Flag();signal=Signal(stop,cancel=True)
    assert not wait_for_release(connection,signal,stop)
    assert connection.messages==[('prepared',None)] and signal.calls==1
    connection=Connection();signal=Signal(Flag());stop=Flag(True)
    assert not wait_for_release(connection,signal,stop) and signal.calls==0 and not connection.messages
    source=object.__new__(ProcessReplay)
    source.start_signal=Flag();source.condition=Condition();source.prepared=False
    source.start_released=False;source.done=False;source.error=None
    expect(TimeoutError,lambda:source.wait_prepared(timeout=0))
    expect(RuntimeError,source.release_start)
    assert not source.start_signal.is_set()
    source.prepared=True;source.wait_prepared(timeout=0);source.release_start()
    assert source.start_signal.is_set() and source.start_released
    expect(RuntimeError,source.release_start)
    source.prepared=False;source.done=True
    expect(RuntimeError,lambda:source.wait_prepared(timeout=0))
    source.done=False;source.error=ValueError('prepared failure')
    expect(ValueError,lambda:source.wait_prepared(timeout=0))
    source.start_signal=None
    expect(ValueError,source.release_start)
    expect(ValueError,source.wait_prepared)
    return ['legacy starts without barrier','waits for explicit release','cancellation exits held producer',
        'already cancelled producer sends no prepared message','timeout does not release',
        'release before prepared rejected','release once only','EOF and producer errors propagate']


def check_order(root):
    process=ast.parse((root/'live_process.py').read_text('utf-8-sig'))
    producer=next(n for n in process.body if isinstance(n,ast.FunctionDef) and n.name=='_producer')
    release=next(n for n in ast.walk(producer) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='wait_for_release')
    epoch=next(n for n in ast.walk(producer) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='epoch' for t in n.targets) and isinstance(n.value,ast.Call))
    assert release.lineno<epoch.lineno
    live=ast.parse((root/'run_live_forecast.py').read_text('utf-8-sig'))
    run=next(n for n in live.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    startup=run.args.args.index(next(a for a in run.args.args if a.arg=='startup_sync'))
    defaults=dict(zip([a.arg for a in run.args.args[-len(run.args.defaults):]],run.args.defaults))
    assert isinstance(defaults['startup_sync'],ast.Constant) and defaults['startup_sync'].value=='none'
    lines={}
    for n in ast.walk(run):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('wait_prepared','release_start'):
            lines[n.func.attr]=n.lineno
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='warm_begin' for t in n.targets):lines['warmup']=n.lineno
        if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='packet':lines['consume']=n.lineno
    assert lines['wait_prepared']<lines['warmup']<lines['release_start']<lines['consume']
    return dict(optional_default_none=True,reader_prepared_before_warmup_before_release_before_consumption=True,
        producer_epoch_created_after_release=True)


def core_unchanged():
    archive=ROOT.parent/'output/optimization_run_20261025/source_candidate.zip'
    count=0
    with zipfile.ZipFile(archive) as z:
        for p in (ROOT/'src/event_drone_forecast/position').glob('*.py'):
            suffix=p.relative_to(ROOT).as_posix()
            names=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)]
            assert len(names)==1,names
            assert hashlib.sha256(z.read(names[0])).digest()==hashlib.sha256(p.read_bytes()).digest(),suffix
            count+=1
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    states=check_states();order=check_order(args.runtime_root.resolve())
    unchanged=core_unchanged()
    heavy=[m for m in sys.modules if m.split('.')[0] in ('numpy','scipy','cv2','torch','onnxruntime','metavision_core','metavision_sdk_cv','metavision_sdk_core')]
    assert not heavy,heavy
    report=dict(states=states,order=order,position_files_unchanged_from_validated_source=unchanged,
        heavy_modules_imported=heavy,processes_started=0,RAW_read=False,GPU_started=False,
        timing_measured=False,real_multiprocessing_or_end_to_end_verified=False)
    if args.output:args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
