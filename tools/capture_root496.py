"""Exact487 stream plus native evidence around the diagnosed early-error case."""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from observed_rotor_predictor import LocalContrastObservedPredictor
from run_rotor_learned_candidate import main
OUT=ROOT.parent/'output/native_root496_20261112'

class Capture(LocalContrastObservedPredictor):
    def step(self,events,t):
        r,f=super().step(events,t)
        if 11.6<=t<=12.15:
            x,y,X,Y=880,330,1280,480
            np.savez_compressed(OUT/f'{round(t*1e6)}.npz',
                on=f['on'][y:Y,x:X].copy(),off=f['off'][y:Y,x:X].copy(),
                native=f['native_mask'][y:Y,x:X].copy(),period=f['period_mask'][y:Y,x:X].copy(),
                density=f['density'][y:Y,x:X].copy(),events=events.copy())
        return r,f

if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    sys.argv=[sys.argv[0],'--raw','E:/dataset/experiment/4/media/event.raw','--start','10','--duration','6',
        '--output',str(OUT/'baseline'),'--gpu-python','C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',
        '--rotor-native-geometry-dll',str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')]
    main(predictor_type=Capture,accuracy_first=True)
    rows=[json.loads(s) for s in (OUT/'baseline/predictions.jsonl').read_text('utf8').splitlines()]
    old=[json.loads(s) for s in (ROOT.parent/'output/raw_local_contrast487_20261112/4_local_contrast/predictions.jsonl').read_text('utf8').splitlines()]
    from physical_publication import evaluate
    delta=max(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(rows,old) if r['valid'])
    assert delta<1e-7,(len(rows),delta)
    (OUT/'parity.json').write_text(json.dumps(dict(frames=len(rows),maximum_prediction_difference=float(delta))), 'utf8')
    print('EXACT BASELINE PARITY',delta,flush=True)
