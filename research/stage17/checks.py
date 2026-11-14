"""Known-frequency/mono-polarity/causality checks before real-data scoring."""
import json
import cv2,numpy as np
from frontend import Frontend,SparseSpectrum,PARAMS

DTYPE=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])


def frequency_events(hz):
    n=400;t=(np.arange(n)+.5)*250
    number=np.rint(8+7*np.sin(2*np.pi*hz*t/1e6)).astype(int)
    e=np.zeros(int(number.sum()),DTYPE);e['x']=12;e['y']=12;e['p']=1;e['t']=np.repeat(t.astype('i8'),number)
    return e


def rotor_events(duration=.5,frequency=300.):
    mask=np.zeros((240,320),np.uint8)
    for c in [(90,70),(230,70),(90,170),(230,170)]:cv2.ellipse(mask,c,(22,7),12,0,360,1,-1)
    yy,xx=np.nonzero(mask);stamps=np.arange(1000,duration*1e6,1e6/frequency).astype('i8')
    e=np.zeros(len(xx)*len(stamps),DTYPE);e['x']=np.tile(xx,len(stamps));e['y']=np.tile(yy,len(stamps));e['p']=1;e['t']=np.repeat(stamps,len(xx))
    return e


def run():
    s=SparseSpectrum(32,32);rows=[]
    for frequency in [200,350,600,1200,1600]:
        r=s.step(frequency_events(frequency),.1);estimated=float(r['hz'][1,1])
        assert abs(estimated-frequency)<=10.,(frequency,estimated)
        rows.append(dict(true_hz=frequency,estimated_hz=estimated))
    try:SparseSpectrum(32,32,1000)
    except ValueError:pass
    else:raise AssertionError('Invalid Nyquist configuration not rejected')
    e=rotor_events();a=Frontend(320,240,'global');b=Frontend(320,240,'roi');results=[]
    future=e[e['t']>=300000]
    twin=Frontend(320,240,'roi')
    for t in [.10,.12,.14,.16,.18,.20]:
        hi=round(t*1e6);history=e[(e['t']>=hi-100000)&(e['t']<hi)];current=e[(e['t']>=hi-20000)&(e['t']<hi)]
        x=a.step(history,current,t);y=b.step(history,current,t)
        z=twin.step(np.concatenate([history,future]),current,t)
        assert y['observation']==z['observation'],t
        assert x['observation']==y['observation'],t
        results.append(y)
    assert not results[0]['observation']['detected'] and not results[1]['observation']['detected']
    assert any(r['observation']['detected'] and len(r['observation']['rotors'])==4 for r in results)
    assert any(r['search']=='roi' for r in results)
    empty=np.empty(0,DTYPE)
    for t in np.arange(.22,.44,.02):
        q=b.step(empty,empty,float(round(t,2)));assert not q['observation']['detected']
    assert q['search']=='global'
    return dict(known_frequencies=rows,nyquist_guard=True,monopolar_four_disks_detected=True,
        confirmation_required=True,roi_equals_global_on_stationary_target=True,
        future_events_do_not_change_output=True,empty_input_not_published=True,loss_returns_to_global=True,
        limits='Synthetic event truth only; does not establish real rotor/RPM/pose accuracy.')


if __name__=='__main__':print(json.dumps(run(),indent=2))
