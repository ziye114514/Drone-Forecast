from pathlib import Path
import sys,unittest,json,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.config import Config
from event_drone_forecast.motion import KinematicFilter,ForecastBank,extrapolate
from event_drone_forecast.perception import RotorPerception

class ForecastTests(unittest.TestCase):
    def observation(self,t,center):
        return dict(t=t,detected=True,center=list(center),confidence=.95,
                    normal_xy=[.2,.7],frequency_squared_relative=1.2,frequency_quality=.8)

    def test_constant_velocity_has_correct_future_units(self):
        f=KinematicFilter(Config(),2)
        for t in np.arange(0,3,.02):x=f.update(float(t),[100+40*t,200-15*t])
        self.assertLess(np.linalg.norm(x[2:]-[40,-15]),.05)
        pred=extrapolate(dict(position=x[:2],velocity=x[2:],acceleration=[0,0]),.4)
        self.assertLess(np.linalg.norm(pred-[100+40*(t+.4),200-15*(t+.4)]),.05)

    def test_constant_acceleration_filter(self):
        f=KinematicFilter(Config(),3)
        for t in np.arange(0,4,.02):x=f.update(float(t),[30+5*t+20*t*t,100+2*t])
        self.assertLess(np.linalg.norm(x[4:]-[40,0]),.5)

    def test_prefix_invariant_under_future_changes(self):
        histories=[]
        for change in (False,True):
            b=ForecastBank(Config());results=[]
            for i in range(180):
                t=i*.02
                center=[100+40*t,200+10*np.sin(t)]
                if change and i>=100:center=[1000-200*t,600]
                results.append(b.step(self.observation(t,center)))
            histories.append(results)
        self.assertEqual(json.dumps(histories[0][:100]),json.dumps(histories[1][:100]))
        self.assertNotEqual(json.dumps(histories[0][150]),json.dumps(histories[1][150]))

    def test_missing_measurements_expire_forecast(self):
        b=ForecastBank(Config())
        for i in range(50):b.step(self.observation(i*.02,[100+i,200]))
        r=b.step(dict(t=1.3,detected=False,confidence=0))
        self.assertFalse(r['active'])
        self.assertIsNone(b.step(dict(t=2.,detected=False,confidence=0)))

    def test_online_updates_use_only_completed_delay(self):
        b=ForecastBank(Config())
        for i in range(25):r=b.step(self.observation(i*.02,[100+i,200]))
        self.assertEqual(r['online_updates']['adaptive_rotor'],0)
        for i in range(25,100):r=b.step(self.observation(i*.02,[100+i,200]))
        self.assertGreater(r['online_updates']['adaptive_rotor'],30)

class PerceptionTests(unittest.TestCase):
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])

    def test_empty_input_has_no_target(self):
        p=RotorPerception(1280,720,Config())
        r,_=p.detect(np.empty(0,dtype=self.dtype),.02)
        self.assertFalse(r['detected'])

    def test_four_repetitive_disks_auto_detect_without_box(self):
        rng=np.random.default_rng(12);chunks=[]
        for cx,cy in [(400,280),(600,280),(420,370),(620,370)]:
            yy,xx=np.mgrid[cy-18:cy+19,cx-55:cx+56]
            inside=((xx-cx)/55)**2+((yy-cy)/18)**2<=1
            x,y=xx[inside],yy[inside]
            e=np.zeros(len(x)*12,dtype=self.dtype)
            e['x']=np.repeat(x,12);e['y']=np.repeat(y,12)
            e['p']=np.tile(np.arange(12)%2,len(x));e['t']=rng.integers(0,20000,len(e));chunks.append(e)
        e=np.concatenate(chunks);e.sort(order='t')
        r,_=RotorPerception(1280,720,Config()).detect(e,.02)
        self.assertTrue(r['detected']);self.assertEqual(len(r['rotors']),4)
        self.assertLess(abs(r['center'][0]-510),15)

    def test_known_periodic_signal(self):
        t=np.arange(0,100000,100)
        counts=(15+12*np.cos(2*np.pi*300*t/1e6)).astype(int)
        e=np.zeros(int(counts.sum()),dtype=self.dtype)
        e['t']=np.repeat(t,counts);e['x']=510;e['y']=490;e['p']=np.arange(len(e))%2
        p=RotorPerception(1280,720,Config());p.history.append((.1,e))
        frequencies,quality=p._frequency([dict(center=np.array([500,500]),major=100)],.1)
        self.assertAlmostEqual(frequencies[0],300,delta=10)
        self.assertGreater(quality,.5)

if __name__=='__main__':unittest.main()
