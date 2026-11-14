"""OpenCV's Kalman engine, existing history fit, no new filtering library."""
from collections import deque
import cv2,numpy as np

PARAMS=dict(position_sigma=7.,acceleration_sigma=1800.,jerk_sigma=9000.,
    reset_gap_s=.5,coast_s=.16,history_s=.35,history_min_s=.15,
    history_min_samples=5,history_acceleration_clip=2000.)
METHODS=('cv','ca_kf','ca_history')

class MotionBank:
    def __init__(self):
        self.filters=None;self.t=None;self.seen=None;self.history=deque()

    def initialize(self,z,t):
        self.filters={}
        for n in (4,6):
            f=cv2.KalmanFilter(n,2,0,cv2.CV_64F)
            H=np.zeros((2,n));H[:,:2]=np.eye(2);f.measurementMatrix=H
            f.measurementNoiseCov=np.eye(2)*PARAMS['position_sigma']**2
            f.errorCovPost=np.diag([PARAMS['position_sigma']**2]*2+[1e6]*(n-2))
            x=np.zeros((n,1));x[:2,0]=z;f.statePost=x;self.filters[n]=f
        self.t=t;self.seen=t;self.history=deque()

    def step(self,t,z=None):
        if z is not None:z=np.asarray(z,float)
        if self.filters is None or (self.seen is not None and t-self.seen>PARAMS['reset_gap_s']):
            if z is None:self.filters=None;return None
            self.initialize(z,t)
        elif t>self.t+1e-12:
            dt=t-self.t
            for n,f in self.filters.items():
                F=np.eye(n);F[:2,2:4]=np.eye(2)*dt
                if n==6:
                    F[:2,4:6]=np.eye(2)*dt*dt/2;F[2:4,4:6]=np.eye(2)*dt
                    b=np.array([dt**3/6,dt*dt/2,dt]);sigma=PARAMS['jerk_sigma']
                else:b=np.array([dt*dt/2,dt]);sigma=PARAMS['acceleration_sigma']
                f.transitionMatrix=F;f.processNoiseCov=np.kron(np.outer(b,b)*sigma**2,np.eye(2));f.predict()
                if z is not None:f.correct(z.reshape(2,1))
            self.t=t
        if z is not None:
            self.seen=t;self.history.append((t,z.copy()))
        while self.history and t-self.history[0][0]>PARAMS['history_s']+1e-10:self.history.popleft()
        if t-self.seen>PARAMS['coast_s']+1e-10:return None
        cv=self.filters[4].statePost[:,0].copy();ca=self.filters[6].statePost[:,0].copy()
        a=np.zeros(2)
        if len(self.history)>=PARAMS['history_min_samples'] and self.history[-1][0]-self.history[0][0]>=PARAMS['history_min_s']:
            times=np.array([u-t for u,z in self.history]);Z=np.array([z for u,z in self.history])
            X=np.c_[np.ones(len(times)),times,times**2/2]
            # Same bounded quadratic history principle used in Stage6. No target labels here.
            beta=np.linalg.solve(X.T@X+np.diag([1e-9,1e-7,1e-6]),X.T@Z)
            a=np.clip(beta[2],-PARAMS['history_acceleration_clip'],PARAMS['history_acceleration_clip'])
        return dict(cv=dict(position=cv[:2].tolist(),velocity=cv[2:4].tolist(),acceleration=[0.,0.]),
            ca_kf=dict(position=ca[:2].tolist(),velocity=ca[2:4].tolist(),acceleration=np.clip(ca[4:6],-2000,2000).tolist()),
            ca_history=dict(position=cv[:2].tolist(),velocity=cv[2:4].tolist(),acceleration=a.tolist()))

def predict(state,dt):
    return np.array(state['position'])+np.array(state['velocity'])*dt+.5*np.array(state['acceleration'])*dt**2

def tests():
    b=MotionBank();records=[]
    for t in np.arange(0,2,.01):records.append(b.step(float(t),[100+80*t,200-20*t]))
    np.testing.assert_allclose(records[-1]['cv']['velocity'],[80,-20],atol=.2)
    np.testing.assert_allclose(records[-1]['ca_history']['acceleration'],[0,0],atol=.1)
    assert b.step(2.3,None) is None
    assert b.step(2.7,None) is None
    assert b.step(2.8,[1,2])['cv']['position']==[1.,2.]
    # Prefix causality: extending the future does not change stored past states.
    c=MotionBank()
    prefix=[c.step(float(t),[100+80*t,200-20*t]) for t in np.arange(0,1,.01)]
    assert prefix==records[:len(prefix)]
    return dict(opencv_engine=cv2.__version__,constant_velocity_recovery=True,history_acceleration_zero=True,
        coast_expiry_and_reinitialization=True,prefix_causality=True)

if __name__=='__main__':print(tests())
