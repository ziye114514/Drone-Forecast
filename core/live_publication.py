"""Independent fixed-deadline publication from an immutable latest estimate.

Publication continues with explicit invalid heartbeats during tracking gaps.
It never consumes an estimate before that estimate was made available. Missed
deadlines are skipped; a late callback cannot cause a burst of catchup output.
"""
from threading import Event,Lock,Thread
import math,time


class TimedPublisher:
    def __init__(self,source,duration,hz,callback):
        if hz<=0 or duration<=0:raise ValueError('Positive duration/frequency required')
        self.source=source;self.duration=duration;self.hz=hz;self.period=1./hz
        self.callback=callback;self.lock=Lock();self.stop=Event();self.state=None
        self.thread=None;self.error=None;self.missed=0;self.published=0
        self.lateness_ms=[]

    def start(self):
        self.thread=Thread(target=self._run,name='forecast-fixed-deadlines')
        self.thread.start()

    def __enter__(self):
        self.start();return self

    def __exit__(self,kind,value,tb):
        if kind is None:self.finish()
        else:self.abort()

    def update(self,state):
        # Caller must not mutate this dictionary or its arrays afterwards.
        with self.lock:self.state=state

    def _run(self):
        try:
            while self.source.epoch is None:
                if self.stop.wait(.005):return
                if getattr(self.source,'done',False):return
            last_wall=None
            slots=math.floor(self.duration*self.hz+1e-8)
            for index in range(1,slots+1):
                deadline=self.source.epoch+index*self.period
                if self.stop.wait(max(0.,deadline-time.perf_counter())):return
                with self.lock:
                    state=self.state
                    now=time.perf_counter()
                if now-deadline>=self.period or (last_wall is not None and now-last_wall<self.period*.5):
                    self.missed+=1;continue
                self.callback(state,now,index,self.source.start+index*self.period)
                self.lateness_ms.append(max(0.,now-deadline)*1000)
                self.published+=1;last_wall=now
        except BaseException as error:
            self.error=error

    def finish(self):
        if self.thread is not None:
            self.thread.join(timeout=self.period+15.)
            if self.thread.is_alive():self.abort();raise RuntimeError('Publisher did not finish')
        if self.error is not None:raise self.error

    def abort(self):
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=15)
