"""Scoped collection timing only; never changes GC or filter behavior."""
import gc,time,threading

class GCTrace:
    def __init__(self):self.events=[];self.pending={}
    def callback(self,phase,info):
        generation=info['generation'];thread=threading.get_ident();key=(thread,generation)
        if phase=='start':self.pending[key]=time.perf_counter()
        else:
            begin=self.pending.pop(key,None)
            if begin is not None:self.events.append(dict(generation=generation,thread=thread,
                start_wall=begin,elapsed_ms=1000*(time.perf_counter()-begin),collected=info.get('collected')))
    def start(self):gc.callbacks.append(self.callback)
    def stop(self):gc.callbacks.remove(self.callback)
