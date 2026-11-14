"""Workspace-owned SDK extension: yield when the HAL queue is empty.

poll_buffer()==0 means no raw buffer, per the installed official HAL header.
Decode and end-of-stream/flush behavior remain the original SDK method. No
installed SDK file is modified and no events or time bins are skipped here.
"""
import time
from metavision_core.event_io.raw_reader import RawReaderBase
from metavision_core.event_io import EventsIterator


class CooperativeRawReader(RawReaderBase):
    def _run(self):
        if self._decode_done:return False
        if self.i_events_stream.poll_buffer()==0:
            # A short explicit yield releases Python's GIL while the native
            # producer has nothing to deliver; avoid decoding empty buffers.
            time.sleep(.0005)
            return True
        return super()._run()


class CooperativeEventsIterator(EventsIterator):
    def _init_readers(self,input_path,**kwargs):
        if isinstance(input_path,str):raise ValueError('Cooperative reader requires an explicitly configured HAL device')
        self.reader=CooperativeRawReader.from_device(input_path,delta_t=self.delta_t,ev_count=self.n_events,**kwargs)
