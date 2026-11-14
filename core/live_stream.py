"""Paced RAW replay with one overwriteable current-observation slot.

Every native batch reaches the preprocessor, even if the expensive consumer
skips a display/detection frame. Only the producer owns the reader and SDK
state after warmup. No inference result from the future is exposed.
"""
from dataclasses import dataclass
from itertools import chain
from threading import Condition, Event, Thread
import time


def packet_expired(t,start,epoch,now,maximum_age):
    """A producer-expired packet cannot become fresh after nonnegative transit."""
    return bool(maximum_age is not None and epoch is not None and
                now-epoch-(t-start)>maximum_age+1e-8)


@dataclass(frozen=True)
class Packet:
    t: float
    events: object
    clean: object
    periodic: object = None
    available_wall: float = 0.
    event_count: int = None
    max_event_t: int = None
    transport_lag_ms: float = 0.
    frame_descriptor: object = None
    frame: object = None
    preview_frame: object = None
    frame_sha256: str = None
    preview_sha256: str = None


class LatestReplay:
    def __init__(self, batches, start, prepare, clock=time.perf_counter):
        self.batches = iter(batches)
        self.start = start
        self.prepare = prepare
        self.clock = clock
        self.epoch = None
        self.condition = Condition()
        self.stop = Event()
        self.slot = None
        self.done = False
        self.error = None
        self.thread = None
        self.produced = self.overwritten = self.consumed = 0
        self.native_batches = self.native_events = 0
        self.max_producer_lag_s = 0.

    def _prepare(self, t, events):
        clean, periodic = self.prepare(t, events)
        self.native_batches += 1
        self.native_events += len(events)
        return Packet(t, events, clean, periodic, self.clock())

    def _produce(self, first):
        try:
            for t, events in chain([first], self.batches):
                due = self.epoch + t - self.start
                if self.stop.wait(max(0., due - self.clock())): break
                packet = self._prepare(t, events)
                self.max_producer_lag_s = max(self.max_producer_lag_s, packet.available_wall - due)
                # Native preprocessing may intentionally avoid copying an
                # unused image buffer. Those batches still update SDK state.
                if packet.clean is None: continue
                with self.condition:
                    if self.slot is not None: self.overwritten += 1
                    self.slot = packet
                    self.produced += 1
                    self.condition.notify()
        except BaseException as error:
            self.error = error
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def __iter__(self):
        for t, events in self.batches:
            if t <= self.start + 1e-8:
                yield self._prepare(t, events)
                continue
            self.epoch = self.clock()
            self.thread = Thread(target=self._produce, args=((t, events),), name='raw-latest-replay')
            self.thread.start()
            break
        else: return
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.slot is not None or self.done)
                if self.slot is None:
                    if self.error is not None: raise self.error
                    return
                packet, self.slot = self.slot, None
                self.consumed += 1
            yield packet

    def source_now(self):
        if self.epoch is None: return self.start
        return self.start + self.clock() - self.epoch

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
            if self.thread.is_alive(): raise RuntimeError('RAW producer failed to stop')


class RateGate:
    """Fixed source-time grid, skips missed slots rather than releasing a burst."""
    def __init__(self, start, hz):
        if hz <= 0: raise ValueError('Hz must be positive')
        self.period = 1. / hz
        self.next = start
        self.missed = 0

    def take(self, t):
        if t < self.next - 1e-8: return False
        steps = max(1, int((t - self.next + 1e-8) / self.period) + 1)
        self.missed += steps - 1
        self.next += steps * self.period
        return True


class MinimumIntervalGate:
    """Wall-time rate cap: missed deadlines never cause a catchup burst."""
    def __init__(self, start, hz):
        if hz <= 0: raise ValueError('Hz must be positive')
        self.period = 1. / hz
        self.next = start + self.period
        self.missed = 0

    def take(self, t):
        if t < self.next - 1e-8: return False
        self.missed += max(0, int((t-self.next)/self.period))
        self.next = t + self.period
        return True
