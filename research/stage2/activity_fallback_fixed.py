"""Correct reacquisition reset; keep frozen first attempt for audit."""
from activity_fallback import ActivityTracker as FrozenActivityTracker

class ActivityTracker(FrozenActivityTracker):
    def step(self,events,t):
        if self.last_seen is not None and t-self.last_seen>.3:
            # Reset once. Leaving the old last_seen repeatedly erased the
            # confirmation streak on every later frame, preventing reacquire.
            self.center=None;self.velocity[:]=0;self.pending=None;self.streak=0;self.last_seen=None
        return super().step(events,t)
