"""Prediction publication bookkeeping, separate from numerical estimator."""
class Publication:
    def __init__(self):self.last_body=None;self.observations=0;self.last_t=None
    def step(self,t,detected):
        assert self.last_t is None or t>self.last_t;self.last_t=t
        if self.last_body is not None and t-self.last_body>.50000001:self.observations=0
        if detected:self.last_body=t;self.observations+=1
        return bool(self.observations>=5 and self.last_body is not None and t-self.last_body<=.16000001)
