"""Causal local wrapper: run BOTH initial signs of the unmodified author core.

Scores are diagnostics, not calibrated probabilities or true mechanical RPM.
The author checkout/DLL is local only and not part of the source candidate.
"""
import math
from reference_api import AuthorTracker


class SignBank:
    def __init__(self, seed_phase_hz, blades=2, tracker_factory=AuthorTracker):
        self.trackers={s:tracker_factory(s*60*seed_phase_hz,blades) for s in (1,-1)}
        self.quality={s:None for s in self.trackers}
        self.streak={s:0 for s in self.trackers}
        self.selected=None
        self.last_t=None
        self.switches=0

    def feed(self, t, events, solve_pose=False, current_period=None):
        if self.last_t is not None and t<=self.last_t:
            raise ValueError('Sign bank requires increasing event cutoff times')
        dt=.02 if self.last_t is None else t-self.last_t
        self.last_t=t
        modes={s:tracker.feed(events['x'],events['y'],events['t'],events['p'],solve_pose)
               for s,tracker in self.trackers.items()}
        for s,r in modes.items():
            good=r['weighted_phase_concentration']>=.5 and r['informative_ring_sectors']>=9
            if current_period is not None:
                f=current_period.get('frequency_hz')
                ratio=r['phase_hz']/f if f and f>0 else 0.
                harmonic=min((.5,1.,2.),key=lambda v:abs(ratio-v))
                match=bool(current_period.get('qualified') and ratio>0 and abs(math.log(ratio/harmonic))<=math.log(1.1))
                r['current_pixel_period_agreement']=match
                r['phase_to_pixel_harmonic_hypothesis']=harmonic if match else None
                good=good and match
            self.streak[s]=self.streak[s]+1 if good else 0
            a=1-math.exp(-dt/.1)
            old=self.quality[s]
            self.quality[s]=r['weighted_phase_concentration'] if old is None else old+a*(r['weighted_phase_concentration']-old)
        eligible=[s for s in modes if self.streak[s]>=3]
        reason='insufficient_current_phase_support'
        accepted=False
        ambiguity=False
        if eligible:
            candidate=max(eligible,key=self.quality.get)
            if self.selected not in eligible or self.quality[candidate]>self.quality[self.selected]+.1:
                if self.selected is not None and self.selected!=candidate:self.switches+=1
                self.selected=candidate
            frequencies=[modes[s]['phase_hz'] for s in eligible]
            ambiguity=len(frequencies)>1 and max(frequencies)/max(min(frequencies),1e-9)>1.1
            accepted=not ambiguity
            reason='conflicting_phase_frequency_hypotheses' if ambiguity else 'current_phase_supported'
        selected=modes[self.selected] if accepted else None
        directions=sorted({int(math.copysign(1,modes[s]['angular_velocity_rad_s']))*modes[s]['rotation_sign']
                           for s in eligible})
        return dict(t=t,valid=accepted,reason=reason,selected_initial_sign=self.selected,
                    selected=selected,modes={str(s):v for s,v in modes.items()},
                    quality={str(s):v for s,v in self.quality.items()},
                    consecutive_supported={str(s):v for s,v in self.streak.items()},
                    direction_ambiguous=len(directions)>1,physical_image_directions=directions,
                    current_pixel_period=current_period,
                    mechanical_RPM_truth=False,force_valid=False)

    def close(self):
        for tracker in self.trackers.values():tracker.close()
