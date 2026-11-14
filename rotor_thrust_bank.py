"""Current observed inputs and explicit unknown-force ownership.

The stage161 hidden-prior bank is retained for reproducible controls. The
default bank never inserts a hidden motor into its observed contribution sum.
"""
import numpy as np
from visibility_period import LegacyPeriodProxy,VisibilityPeriodProxy,force_input


class LegacyRotorThrustBankProxy:
    ttl_s=.12
    def __init__(self):
        self.legacy=LegacyPeriodProxy();self.mapper=VisibilityPeriodProxy()
        self.epoch=None;self.last_t=None;self.previous={}
        self.values=None;self.variances=None;self.seen=np.full(4,-np.inf)
        self.reference=None;self.anchor=None
    def step(self,t,structure,rotors,rates,pooled):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing causal time required')
        epoch=structure.get('geometry_epoch',0)
        if self.last_t is not None and (epoch!=self.epoch or t-self.last_t>.5):
            # Slot identity resets must not silently reset the whole-body
            # frequency gauge and create an artificial thrust step.
            legacy=self.legacy;self.__init__();self.legacy=legacy
        dt=0. if self.last_t is None else t-self.last_t;self.last_t=t;self.epoch=epoch
        legacy_u,legacy_q,old=self.legacy.step(t,structure,rotors,rates,pooled)
        samples,mapping=self.mapper._slots(structure,rotors,rates)
        stable={}
        for i,f in samples.items():
            previous=self.previous.get(i)
            if previous is not None and t-previous[0]<self.ttl_s and abs(np.log(f/previous[1]))<.08:stable[i]=f
            self.previous[i]=(t,f)
        if self.values is None and structure.get('mode')=='four' and len(stable)==4:
            self.values=np.array([stable[i]**2 for i in range(4)])
            self.variances=(.04*self.values)**2;self.seen[:]=t
            self.reference=float(self.values.sum())
            self.anchor=old['relative_f2'] if old['relative_f2'] is not None else 1.
        direct=sorted(stable)
        if self.values is not None:
            # Fractional random walk: unresolved motors get less certain with time.
            self.variances+=(self.values*.8)**2*max(dt,0.)
            for i,f in stable.items():
                self.values[i]=f*f;self.variances[i]=(.04*f*f)**2;self.seen[i]=t
        current_nonflat=structure.get('mode') in ('four','three','partial_two') and len(samples)>=2
        if not current_nonflat:
            return legacy_u,legacy_q,dict(old,rotor_bank_used=False,
                rotor_bank_reason='no_current_identified_geometry',hidden_frequencies_measured=0)
        if self.values is None:
            return None,0.,dict(old,source='rotor_bank_uninitialized',relative_f2=None,
                learning_input_current=False,rotor_bank_used=False,
                reason='need_four_independently_identified_coherent_periods',mapping=mapping)
        age=t-self.seen;fresh=bool(np.all(age<=self.ttl_s+1e-9))
        total=float(self.values.sum());sig=float(np.sqrt(self.variances.sum()))
        relative=float(self.anchor*total/self.reference)
        eligible=fresh and len(stable)>=2 and 2*sig/max(total,1e-9)<=.3
        complete=len(stable)==4
        hidden=[i for i in range(4) if i not in stable]
        u,q=force_input(structure,relative,eligible)
        info=dict(source='four_individual_f2_sum' if complete else 'partial_individual_short_prior',
            frequency_hz=None,relative_f2=relative if u is not None else None,coherent=eligible,
            reason='current_four_rotor_sum' if complete else 'bounded_hidden_prior' if eligible else 'hidden_age_or_uncertainty_or_period_support',
            rotor_bank_used=u is not None,learning_input_current=complete and u is not None,
            current_model_ids=direct,mapping=mapping,
            individual_f2_estimates=self.values.tolist(),individual_sigma_proxy=np.sqrt(self.variances).tolist(),
            hidden_model_ids=hidden,hidden_ages_s={str(i):float(age[i]) for i in hidden},
            relative_sum_interval_proxy=[self.anchor*max(0.,total-2*sig)/self.reference,self.anchor*(total+2*sig)/self.reference],
            uncertainty_calibrated=False,partial_sum_requires_common_scale=False,
            maximum_hidden_prior_s=self.ttl_s,hidden_frequencies_measured=0,
            metric_thrust_measured=False,mechanical_RPM_measured=False,
            attitude_moment_constraints_used=False)
        return u,q,info


class RotorThrustBankProxy:
    """Current contributions only, on a fixed image-input gauge.

    Hidden motors are owned by the observer's unknown state. A slot geometry
    epoch labels association evidence, not the vehicle's head or tail. The
    constant reference defines units; it is not a measured hover speed.
    """
    ttl_s = .12
    reference_hz = 300.
    # Nearest distinct ratio between the first four harmonics is 4/3.
    # Retain the old .08 log margin to that boundary, while avoiding an
    # automatic dropout on every plausible 10-20% same-slot frequency step.
    # This is a hypothesis gate, not proof of mechanical speed accuracy.
    current_change_limit = np.log(4./3.)-.08

    def __init__(self,vector_pool=False):
        self.mapper = VisibilityPeriodProxy()
        self.last_t = self.epoch = None
        self.previous = {}
        self.previous_pool = None
        self.pool_anchor_f2 = None
        self.vector_pool=vector_pool;self.pool_anchor_vector=None
        self.pool_sources={};self.previous_pool_source=None

    def step(self, t, structure, rotors, rates, pooled):
        if self.last_t is not None and t <= self.last_t:
            raise ValueError('Increasing causal time required')
        epoch = structure.get('geometry_epoch', 0)
        if epoch != self.epoch or (self.last_t is not None and t-self.last_t > .5):
            self.previous.clear(); self.previous_pool = None
        self.last_t = float(t); self.epoch = epoch
        samples, mapping = self.mapper._slots(structure, rotors, rates)
        phase_slots={a['model_id'] for a in mapping.get('assignments',[])
            if rates[a['region_index']].get('spatial_phase_supported')}
        stable = {}; rapid = []
        for i, f in samples.items():
            previous = self.previous.get(i)
            if previous is not None and t-previous[0] < self.ttl_s:
                change = abs(np.log(f/previous[1]))
                if change < self.current_change_limit or i in phase_slots:
                    stable[i] = f
                    if change >= .08: rapid.append(i)
            self.previous[i] = (float(t), f)
        axis = structure.get('axis_xy')
        n = np.asarray(axis, float) if axis is not None else np.zeros(0)
        axis_good = n.shape == (2,) and np.isfinite(n).all() and np.linalg.norm(n) > 1e-6
        if axis_good and n[1] > 0: n = -n
        denominator = 4*self.reference_hz**2
        components = {}; source = 'unavailable'; relative = None; sigma = 0.
        identified = structure.get('mode') in ('four', 'three', 'partial_two')
        # Current slot evidence can start from a partial geometry immediately
        # after temporal confirmation. No four-slot bank is initialized.
        if identified and len(stable) >= 2 and axis_good:
            components = {f'slot:{epoch}:{i}': (stable[i]**2/denominator*n).tolist() for i in stable}
            relative = float(sum(f*f for f in stable.values())/denominator)
            sigma = float(np.sqrt(sum(((.12 if i in rapid else .04)*f*f)**2
                for i,f in stable.items()))/denominator)
            source = 'current_visible_slot_f2_sum'
        elif not identified:
            f = pooled.get('frequency_hz') if pooled.get('qualified') else None
            if f is not None and np.isfinite(f) and 60 <= f <= 1200:
                old = self.previous_pool
                pool_source=pooled.get('selection_source')
                change_limit=self.current_change_limit if pooled.get('same_source_spatial_tracking_confirmed') else .08
                coherent = old is not None and t-old[0] < self.ttl_s and abs(np.log(f/old[1])) < change_limit and pool_source==self.previous_pool_source
                self.previous_pool = (float(t), float(f))
                self.previous_pool_source=pool_source
                if coherent and axis_good:
                    key='pooled_change_proxy' if pool_source is None else f'pooled_change_proxy:{pool_source}'
                    anchor_f2,anchor_vector=self.pool_sources.setdefault(key,(float(f*f),float(f*f)*n.copy()))
                    if self.pool_anchor_f2 is None: self.pool_anchor_f2 = float(f*f)
                    if self.pool_anchor_vector is None:self.pool_anchor_vector=float(f*f)*n.copy()
                    # An unresolved spectral mixture is only a change proxy.
                    # Its baseline/hidden force remains entirely in d. It is
                    # never multiplied by four or called observed total thrust.
                    relative = float((f*f-anchor_f2)/denominator)
                    vector=(f*f*n-anchor_vector)/denominator if self.vector_pool else relative*n
                    components = {key: vector.tolist()}
                    sigma = .25+.5*abs(relative)
                    source = 'ambiguous_pooled_f2_change_proxy'
        u = np.sum(list(components.values()), axis=0) if components else None
        quality = max(.35, float(structure.get('geometric_weight', 0.))) if u is not None else 0.
        return u, quality, dict(
            source=source, frequency_hz=None, relative_f2=relative,
            coherent=u is not None, rotor_bank_used=source == 'current_visible_slot_f2_sum',
            reason='current_visible_only' if source == 'current_visible_slot_f2_sum' else
                'unresolved_change_not_total_force' if components else 'await_current_period_and_axis',
            learning_input_current=source == 'current_visible_slot_f2_sum' and not rapid,
            current_model_ids=sorted(stable) if identified else [], mapping=mapping,
            current_individual_frequency_hz={str(i):float(f) for i,f in stable.items()} if identified else {},
            individual_frequencies_are_current_native_harmonic_proxies=True,
            observed_components=components, input_sigma_proxy=sigma,
            input_uncertainty_calibrated=False, fixed_denominator_f2=denominator,
            reference_hz_is_unit_convention=True, partial_startup_allowed=True,
            rapid_current_change_hypothesis_model_ids=rapid,
            maximum_current_log_step=self.current_change_limit,
            rapid_change_has_independent_phase_confirmation=False,
            spatial_phase_supported_model_ids=sorted(phase_slots),
            hidden_model_ids=[i for i in range(4) if i not in stable] if identified else [],
            hidden_prior_in_observed_sum=False, hidden_force_owner='joint_unknown_state_d',
            hidden_frequencies_measured=0, partial_sum_requires_common_scale=False,
            mixed_spectrum=source == 'ambiguous_pooled_f2_change_proxy',
            mixed_proxy_is_total_thrust=False, mixed_proxy_gain_learning=False,
            mixed_source_membership_observed=False,
            pooled_estimator_source=pooled.get('selection_source'),
            estimator_source_change_is_not_thrust_change=True,
            metric_thrust_measured=False, mechanical_RPM_measured=False)
