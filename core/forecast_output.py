"""Small source-time output schema; never changes forecasting state."""


def select(value,keys):return {k:value[k] for k in keys if k in value}


def compact_record(row):
    result=select(row,('t','body','coefficient','valid','physical','force_observed_position','nn_fresh','nn_answer_age_s',
        'raw_events','frequency_input_events','max_event_t','cost','weak_verifier'))
    s=row['structure']
    result['schema']='event_drone_forecast.compact.v2'
    result['full_internal_diagnostics']=False
    result['structure']=select(s,('mode','geometry_epoch','observed_count','observed_centers','observed_model_ids',
        'model_centers','axis_xy','geometric_weight','current_geometric_center','center_source','pose_source','five_circles'))
    result['force']=select(row['force'],('active','input_u','effective_gain_px_s2','acceleration_px_s2',
        'disturbance_px_s2','gain_learned','gain_ever_learned','short_gap_acceleration_held','acceleration_source','acceleration_correlation_s',
        'last_current_force_age_s','held_input_not_a_current_thrust_measurement','metric_force_measured','future_commands_observed'))
    result['axis_amplitude_bridge']=select(row.get('axis_amplitude_bridge') or {},('applied','amplitude_current',
        'learning_input_current','period_age_s','maximum_period_hold_s','held_relative_f2','current_frequency_invented',
        'hidden_rotor_frequencies_added','held_amplitude_excluded_from_gain_and_input_integral_learning'))
    result['period']=select(row['period'],('qualified','frequency_hz','pixels','mode_fraction','reason','harmonic_ambiguous'))
    result['period_selection']=select(row['period_selection'],('source','frequency_hz','relative_f2','coherent','reason'))
    return result


class LeanHistory:
    """One full current record plus lossless numeric timing/counter history.

    Forecasting needs only the preceding full record. Older diagnostics remain
    in JSONL on disk; final exact timing quantiles still require O(N) numeric
    storage. This reduces history substantially, not an unbounded-stream claim.
    """
    def __init__(self):
        from array import array
        self.array=array;self.times=array('d');self.flags=array('B');self.mode_ids=array('B')
        self.modes=[];self.costs={};self.latest=None
    def __len__(self):return len(self.times)
    def append(self,row):
        import math
        count=len(self)
        for key in row['cost']:
            if key not in self.costs:self.costs[key]=self.array('d',[math.nan])*count
        for key,column in self.costs.items():column.append(row['cost'].get(key,math.nan))
        mode=row['structure']['mode']
        if mode not in self.modes:self.modes.append(mode)
        if len(self.modes)>256:raise ValueError('Unexpected number of structure modes')
        bits=(bool(row['body'].get('detected')),
            bool(row['valid']),bool(row['force']['active']),bool(row['force']['gain_learned']),
            bool((row.get('axis_amplitude_bridge') or {}).get('applied')),
            bool((row.get('nn_structure_schedule') or {}).get('deferred_original_due')))
        self.flags.append(sum(int(value)<<i for i,value in enumerate(bits)))
        self.mode_ids.append(self.modes.index(mode));self.times.append(row['t']);self.latest=row
    def light_record(self,index):
        import math
        flag=self.flags[index]
        return dict(t=self.times[index],cost={k:v[index] for k,v in self.costs.items() if not math.isnan(v[index])},
            body=dict(detected=bool(flag&1)),valid=bool(flag&2),structure=dict(mode=self.modes[self.mode_ids[index]]),
            force=dict(active=bool(flag&4),gain_learned=bool(flag&8)),
            axis_amplitude_bridge=dict(applied=bool(flag&16)),
            nn_structure_schedule=dict(deferred_original_due=bool(flag&32)))
    def __iter__(self):return (self.light_record(i) for i in range(len(self)))
    def __getitem__(self,key):
        if isinstance(key,slice):return HistoryView(self,range(*key.indices(len(self))))
        index=key if key>=0 else len(self)+key
        if not 0<=index<len(self):raise IndexError(key)
        return self.latest if index==len(self)-1 else self.light_record(index)
    @property
    def numeric_storage_bytes(self):
        return sum(len(v)*v.itemsize for v in [self.times,self.flags,self.mode_ids,*self.costs.values()])
    def finish_current_costs(self):
        for key,value in self.latest['cost'].items():self.costs[key][-1]=value


class HistoryView:
    def __init__(self,history,indices):self.history=history;self.indices=indices
    def __len__(self):return len(self.indices)
    def __iter__(self):return (self.history.light_record(i) for i in self.indices)
