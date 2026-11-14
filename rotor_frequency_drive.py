"""Thrust-change cue with geometry held out of the temporal difference.

Use -r_i(now) * delta(f_i^2), not delta(-r_i*f_i^2): translating/re-estimating
the circle centre must not be called a rotor thrust change. Missing/new phase
instances transfer their baselines to the unknown angular response.
"""
import numpy as np
from native_torque_inputs import selected
from axis_amplitude_bridge import current_axis_qualified


def sources(row,period):
    st=row['structure'];center=st.get('current_geometric_center')
    if center is None:center=row['body'].get('center')
    if center is None or not row['body'].get('detected') or not current_axis_qualified(st):return {}
    joint='records' in (period.get('meta') or {});center=np.asarray(center,float)
    if joint:
        # Ring charts are motion-compensated to cutoff; native20ms hubs refer
        # to the interval midpoint. Bring the lever-arm origin to that cutoff.
        mo=row.get('motion') or {};s=mo.get('estimate') or {};dt=mo.get('dt_s')
        if dt and mo.get('current_observation') and not s.get('peak_at_search_boundary') and s.get('score',0)>0:
            center=center+.010*np.array([s['dx'],s['dy']])/dt
        records=[]
        for r in period['meta']['records']:
            if not r.get('qualified') or not r.get('current_force_admitted'):continue
            g=r.get('result',{}).get('arms',{}).get('joint_pose',{}).get('inferred_geometry') or {};f=r.get('proxy_twofold_hz')
            if not f or g.get('fov_clipped') or not g.get('major'):continue
            p=g.get('projected_shaft_xy',[g.get('cx'),g.get('cy')])
            records.append((str(r['track_id']),str(r.get('phase_instance_id',r['track_id'])),f,g,p))
    else:records=[(key,key,f,g,[g['cx'],g['cy']]) for key,f,g in selected(row,period)]
    if len(records)==4:center=np.mean([r[4] for r in records],axis=0)
    return {key:dict(weight=(float(f)/300)**2/4,lever=(np.asarray(p,float)-center)/(g['major']/2),
        position=np.asarray(p,float),radius=float(g['major']/2),instance=instance)
        for key,instance,f,g,p in records}


def change(current,previous,tangent,normal_amplitude):
    shared=[k for k in current.keys()&previous.keys() if current[k]['instance']==previous[k]['instance']]
    raw=sum((-current[k]['lever']*(current[k]['weight']-previous[k]['weight']) for k in shared),np.zeros(2))
    variance=sum(.04**2*(current[k]['weight']**2+previous[k]['weight']**2)*(tangent@current[k]['lever'])**2 for k in shared)/normal_amplitude**2
    return float(tangent@raw/normal_amplitude),float(variance),sorted(shared)


def redistribution(current,previous,tangent,normal_amplitude):
    """Origin-invariant visible force redistribution; collective stays unknown.

For shared sources d_i=w_i(now)-sum(now)/sum(previous)*w_i(previous), sum(d)=0.
Thus a common RPM scaling or an uncertain image centre cannot create a tilt
cue. Under partial visibility this is a contrast, never complete body torque.
"""
    shared=sorted(k for k in current.keys()&previous.keys() if current[k]['instance']==previous[k]['instance'])
    if len(shared)<2:return 0.,0.,shared
    now=np.array([current[k]['weight'] for k in shared]);old=np.array([previous[k]['weight'] for k in shared]);scale=now.sum()/old.sum()
    delta=now-scale*old;radius=float(np.median([current[k]['radius'] for k in shared]))
    p=np.array([current[k]['position'] for k in shared]);p=p-p.mean(0)
    lever=(p@tangent)/(radius*normal_amplitude);drive=-float(lever@delta)
    J=np.eye(len(shared))-np.outer(old/old.sum(),np.ones(len(shared)))
    covariance=J@np.diag(.04**2*(now**2+scale**2*old**2))@J.T
    variance=float(lever@covariance@lever)
    return drive,max(0.,variance),shared
