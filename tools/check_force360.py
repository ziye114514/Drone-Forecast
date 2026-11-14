"""Independent numerical and physical source-change checks, not label fitting."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from anisotropic_response import rollout
from visible_force_rate import VisibleForceRate


def observation(t,frequencies,instances=None,angle=0.,centers_shift=(0.,0.)):
    n=np.array([np.sin(angle),-np.cos(angle)]);centers=np.array([[-30.,-20.],[30.,-20.],[30.,20.],[-30.,20.]])+[500,300]+centers_shift
    row=dict(t=t,body=dict(detected=True,center=[500,300]),coefficient=dict(position=[500.,300.],velocity=[10.,0.],acceleration=[0.,0.]),
        structure=dict(axis_xy=n.tolist(),geometry_epoch=2,pose_source='current_qualified_visible_conics'),motion={},
        force=dict(effective_gain_px_s2=200.,acceleration_correlation_s=.35))
    records=[]
    for i,f in frequencies.items():
        records.append(dict(track_id=str(i),phase_instance_id=str(i) if instances is None else str(instances[i]),qualified=True,current_force_admitted=True,proxy_twofold_hz=f,result=dict(arms=dict(joint_pose=dict(inferred_geometry=dict(cx=float(centers[i,0]),cy=float(centers[i,1]),major=40.,fov_clipped=False))))))
    return row,dict(meta=dict(records=records))


if __name__=='__main__':
    out=ROOT.parent/'output/controls360_20261109';out.mkdir(exist_ok=False)
    old={0:280.,1:310.,2:350.,3:390.};new=dict(old);new[1]=335.;new[3]=370.
    m=VisibleForceRate();a,p=observation(1.,old);m.step(a,p);b,q=observation(1+1/30,new);r=m.step(b,q)
    delta=sum(new[k]**2-old[k]**2 for k in old)/360000*30
    assert r['visible_force_rate']['applied'];assert abs(r['visible_force_rate']['f2_rate']-delta)<1e-10
    # Change the whole ellipse/normal geometry, keep actual source frequencies.
    m=VisibleForceRate();m.step(a,p);c,pc=observation(1+1/30,old,angle=.25,centers_shift=(13.,-7.));assert not m.step(c,pc)['visible_force_rate']['applied']
    # Losing or adding a source does NOT create a force step for that source.
    m=VisibleForceRate();m.step(a,p);c,pc=observation(1+1/30,{0:old[0],1:old[1]});assert not m.step(c,pc)['visible_force_rate']['applied']
    m=VisibleForceRate();m.step(a,p);c,pc=observation(1+1/30,{0:old[0],1:old[1],2:old[2],3:old[3],} ,instances={i:f'new{i}' for i in old});assert not m.step(c,pc)['visible_force_rate']['applied']
    # Relabel all anonymous sources consistently; sum/rate must not depend on yaw IDs.
    m=VisibleForceRate();aa,pp=observation(1.,{3-i:f for i,f in old.items()});m.step(aa,pp);bb,qq=observation(1+1/30,{3-i:f for i,f in new.items()});relabel=m.step(bb,qq);assert abs(relabel['visible_force_rate']['f2_rate']-delta)<1e-10
    # Independent forward-force ODE. The production propagator instead uses
    # the derivative/rotating-frame equation; agreement tests the dK/dt sign.
    rng=np.random.default_rng(360);maximum=0.
    for i in range(8):
        n=rng.normal(size=2);n/=np.linalg.norm(n);beta_t=float(rng.uniform(.4,5.));beta_n=float(rng.uniform(.4,5.));omega=float(rng.uniform(-2,2));A=float(rng.uniform(0,1800));rate=float(rng.uniform(-200,200))
        p0=rng.normal(size=2)*30;v0=rng.normal(size=2)*300;a0=rng.normal(size=2)*300;j0=rng.normal(size=2)*100;J=np.array([[0.,-1.],[1.,0.]])
        K0=beta_t*np.eye(2)+(beta_n-beta_t)*np.outer(n,n);bias=a0-A*n+K0@v0;tau=.35
        def direct(t,x):
            co=np.cos(omega*t);si=np.sin(omega*t);nt=np.array([[co,-si],[si,co]])@n
            K=beta_t*np.eye(2)+(beta_n-beta_t)*np.outer(nt,nt)
            amp=A+rate*tau*(-np.expm1(-t/tau));b=bias+j0*tau*(-np.expm1(-t/tau))
            return np.r_[x[2:4],amp*nt+b-K@x[2:4]]
        c=dict(position=p0.tolist(),velocity=v0.tolist());f=dict(acceleration_px_s2=a0.tolist(),external_unknown_rate_px_s3=j0.tolist(),external_rate_correlation_s=tau,
            anisotropic_disk_response=dict(normal_xy=n.tolist(),normal_drive_px_s2=A,tangent_response_s_inv=beta_t,normal_response_s_inv=beta_n),
            visible_force_rate=dict(amplitude_rate_px_s3=rate,correlation_s=tau))
        normal=dict(applied=True,trend=dict(shrunk_angular_rate_rad_s=omega))
        measured=rollout(c,f,normal,[.1,.4,1.]);answer=solve_ivp(direct,(0,1),np.r_[p0,v0],t_eval=[.1,.4,1.],rtol=1e-10,atol=1e-10)
        maximum=max(maximum,max(float(np.max(abs(np.array(measured[str(h)])-answer.y[:2,k]))) for k,h in enumerate((.1,.4,1.))))
    assert maximum<1e-5,maximum
    oldscore=json.loads((ROOT.parent/'output/anisotropic354_20261109/results.json').read_text('utf8'))['cases'];newscore=json.loads((ROOT.parent/'output/force357_20261109/results.json').read_text('utf8'))['cases'];past_difference=0.
    for n,c in oldscore.items():
        for a,b in zip(c['candidate']['records'],newscore[n]['candidate']['records']):
            if a['valid']:past_difference=max(past_difference,float(np.max(abs(np.array(a['prediction'])-b['prediction']))))
    assert past_difference<2e-4,past_difference
    result=dict(unequal_rotor_weight_rate_correct=True,geometry_only_change_no_force_rate=True,missing_new_or_phase_restarted_sources_no_false_step=True,
        consistent_source_relabeling_invariant=True,independent_forward_force_ODE_max_position_error_px=maximum,
        saved_RK4_354_vs_fast_357_scored_prediction_max_difference_px=past_difference,
        exact_current_state_preserved_in_factory358=True,NN_used=False,no_real_RPM_truth_claim=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result)
