"""Small forward-side-phase feasibility probe, not a forecast sweep."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from line_phase_moments import measure_xy,measure_cells
from pure_visibility170 import window


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',default='line366_20261109');p.add_argument('--spatial-cells',action='store_true');args=p.parse_args()
    measure=measure_cells if args.spatial_cells else measure_xy
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False)
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('line_phase_moments.py','tools/pilot_line366.py')}
    for f in hashes:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    rng=np.random.default_rng(366);controls=[];traces={}
    for name in ('rotating_line','opposite_phase_pair','translated_static_line','periodic_illumination','colored_nonperiodic_extent'):
        for seed in range(8):
            rng=np.random.default_rng(366+seed);t=np.sort(rng.uniform(0,.033333,6000))
            if name=='periodic_illumination':
                accept=rng.random(len(t))<(.6+.4*np.cos(2*np.pi*500*t));t=t[accept]
            radial=rng.uniform(-1,1,len(t));phase=2*np.pi*250*t+.4
            if name=='rotating_line':u=radial*np.cos(phase)
            elif name=='opposite_phase_pair':
                group=rng.integers(0,2,len(t));u=(2*group-1)*2/3+radial/3*np.cos(phase+group*np.pi/2)
            elif name=='colored_nonperiodic_extent':
                noise=rng.normal(size=160);filtered=np.convolve(noise,np.ones(5)/5,mode='same');extent=.6+.25*np.tanh(filtered);u=radial*np.interp(t,np.linspace(0,.033333,len(extent)),extent)
            else:u=radial
            xy=np.c_[60*u,rng.normal(0,.8,len(t))];polarity=rng.integers(0,2,len(t));info,trace=measure(xy,t,polarity)
            controls.append(dict(case=name,seed=seed,**info))
            if seed==0 and trace is not None:traces[name]=trace
    results={}
    for n in ('4','11','16'):
        source=ROOT.parent/('output/bounded362_20261109/recording4/continued.jsonl' if n=='4' else f'output/structural348_20261109/recording11/continued.jsonl' if n=='11' else 'output/continuation339_20261109/recording16/continued.jsonl')
        rr=[json.loads(s) for s in source.read_text('utf8').splitlines()]
        cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else f'output/phase185_20261107/cache_recording{n}')
        manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames'] if n=='4' else None
        outputs=[]
        for i in range(20,min(175,len(rr)),10):
            t=rr[i]['t'];begin=t-.033333
            candidates=[r for r in rr[:i] if r['t']<=begin+1e-9 and r['body'].get('detected') and not r['body'].get('center_provisional') and 0<t-r['t']<=.12]
            if not candidates:outputs.append(dict(t=t,qualified=False,reason='no_past_confirmed_rotor_ROI'));continue
            old=candidates[-1];shift=old['motion'].get('estimate') or {};dt=old['motion'].get('dt_s')
            if not dt or shift.get('score',0)<=0 or not shift.get('current_displacement_qualified',True):outputs.append(dict(t=t,qualified=False,reason='no_past_qualified_transport'));continue
            velocity=np.array([shift['dx'],shift['dy']])/dt;e=np.load(cache/manifest[i]['file']) if manifest is not None else window(cache,t)
            end=round(t*1e6);keep=(e['t']>=end-33333)&(e['t']<end);e=e[keep]
            ages=t-e['t']/1e6;xy=np.c_[e['x']+ages*velocity[0],e['y']+ages*velocity[1]]
            bounds=np.array(old['body']['bbox'],float)+np.tile(velocity*(t-old['t']),2)+np.array([-24.,-24.,24.,24.])
            keep=(xy[:,0]>=bounds[0])&(xy[:,0]<=bounds[2])&(xy[:,1]>=bounds[1])&(xy[:,1]<=bounds[3]);xy=xy[keep];e=e[keep]
            info,trace=measure(xy,(e['t']-(end-33333))/1e6,e['p'])
            outputs.append(dict(t=t,past_transport_source_t=old['t'],**info))
            if trace is not None and f'recording{n}' not in traces:traces[f'recording{n}']=trace
        results[n]=outputs;print(n,'frames',len(outputs),'line fits',sum(r['qualified'] for r in outputs),'late R2',[round(r['late_R2'],3) for r in outputs if r['qualified']],flush=True)
    result=dict(cases=results,controls=controls,source_sha256=hashes,
        labels_or_forecasts_used_for_method_selection=False,ROI_is_past_rotor_region_not_NN=True,
        current_spatial_model_uses_early_events_only=True,late_is_frozen_validation_only=True,
        direct_rotating_line_moment_model_not_author_CoHelix_reproduction=True,
        validation_slow_nuisance_refit_does_not_change_frozen_harmonic_clock=True,
        event_ROI_not_claimed_all_current_pixels_individually_periodic=True,
        spatial_cells_are_fixed_from_early_geometry_and_all_reported=args.spatial_cells,
        diagnostic_only_no_force_or_predictor_integration=True,RAW_FPS_measured=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');(out/'traces.json').write_text(json.dumps(traces),'utf8')
    for name in sorted(set(r['case'] for r in controls)):
        aa=[r['late_R2'] for r in controls if r['case']==name and r['qualified']];print(name,'late R2',min(aa),float(np.median(aa)),max(aa),flush=True)
    if args.spatial_cells:
        for n,records in results.items():
            values=[(r['t'],k,round(c['late_R2'],3)) for r in records for k,c in r.get('spatial_cells',{}).items() if c['qualified']];print(n,'all local late R2',values,flush=True)
