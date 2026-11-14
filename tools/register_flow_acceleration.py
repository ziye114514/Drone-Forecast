"""Register the measured full cumulative acceleration+overlap candidate."""
from pathlib import Path
import copy,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/flow_acceleration_20261026'
r=json.loads((OUT/'comparison.json').read_text('utf-8'));combo=json.loads((OUT/'overlap_combination.json').read_text('utf-8'))
policy=json.loads((ROOT/'configs/accumulation_policy.json').read_text('utf-8'));failures=[]
for case in ['side','train2']:
    for h,s in r['results'][case]['scores'].items():
        for field,key in [('mean','mean_regression_allowance'),('p95','p95_regression_allowance')]:
            rule=policy['accuracy'][key];a=s['v5'][field];b=s['position_consistent'][field]
            if b>a+max(rule['absolute_px'],rule['relative_fraction']*a):failures.append(f'{case}:{h}:{field}')
assert not failures and r['development_macro04s']['position_consistent']<r['development_macro04s']['v5']
assert combo['tables']['0.4']['new_acceleration_overlap']['mean']<combo['tables']['0.4']['v5_serial']['mean']
path=ROOT/'configs/live_tracks.json';m=json.loads(path.read_text('utf-8'));assert m['recommended']=='accumulated_v5_fast_ci'
for v in m['tracks'].values():v['parameters'].setdefault('flow_acceleration','none')
v=copy.deepcopy(m['tracks']['accumulated_v6_support_overlap']);v['parameters']['flow_acceleration']='position-consistent'
v.update(status='candidate_flow_acceleration_and_overlap_pending_RAW',parent='accumulated_v5_fast_ci',
    combined_with='accumulated_v6_support_overlap',evidence=str(OUT/'comparison.json'),
    note='Raw causal flow velocity derivative, uncertainty shrinkage, past-position consistency; p/v/validity unchanged. Also includes the measured support overlap.')
m['tracks']['accumulated_v7_flow_acceleration']=v
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stub=OUT/'dry_run_only.raw';stub.write_bytes(b'')
try:
    p=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'forecast_mainline.py'),'run','--config',str(ROOT/'configs/local.json'),
        '--raw',str(stub),'--output',str(OUT/'dry_run_not_created'),'--live-revision','accumulated_v7_flow_acceleration','--dry-run'],capture_output=True,text=True,encoding='utf-8',cwd=ROOT)
    assert p.returncode==0,p.stderr;d=json.loads(p.stdout);assert not d['raw_events_read'];argv=d['argv']
    for flag,value in [('--local-flow','klt'),('--flow-preparation','gpu-overlap'),('--flow-acceleration','position-consistent')]:assert argv[argv.index(flag)+1]==value
    (OUT/'candidate_entry_check.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
finally:stub.unlink()
(OUT/'candidate_decision.json').write_text(json.dumps(dict(guard_failures=failures,default_promoted=False,
    policy_selection_reason='Position-history qualification rejects known local-motion inconsistency; no threshold/window sweep.',
    full_combination_includes_overlap=True),indent=2),encoding='utf-8')
print('v7 full candidate registered; v5 remains recommended pending RAW.')
