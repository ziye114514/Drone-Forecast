"""Same-state partial-heading forecast ablation using64 current body directions."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,sys
import numpy as np
from partial65 import *
sys.path[:0]=[str(ROOT/'research/stage64'),str(ROOT/'research/stage22')]
import study64 as observation
base=observation.prior.previous
OUT=ROOT.parent/'output/stage65_partial_heading_20261021'
CASES={c:dict(observation.CASES[c]) for c in observation.MAIN}
load=base.load;save=base.save;sha=base.sha;rows=base.rows
HISTORY=base.HISTORY;FREQUENCY=base.FREQUENCY


def sources():
    result=base.sources()
    for p in [Path(__file__),Path(__file__).with_name('partial65.py')]:result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def inputs():
    return {c:dict(current=sha(base.REFERENCE/f'off_{c}/observations_forecasts.jsonl'),
                   body=sha(observation.OUT/c/'observations.jsonl'),
                   baseline=sha(base.OUT/c/'forecasts.jsonl')) for c in CASES}


def freeze():
    observation.verify();assert not OUT.exists();assert load(observation.OUT/'raw_prefix_checks.json')['actual_RAW_prefix']
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         source_hashes=sources(),input_hashes=inputs(),cases=CASES,parameters=PARAMS,controls=controls(),
         label_hashes=load(base.OUT/'protocol.json')['label_hashes'],
         design='Axis62 complete15/lifecycle baseline exact. Body-static and body-aligned '
                'only correct axis-only rollout azimuth; unchanged reference projection '
                'magnitude, depth component and .16s expiry. All RLS/frequency/history '
                'states exact. Cached current measurements only; predictions before labels.',
         scope='Known development recordings, sparse future boxes. This cache ablation '
               'does not extend axis availability or recover unseen tilt/thrust/RPM; '
               'no deployment or realtime claim. Body plane interpretation still conditional.',
         prior_art=['https://arxiv.org/abs/2604.18289',
                    'https://openaccess.thecvf.com/content_cvpr_2018/html/Gallego_A_Unifying_Contrast_CVPR_2018_paper.html']))
    print('65 partial-direction protocol frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs();return p


def step(model,selector,r,body):
    p=model.predictor
    key='static_total' if p.correction_method=='body_static' else 'aligned_total'
    p.current_body=(r['t'],body['methods'][key])
    q,d=model.step(r['t'],r['observation'],r['sensing'],r['slot_audit']['calibration_epoch'])
    q=selector.step(q,r['observation']);q['association_diagnostic']=r['frontends']['delayed_feedback']['association_diagnostic']
    return q,d,p.correction_diagnostic


def infer():
    verify();sys.addaudithook(base.guard);summary=[]
    for case in CASES:
        rr=rows(base.REFERENCE/f'off_{case}/observations_forecasts.jsonl')
        bb=rows(observation.OUT/case/'observations.jsonl');old=rows(base.OUT/case/'forecasts.jsonl')
        models={m:Managed65(m) for m in METHODS};selectors={m:Selector() for m in METHODS}
        folder=OUT/case;folder.mkdir(exist_ok=False);stats={m:dict(applied=0,reasons={}) for m in METHODS}
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r,b,ref in zip(rr,bb,old):
                assert r['t']==b['t']==ref['t'];predictions={};lifecycle={};audit={}
                for method in METHODS:
                    q,d,diag=step(models[method],selectors[method],r,b)
                    original=ref['frontends']['axis_learn']
                    if method=='axis62':
                        assert q==original,(case,r['t'],'baseline complete15')
                        assert d==ref['lifecycle']['axis_learn']
                    for name in HISTORY+FREQUENCY+['fixed_history_mix','adaptive_history_mix']:
                        assert q['coefficients'].get(name)==original['coefficients'].get(name),(case,r['t'],method,name)
                    assert q['valid']==original['valid']
                    if method!='axis62':
                        assert d==lifecycle['axis62']
                        for name,model in models[method].predictor.models.items():
                            control=models['axis62'].predictor.models[name]
                            assert model.updates==control.updates
                            np.testing.assert_array_equal(model.parameters,control.parameters)
                            np.testing.assert_array_equal(model.cov,control.cov)
                            np.testing.assert_array_equal(model.disturbance,control.disturbance)
                    predictions[method]=q;lifecycle[method]=d;audit[method]=diag
                    stats[method]['applied']+=diag['applied'];reason=diag['reason'];stats[method]['reasons'][reason]=stats[method]['reasons'].get(reason,0)+1
                f.write(base.json.dumps(dict(t=r['t'],frontends=predictions,lifecycle=lifecycle,correction_audit=audit),allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(rr),labels_read=False,baseline_full15_exact62=True,
            all_physics_learning_and_frequency_states_exact62=True,same_fallback=True,methods=stats))
        summary.append(dict(case=case,parent=len(rr),methods=stats));print(summary[-1],flush=True)
    save(OUT/'inference_checks.json',dict(table=summary,parent_input_ticks=700,method_steps=2100,new_RAW_reads=False))


def prefix():
    verify();sys.addaudithook(base.guard);result=[]
    for case,n in [('side6',225),('train3',60)]:
        rr=rows(base.REFERENCE/f'off_{case}/observations_forecasts.jsonl')[:n]
        bb=rows(observation.OUT/case/'observations.jsonl')[:n];expected=rows(OUT/case/'forecasts.jsonl')[:n]
        for method in METHODS:
            model=Managed65(method);selector=Selector()
            for r,b,e in zip(rr,bb,expected):
                q,d,a=step(model,selector,r,b)
                assert q==e['frontends'][method] and d==e['lifecycle'][method] and a==e['correction_audit'][method]
        result.append(dict(case=case,parent=n,three_complete15_states_exact=True))
    save(OUT/'prefix_checks.json',dict(table=result,actual_RAW_prefix=False,
         scope='Independent current-cache inference prefixes;64 separately checked actual RAW measurement prefixes.'))
    print('65 causal prediction prefixes passed',flush=True)


def score():
    verify();assert len(load(OUT/'prefix_checks.json')['table'])==2
    for case in CASES:assert load(OUT/case/'completion.json')['baseline_full15_exact62']
    spec=importlib.util.spec_from_file_location('score65_original22',ROOT/'research/stage22/study22.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=METHODS;m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate
    m.label_path=lambda c:base.BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json')
    m.score();new=load(OUT/'analysis.json');old=load(base.OUT/'analysis.json')
    lookup={(r['case'],r['model'],r['nominal_h']):r for r in old['table'] if r['frontend']=='axis_learn'}
    baseline=0
    for r in new['table']:
        b=lookup[r['case'],r['model'],r['nominal_h']]
        if r['frontend']=='axis62':assert r==dict(b,frontend=r['frontend']);baseline+=1
        for key in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors']:assert r[key]==b[key]
    save(OUT/'score_checks.json',dict(rows=len(new['table']),baseline_rows_exact62=baseline,all_coverage_exact62=True))
    for r in new['table']:
        if r['nominal_h']==.4 and r['case']!='negative5' and r['model'] in ['adaptive_axis_only','adaptive_all_mix','trusted_acc']:
            print({k:r[k] for k in ['case','frontend','model','mean_fde_px','joint_success']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','prefix','score']);globals()[p.parse_args().action]()
