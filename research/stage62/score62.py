"""Scoring adapter: resolve frozen case names to original RAW time intervals."""
import importlib.util, sys
from pathlib import Path
import study62 as current
sys.path.insert(0,str(current.ROOT/'research/stage22'))


def run():
    protocol=current.verify()
    assert (current.OUT/'prefix_checks.json').exists()
    assert not (current.OUT/'analysis.json').exists()
    reference=current.load(current.REFERENCE/'protocol.json')
    cases={c:reference['cases']['off_'+c] for c in current.CASES}
    spec=importlib.util.spec_from_file_location('score62_original22',current.ROOT/'research/stage22/study22.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    original_load=m.load
    # The frozen62 protocol intentionally records a case-name list, whereas
    # original22 expects a mapping with start/duration. Supply a read adapter;
    # do not mutate the frozen protocol or the original scoring mathematics.
    def load(path):
        value=original_load(path)
        return dict(value,cases=cases) if Path(path)==current.OUT/'protocol.json' else value
    m.load=load;m.OUT=current.OUT;m.INPUT=current.OUT;m.CASES=current.CASES
    m.FRONTENDS=current.METHODS;m.MODELS=current.MODELS;m.verify=current.verify;m.evaluate=current.prior.evaluate
    m.label_path=lambda c:current.BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json')
    current.save(current.OUT/'score_adapter.json',dict(helper_sha256=current.sha(__file__),
        scorer_sha256=current.sha(current.ROOT/'research/stage22/study22.py'),cases=cases,
        correction='Original62 score entry failed before writing analysis because cases was a list. '
                   'Use original61 frozen time intervals through a loader adapter; '
                   'all inference/source/protocol files unchanged.'))
    m.score()
    new=current.load(current.OUT/'analysis.json');old=current.load(current.REFERENCE/'analysis.json')
    lookup={(r['case'].removeprefix('off_'),r['model'],r['nominal_h']):r for r in old['table'] if r['case'].startswith('off_')}
    baseline=0
    for r in new['table']:
        b=lookup[r['case'],r['model'],r['nominal_h']]
        if r['frontend']=='shared':assert r==dict(b,case=r['case'],frontend=r['frontend']),(r,b);baseline+=1
        for key in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors']:assert r[key]==b[key]
    current.save(current.OUT/'score_checks.json',dict(rows=len(new['table']),baseline_rows_exact61=baseline,
        all_coverage_exact61=True,scoring_adapter_sha256=current.sha(__file__)))
    print('62 sparse scores complete; baseline and coverage exact61',flush=True)
    for r in new['table']:
        if r['nominal_h']==.4 and r['case']!='negative5' and r['model'] in ['cv','trusted_acc','adaptive_axis_only','adaptive_axis_freq','adaptive_all_mix']:
            print({k:r[k] for k in ['case','frontend','model','valid_forecasts','mean_fde_px','joint_success']},flush=True)


if __name__=='__main__':run()
