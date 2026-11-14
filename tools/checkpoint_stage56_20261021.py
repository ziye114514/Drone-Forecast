"""Add verified56 association evidence to the existing checkpoint machinery."""
from pathlib import Path
import importlib.util, json, re, sys
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage56'))
import study56 as current

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def run():
    current.verify();completion=load(current.OUT/'completion.json');results=load(current.OUT/'results.json')
    assert completion['source_hashes_unchanged'] and completion['baseline_exact55'] and completion['total_cached_ticks']==2600 and completion['prefix_ticks']==720 and completion['changed_ticks_total']==0
    assert results['no_labels_read'] and results['no_frequency_or_forecast_executed']
    for case,value in load(current.OUT/'current_input_hashes.json').items():assert current.sha(current.OUT/case/'current_only.jsonl')==value
    for case in current.CASES:
        r=load(current.OUT/case/'completion.json');assert r['baseline_geometry_ids_lifecycle_exact55'] and r['prefix_exact_ticks']==120 and not r['frequency_or_forecast_executed']
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='''<div class="lead"><b>已完成55完整预测对照与56机架关联诊断，尚无稳定精度提升。</b><p>55同位置0.4s物理：侧视122.46→120.94px，清晰14.22→17.20px，3号不变；联合成功均未变，侧视仍差于历史81.91px。完整1715父/3430桨叶时刻、真实前缀及360行未来评分通过。</p><p>56复用原机架关联器，2600缓存当前时刻/720前缀全部通过，但新旧输出相同；开启硬ROI后清晰初始化合格60→3，侧视最长连续仅1。下一步57验证候选筛选ROI，保留全图密度阈值和完整轮廓。54基线保留，长期目标active。</p></div>'''
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    if '../stage56_roi_layout_association_20261021/' not in page:
        page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="../stage56_roi_layout_association_20261021/第五十六阶段ROI机架关联复用报告.html">56 · 共同机架关联与初始化诊断</a></h2><p>原50/4/33模块直接复用，输出全部相同；定位到旧机架初始化无法启用，未重复计算相同RAW频率。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('checker19_for56',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);old_save=checker.save
    next_action='Stage57 not started. Verify prior art then test past-body candidate gating while keeping global count/threshold and complete contours; no hard event clipping. Compare original54 and hard-ROI55 with exact control, same body/history, full frequency/15 models, negative and actual RAW prefix/future scoring. Quantify clipping/threshold effect; do not loosen old initialization gates or fabricate rotors.'
    original_handoff=load(BASE/'checkpoints/latest_handoff.json')
    assert original_handoff['last_completed_stage']==55 and original_handoff['zip_verified']
    def routed(path,value):
        if isinstance(value,dict) and 'last_completed_stage' in value:
            assert value['last_completed_stage']==56 and not value['in_progress']
            value.update(next_action=next_action,latest_full_RAW_stage=55,latest_verified_baseline_entry=str(ROOT/'core/run_deploy_forecast.py'),
                latest_full_prediction_report=str(current.previous.OUT/'桨盘提取对照结果与固定时刻画面.html'),latest_association_report=str(current.OUT/'第五十六阶段ROI机架关联复用报告.html'),
                verified_association_cached_ticks=2600,association_prefix_ticks=720,association_changed_ticks=0,stage56_new_frequency_or_forecast_executed=False,
                stage55_full_RAW_counts=original_handoff['verified_counts'],stage55_future_04s=original_handoff['future_04s_selected'],default_promoted=False,
                experiments_running=[],terminal_batch_results='55 formal30794 and scoring12631 exited0;56 freeze/prepare/run exited0. No worker subprocess launched by56.',
                current_status_document=str(PROJECT/'PROJECT_STATUS_20261021.md'),
                last_goal_turn_classification='progress: completed55 actual RAW/future intervention and56 unchanged association reuse; mixed/no prediction improvement and initialization bottleneck verified')
        if path.name=='current_delivery_checks_20261019.json':path=path.with_name('current_delivery_checks_stage56_20261021.json')
        old_save(path,value)
    checker.save=routed;checker.run(True,'active')
    summary=load(BASE/'checkpoints/latest_handoff.json');assert summary['last_completed_stage']==56 and summary['zip_verified'] and summary['goal_status']=='active'
    old_save(BASE/'checkpoints/latest_progress_20261021.json',summary)

if __name__=='__main__':run()
