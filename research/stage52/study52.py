from pathlib import Path
import argparse,datetime,hashlib,html,inspect,importlib.util,json,sys
from feedback52 import *
from checks52 import run as checks
import study51 as previous
BASE=ROOT.parent/'output';OUT=BASE/'stage52_delayed_position_feedback_20261020';REFERENCE_OUT=previous.OUT
CASES=previous.CASES;METHODS=['stable_reference','delayed_feedback'];MODELS=previous.MODELS
load=previous.load;save=previous.save;sha=previous.sha
PARAMS=dict(feature_launch='Unchanged old quality>.45, four_visible+active, current position, normal, warmup0.6s.',
    delayed_feedback='At maturity use stored old_n/old_s/old_q and current observed whole-body position state, even when current rotors absent. No new regressor appended without qualified current rotor features. No feedback when position absent; same0.2s delay,0.26s expiry.',
    preserved='Same51 stable initial frequency reference, original RLS arithmetic/loss/bounds/forgetting/excitation, same history filters/selector. Only two scheduling guards change. Old shared current force availability gate remains.')
def infer_source():
    source=inspect.getsource(previous.infer)
    replacements={"base.OUT/case/'forecasts.jsonl'":"REFERENCE_OUT/case/'forecasts.jsonl'",
        "r['frontends']['associated_slotfix']":"r['frontends']['stable_reference']",
        "if method=='reset_all':assert q==control[key],(case,t,'original48 control differs')":"if method=='stable_reference':assert q==control[key],(case,t,'original51 control differs')",
        'all_reset_control_exact_to48=True':'baseline_stable_reference_exact_to51=True'}
    for old,new in replacements.items():assert source.count(old)==1;source=source.replace(old,new)
    return source
def sources():
    result=previous.sources()
    for n in ['feedback52.py','checks52.py','study52.py']:result['research/stage52/'+n]=sha(Path(__file__).with_name(n))
    return result
def derived_hashes():return {name:hashlib.sha256(text.encode('utf-8')).hexdigest() for name,text in [('update',update_source()),('infer',infer_source())]}
def freeze():
    assert not OUT.exists();check=checks();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=previous.base.inputs(),
        label_hashes=load(REFERENCE_OUT/'protocol.json')['label_hashes'],reference_forecast_hashes={c:sha(REFERENCE_OUT/c/'forecasts.jsonl') for c in CASES},cases=CASES,
        parameters=PARAMS,module_checks=check,derived_function_hashes=derived_hashes(),
        design='Reuse actual51 inference function in isolated globals; change only output/control path/method factory and original paper_physics update two scheduling guards. No original file edits.700 same-state saved forecasts before sparse future labels. Baseline full15 experts exact51, six histories exact45.',
        prior_art=['https://www.mathworks.com/help/ident/ug/online-recursive-least-squares-estimation.html','https://arxiv.org/abs/2404.10850','https://arxiv.org/abs/2604.18289'],
        interpretation='Fix of local delayed-feedback scheduling. Standard causal stored regressor/output pairing; no novelty claim, no true physical parameter recovery, no new RAW runtime/latency measurement.'))
    (OUT/'derived_update.py').write_text(update_source(),encoding='utf-8');(OUT/'derived_infer.py').write_text(infer_source(),encoding='utf-8')
    print('Stage52 frozen; actual old update exact on full observations and occlusion controls passed',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==previous.base.inputs() and p['derived_function_hashes']==derived_hashes()
    for c,value in p['reference_forecast_hashes'].items():assert sha(REFERENCE_OUT/c/'forecasts.jsonl')==value
    return p
def infer():
    verify();namespace=dict(previous.__dict__)
    def factory(name):return Managed('stable_reference') if name=='stable_reference' else Managed52()
    namespace.update(OUT=OUT,METHODS=METHODS,Managed=factory,REFERENCE_OUT=REFERENCE_OUT,verify=verify)
    exec(compile(infer_source(),'<stage52 frozen51 inference reuse>','exec'),namespace);namespace['infer']()
def score():
    verify();spec=importlib.util.spec_from_file_location('score52',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=METHODS;m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate
    m.label_path=lambda c:BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');m.score()
def report():
    verify();a=load(OUT/'analysis.json');learning=[]
    for case in CASES:
        for method,d in load(OUT/case/'completion.json')['methods'].items():learning.append(dict(case=case,method=method,epochs=d['epoch_changes'],active=d['physics_active'],axis_updates=d['actual_updates']['adaptive_axis_only'],frequency_updates=d['actual_updates']['adaptive_axis_freq']))
    rr=[r for r in a['table'] if r['nominal_h']==.4 and r['case']!='negative5' and r['model'] in ['cv','trusted_acc','adaptive_axis_freq','adaptive_all_mix']]
    def table(rows,cols):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[x],5) if isinstance(r[x],float) else r[x]))+'</td>' for x in cols)+'</tr>' for r in rows)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>52：缺桨时用已存特征接受位置反馈</title><style>body{max-width:1300px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ccd8df}</style><h1>52：把历史桨叶特征与当前整机位置反馈分开</h1><p>51保留状态后side仍0更新，直接证据是合格特征仅在110.06/110.10及112.20–112.38s稀疏出现，反馈成熟时当前四桨/质量门槛不再成立，队列随后过期。修复将“是否存入新特征”和“能否用当前位置检验已有特征”分开：只有旧门槛通过才入队；0.2秒后若当前位置观测有效，就用已存的旧轴向/频率更新。当前桨叶缺失不产生新特征，也不把预测中心当独立测量；位置反馈来自原NN检测支持的CV状态，仍可能带检测和滤波误差。</p><p>复用51完整回放，只有原模型update两个调度判断改变，原矩阵/损失/边界/遗忘/0.26秒过期保持。全观测控制参数/协方差/扰动逐项精确；受控缺桨但位置仍可见能消费旧队列，位置也缺失则不学习，队列空后不重复学习。baseline完整15专家精确回归51，六历史系数逐项回归45，700时刻预测先保存后评分。</p>'''+table(learning,['case','method','epochs','active','axis_updates','frequency_updates'])+'''<h2>同位置0.4秒未来误差</h2>'''+table(rr,['case','frontend','model','present_targets','valid_forecasts','mean_fde_px','joint_success'])+'''<p>新增更新不等于稳定预测增益。稀疏可见框不是质心；没有实飞姿态/机械RPM/质量/风速真值。本轮为缓存因果消融，不是新RAW整合或实时性证明。原默认、所有负结果及失败保持。</p><p>RLS存储回归量/输出对齐有既有基础，参见<a href="https://www.mathworks.com/help/ident/ug/online-recursive-least-squares-estimation.html">官方在线RLS示例</a>和<a href="https://arxiv.org/abs/2404.10850">输入输出系统辨识</a>；本轮修复不称新算法。</p><p><a href="learning_summary.json">真实学习计数</a> · <a href="analysis.json">全部未来评分</a> · <a href="protocol.json">协议与控制</a></p></html>'''
    (OUT/'第五十二阶段历史桨叶与位置反馈报告.html').write_text(page,encoding='utf-8');save(OUT/'learning_summary.json',dict(table=learning));print('Stage52 report complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score','report']);globals()[p.parse_args().action]()
