"""Record the fixed negative mechanism and its state attribution; no new filtering."""
from pathlib import Path
import copy,datetime,json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import predictions
from summarize_live_pilot import reference_at,stats,read_rows

OUT=WORK/'output/live_innovation_covariance_20261025'
report=json.loads((OUT/'comparison.json').read_text('utf-8'))
assert not report['provisional_replay_gate_passed'] and not report['default_promoted']
attribution={}
for case in ['side','train2']:
    run=WORK/'output/live_body_association_v2_20261025'/case
    original=read_rows(run/'tracking.jsonl');pubs=read_rows(run/'predictions.jsonl')
    candidate={round(r['t'],6):r for r in read_rows(OUT/('independent_clock_'+case+'.jsonl'))}
    states={round(r['t'],6):r for r in original}
    label=WORK/'output'/('dense_labels_side_110_1195' if case=='side' else 'dense_labels_train2_40_46')/'labels.json'
    records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda q:q['t'])
    errors={m:[] for m in ['baseline','position_only','velocity_only','adaptive']}
    centers={m:[] for m in ['baseline','adaptive']}
    for row in original:
        c=row['coefficient'];a=candidate[round(row['t'],6)]['coefficient']
        if c is None:continue
        assert c['acceleration']==a['acceleration']
        truth=reference_at(records,row['t'])
        if truth is not None and row['valid']:
            for name,state in [('baseline',c),('adaptive',a)]:centers[name].append(float(np.linalg.norm(np.asarray(state['position'])-truth)))
    for pub in pubs:
        if not pub['valid']:continue
        truth=reference_at(records,pub['t']+.4)
        if truth is None:continue
        key=round(pub['observation_t'],6);base=states[key]['coefficient'];a=candidate[key]['coefficient']
        pos=copy.deepcopy(base);pos['position']=a['position']
        vel=copy.deepcopy(base);vel['velocity']=a['velocity']
        for name,state in [('baseline',base),('position_only',pos),('velocity_only',vel),('adaptive',a)]:
            point=predictions(state,pub['observation_age_ms']/1000.)[0]['0.4']
            errors[name].append(float(np.linalg.norm(np.asarray(point)-truth)))
    attribution[case]=dict(current_center_interpolated_manual_error={k:stats(v) for k,v in centers.items()},
        horizon0p4_state_ablation={k:stats(v) for k,v in errors.items()},
        same_acceleration_for_all_states=True,
        interpretation='Ablations use the same frozen origins; velocity includes nonlinear rollout interactions. Not additive causal physical effects.')
report['posthoc_state_attribution']=attribution
report['decision']='Not adopted; do not sweep parameters or run new RAW for this negative candidate'
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

table=''.join('<tr><td>'+n+'</td>'+''.join('<td>'+f"{attribution[n]['horizon0p4_state_ablation'][k]['mean']:.3f}"+'</td>' for k in ['baseline','position_only','velocity_only','adaptive'])+'</tr>' for n in ['side','train2'])
page='''<!doctype html><meta charset="utf-8"><title>速度通道自适应协方差：负结果</title>
<style>body{font:17px/1.7 system-ui;max-width:1050px;margin:35px auto;padding:0 24px;color:#19232d}table{border-collapse:collapse}th,td{border:1px solid #ccd4dd;padding:8px 14px}a{color:#1768ae}</style>
<h1>速度通道：一个固定自适应协方差候选没有改善精度</h1>
<p>只用已保存的实际观测、相同发布起点与年龄补偿，不重放RAW/GPU。人工框相邻插值仅用于最后评分（间隔≤150ms），没有参与自适应。已知开发录像，不是盲测。</p>
<p>复用当前FilterPy预测/更新、位置资格与外推，创新残差只影响下一时刻Q。平稳时缩小Q、异常残差恢复原Q，所有规则先固定；没有扫参数。它是通用方法试验，不是参考论文完整复现或新方法。</p>
<table><tr><th>片段</th><th>原基线0.4s</th><th>只换位置</th><th>只换速度</th><th>两者均换</th></tr>'''+table+'''</table>
<p>单位px，side共同14对，train2共同13对；加速度完全相同。分项有非线性外推耦合，不能把差值当可加的物理因果贡献。完整三时域和旧发布队列结果在JSON中。</p>
<p>侧视Q倍率中位降至0.01，响应偏慢，两个片段的0.1/0.2/0.4s均没有一致收益。仅凭这个候选不能推断全部自适应滤波不可行，但足以决定：不接主入口，不再扫这个机制的门槛和下界，也不追加RAW实跑。</p>
<p>关闭自适应的实现与原基线逐项一致；有效性没有改变。保留原有默认代码、最小源码包和已验证位置核心。</p>
<p><a href="comparison.json">完整结果与状态归因</a> · <a href="design.json">先固定的机制及准入条件</a> · <a href="https://arxiv.org/abs/2112.12082">自适应噪声已有研究</a> · <a href="https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html">FilterPy作者文档</a></p>'''
(OUT/'index.html').write_text(page,encoding='utf-8')

hpath=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(hpath.read_text('utf-8'))
h['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
h['latest_velocity_probe_20261025']=dict(report=str(OUT/'index.html'),results=str(OUT/'comparison.json'),
    status='fixed mechanism rejected',default_promoted=False,RAW_or_GPU_replayed=False,
    side0p4=[51.088505376197375,66.67697775606887],train2_0p4=[73.85640079632785,81.70003782143888])
h['experiments_running']=False;h['running_jobs']=[]
h['in_progress']='Velocity covariance mechanism finished negative; save quota and next precision direction at threshold'
h['next_action']='Check primary/secondary quota. At <=10% remaining pause goal as requested; resume automation01:08:30 already created. Do not tune this negative covariance mechanism. Next precision mechanism must preserve agility and have a fixed small causal comparison.'
hpath.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')

main=WORK/'output/optimization_run_20261025';p=main/'summary.json';s=json.loads(p.read_text('utf-8'))
s['latest_velocity_probe']=h['latest_velocity_probe_20261025'];p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
p=main/'index.html';s=p.read_text('utf-8')
section='''<section id="velocity-covariance-probe"><h2>最新速度机制：未采用</h2><p>一个先固定规则的创新残差自适应Q候选，在相同发布起点上使0.4秒误差side 51.09→66.68px、train2 73.86→81.70px；不是新RAW数据或物理预测。侧视Q降至过低、响应更迟缓，不接默认，不继续扫参数。</p><p><a href="../live_innovation_covariance_20261025/index.html">机制、同起点状态归因与完整结果</a>。原研究默认和位置核心不变；额度到剩余10%按用户要求保存暂停，01:08:30定时恢复。</p></section>'''
if 'id="velocity-covariance-probe"' not in s:
    s=s.replace('<section><h2>这次实际改了什么</h2>',section+'<section><h2>这次实际改了什么</h2>',1)
p.write_text(s,encoding='utf-8')

marker='### 速度协方差固定机制：负结果，不采用（2026-10-25）'
log=WORK/'WORKLOG.md'
if marker not in log.read_text('utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'\n\n')
        f.write('先查现有代码与文献，创新残差/自适应噪声已有研究（arXiv2112.12082，FilterPy渐消记忆），不主张创新或论文完整复现。独立实验live_innovation_motion.py复用作者predict/update，当前创新只调下一tick Q，倍率0.01—1、时间常数沿用0.1s、99%二维NIS门槛，预先固定且不扫参数；低置信度/出画观测不参与调整。\n\n')
        f.write('冻结旧与独立时钟两队列实际观测，原基线回放与禁用自适应控制完全一致，有效性不变。主同起点0.4s：side n14 51.089→66.677px，train2 n13 73.856→81.700px；三个时域均无一致收益。侧视Q倍率中位0.01，增加响应迟缓。判定不采用、不追加RAW、不扫下界；不代表所有自适应方法不可行。\n\n')
        f.write('报告/状态分项归因在'+str(OUT)+'，人工插值标签仅最后评分，未喂给候选。源码主入口、默认位置核心、最新42文件最小运行包都没改。初次脚本在解析阶段缺一个右括号，未创建目录或开始回放，修复后单次机制完成；未隐藏失败运行。额度上次剩13%，下一检查达到10%即保存暂停，自动恢复计划已在handoff。\n')

print(json.dumps(dict(saved=True,report=str(OUT/'index.html'),attribution=attribution),ensure_ascii=False,indent=2))
