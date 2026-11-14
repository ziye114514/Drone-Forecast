"""Single post-run score/report; labels are never read by the RAW predictor."""
from pathlib import Path
import json,html,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
OUT=WORK/'output/stage156_parallel_geometry_20261024'
OLD=WORK/'output/stage149_inertia_scale_20261024'

def load(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def score(rows,refs,keys,published_only=False):
    indexed={round(r['t'],6):r for r in rows};table={};details={}
    for h in (.1,.2,.4):
        records=[]
        for ref in refs:
            stamp=ref['t']-h
            r=min(rows,key=lambda x:abs(x['t']-stamp))
            if abs(r['t']-stamp)>.015 or r['coefficients'].get('optimized_geometry') is None:continue
            if published_only and not r['valid']:continue
            if any(str(h) not in r['predictions'].get(k,{}) for k in keys):continue
            truth=np.asarray(ref['center'])
            errors={k:float(np.linalg.norm(np.asarray(r['predictions'][k][str(h)])-truth)) for k in keys}
            records.append(dict(t=ref['t'],errors=errors,origin_published=bool(r['valid'])))
        table[str(h)]=dict(n=len(records),missing=len(refs)-len(records),
            means={k:float(np.mean([r['errors'][k] for r in records])) for k in keys},
            medians={k:float(np.median([r['errors'][k] for r in records])) for k in keys})
        details[str(h)]=records
    return table,details

summary={};all_rows={};checks={}
for name in ['clear','side','train2']:
    rows=load(OUT/name/'predictions.jsonl');previous=load(OLD/name/'predictions.jsonl');all_rows[name]=rows
    by_old={r['t']:r for r in previous}
    if name=='clear':
        refs=[dict(t=r['t'],center=r['body']['center']) for r in previous if r['body']['detected']]
        reference='future detector centers, self-consistency only'
    else:
        label='dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46'
        labels=json.loads((WORK/'output'/label/'labels.json').read_text(encoding='utf-8'))['records'].values()
        refs=[dict(t=r['t'],center=((np.asarray(r['box'][:2])+r['box'][2:])/2).tolist()) for r in labels if r.get('state')=='present']
        reference='human-drawn bounding-box centers, not physical COM/RPM truth'
    table,details=score(rows,refs,['optimized_geometry','curved_rollout'])
    strict,_=score(rows,refs,['optimized_geometry','curved_rollout'],True)
    old_table,_=score(previous,refs,['optimized_geometry'])
    baseline_changes=[float(np.max(abs(np.asarray(r['predictions']['optimized_geometry'][h])-np.asarray(by_old[r['t']]['predictions']['optimized_geometry'][h]))))
        for r in rows for h in r['predictions'].get('optimized_geometry',{}) if h in by_old[r['t']]['predictions'].get('optimized_geometry',{})]
    done=json.loads((OUT/name/'completion.json').read_text(encoding='utf-8'))
    checks[name]=dict(source_unchanged=done['sources_unchanged'],
        event_cutoff=all(r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6) for r in rows),
        raw_only=done['raw_only'],truth_read_during_inference=done['labels_read'],
        original_detector_boxes_equal=all(r['body']==by_old[r['t']]['body'] for r in rows),
        baseline_max_coordinate_change_px=max(baseline_changes,default=0.))
    summary[name]=dict(reference=reference,frames=len(rows),references=len(refs),
        coefficient_available_score=table,published_only_score=strict,previous144_score=old_table,
        completion=done)
    (OUT/(name+'_actual_paired156.json')).write_text(json.dumps(details,indent=2),encoding='utf-8')
(OUT/'scores_actual156.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'checks156.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
for path in [ROOT/'run_optimized_forecast.py',ROOT/'research/stage153/lm153.py',ROOT/'research/stage154/turn154.py',ROOT/'research/stage156/parallel156.py']:
    shutil.copy2(path,snapshot/path.name)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for ax,name in zip(axes,['side','train2']):
    table=summary[name]['published_only_score'];x=np.arange(3)
    ax.bar(x-.18,[table[str(h)]['means']['optimized_geometry'] for h in (.1,.2,.4)],.36,label='Existing rollout')
    ax.bar(x+.18,[table[str(h)]['means']['curved_rollout'] for h in (.1,.2,.4)],.36,label='Curved fallback156')
    ax.set_xticks(x,['0.1s','0.2s','0.4s']);ax.set_title(name);ax.set_ylabel('Endpoint error (native pixels)');ax.legend(fontsize=8)
fig.savefig(OUT/'forecast_comparison156.png',dpi=160);plt.close(fig)

body=[]
for name,item in summary.items():
    for h,r in item['published_only_score'].items():
        a=r['means']['optimized_geometry'];b=r['means']['curved_rollout']
        body.append(f'<tr><td>{name}</td><td>{h}s</td><td>{r["n"]} / missing {r["missing"]}</td><td>{a:.3f}</td><td>{b:.3f}</td><td>{b-a:+.3f}</td></tr>')
timing=''.join(f'<tr><td>{n}</td><td>{d["frames"]}</td><td>{d["completion"]["processing_median_ms"]:.1f}</td><td>{d["completion"]["geometry_median_ms"]:.1f}</td><td>{d["completion"]["geometry_skipped_ticks"]}</td></tr>' for n,d in summary.items())
report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分层预测优化与并行加速</title>
<style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:17px/1.7 system-ui;background:#f7f9fc;color:#172331}table{border-collapse:collapse;background:white;width:100%;margin:18px 0}td,th{padding:8px;border:1px solid #ccd5df;text-align:left}video,img{max-width:100%}code{background:#e9eef4;padding:2px 5px}.note{background:#fff0c8;padding:16px}</style>
<h1>分层预测优化与并行加速</h1><p>2026-10-24。151—156 为本轮研究编号。三段实际 RAW 串行推理，输出原生 ON/OFF 和未来 0.1/0.2/0.4 秒预测。默认144保留；新分支可切换。</p>
<p class="note">有小幅改进，不是精度突破。事件光流和上一帧初始化没有通过，未接入正式入口。新曲线回退没有使用转速或推力，也不能据此声称物理创新成立。清晰段只有检测器自洽参考。</p>
<h2>交接审计纠正</h2><ul><li>77.9% / 61.0% 是所有处理时刻的有框比例；未标注所有时刻是否有目标，不能叫严格召回率。已标注有目标帧为74/75和36/36。</li>
<li>中心6像素是单帧定位误差，不能直接除以50像素预测误差解释占比；它还可能通过速度/加速度传播。离线只替换当前原点、保持速度和加速度，0.4秒没有改善，支持把重点转到运动状态。</li>
<li>默认结构模型的连续5帧门槛，和146—149独立几何/物理分支的资格不同。149侧视已有36个条件特征时刻，物理模型没有获得权重，不能说完全被5帧门槛挡住。</li>
<li>相关系数0.05及几轮负结果说明当前代理特征没有可靠增量，不证明真实姿态/转速永远无用。频率、混合电机、投影、重力和时间同步仍未被独立隔离。</li></ul>
<h2>实际改动</h2><ul><li>153：固定结构的8参数均无约束，用现成LM优化器，失败回退原TRF；两种形状假设、当前轮廓、门槛和80次预算保持。</li>
<li>154/156：只在尚未学成结构模型时，把法向加速度解释为瞬时转弯率，把切向加速度解释为速度变化，按曲线积分；沿用相同位置、速度、加速度和144阻尼。低速或外推超过90度时保留原方法。</li>
<li>无当前整机检测时跳过无法用于位置更新的前端提取；状态模型仍正常传播、重置，旧模型不充当新观测。</li>
<li>GPU收包分离非JSON日志，避免一条运行时提示使整段推理退出。首轮失败目录完整保留，具体原始行未被旧接收器保存。</li></ul>
<p>154曾对所有状态使用曲线，清晰段0.4秒自洽25.595→25.814，未通过。156保留已建模结构的原路径，只在未建模状态回退；这不是可靠正视/侧视分类器，结构记忆也不是当前姿态观测。</p><p>156另把两套独立几何任务安全并行，移除其进程全局检测函数切换。每套任务保留自己的恢复和结构状态，预测前等待同一时刻的两套结果，不把旧结果充当当前测量。</p><h2>同一起点评分</h2><p>下面仅计起点valid=true的已发布预测，未发布算缺失。历史“有可计算系数”口径也完整保存在scores_actual156.json：侧视66对50.027→49.216；严格口径64对46.430→45.594。两种口径不得交叉作改善比较。人工参考是框中心，不是物理重心；已有录像均属于开发数据。</p>
<table><tr><th>录像</th><th>时域</th><th>评分 / 缺失</th><th>同状态原外推 px</th><th>曲线外推 px</th><th>变化 px</th></tr>'''+''.join(body)+'''</table><img src="forecast_comparison156.png">
<h2>实际耗时</h2><table><tr><th>录像</th><th>处理时刻</th><th>处理中位 ms</th><th>几何中位 ms</th><th>跳过无整机前端次数</th></tr>'''+timing+'''</table>
<p>记录的是本轮单次运行，不把跨轮总耗时差归因于优化；GPU/系统负载会变。模型求解的同输入交替对照见../stage153_LM_geometry_20261024/replay153.json；并行同输入100时刻对照见probe156.json，观测/中心完全一致。并行后frontend/model字段是两臂耗时之和，不能与墙钟geometry耗时相加。50Hz预算20ms，当前仍未达到。</p>
<h2>侧视模糊：已补原生对照</h2><p>同一时刻20/10/5/2ms。短窗口减少拖影，也减少事件与轮廓；5ms目前仅用于显示，不能当作检测收益。</p>
<img src="../stage149_inertia_scale_20261024/blur/side_118_12_windows.png"><p><a href="../stage149_inertia_scale_20261024/blur/side20_vs5ms.mp4">20ms与5ms同期预览</a> · <a href="../stage149_inertia_scale_20261024/blur/diagnostics149.json">原始诊断</a></p>
<h2>预测视频</h2><p>绿：原外推；紫：曲线外推；灰：匀速。5ms显示，不插帧、不使用灰度重建。</p>
<h3>侧视</h3><video controls src="side/predictions.mp4"></video><h3>train2</h3><video controls src="train2/predictions.mp4"></video><h3>清晰</h3><video controls src="clear/predictions.mp4"></video>
<h2>下一步</h2><p>优先减少50Hz链路延迟与前端重复计算；运动预测只增加有明确机制且能超过当前曲线基线的改动。桨盘研究先限缩为姿态变化/机动提示，不继续把未经独立验证的频率代理直接当总推力。不能把侧视信息不足一概归因于算法或压缩。</p>
<p><a href="scores_actual156.json">实际评分</a> · <a href="checks156.json">必要检查</a> · <a href="../stage151_event_motion_20261024/observation_probe_scores.json">事件光流负结果</a> · <a href="../stage152_warm_geometry_20261024/replay152.json">上一帧初始化负结果</a></p></html>'''
path=OUT/'分层预测优化与并行加速.html';path.write_text(report,encoding='utf-8')
(WORK/'output/latest_optimized_forecast.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="0;url=stage156_parallel_geometry_20261024/'+path.name+'">',encoding='utf-8')
print(json.dumps(dict(scores={n:d['coefficient_available_score'] for n,d in summary.items()},checks=checks,report=str(path)),ensure_ascii=False),flush=True)
