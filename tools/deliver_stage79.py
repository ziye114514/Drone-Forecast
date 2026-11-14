"""Render completed module evidence; no detector/future inference in delivery."""
from pathlib import Path
import hashlib,html,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/stage79_native_frequency_20261021'
load=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
checks=load(OUT/'checks79.json');controls=load(OUT/'controls79.json')
assert checks['parent']==2185 and all(r['full_features_exact'] for r in checks['prefix'])
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
cases=['clear6','side6','train3','extra1','extra2','extra4'];names=['default7','strict7','fast3','aligned_strict7']
labels=['原频谱','原生默认7','原生严格7','原生短3','对齐严格7'];colors=['#707d8f','#c78c43','#3776b9','#9c59a8','#23936d']
table={r['case']:r for r in checks['table']};fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(len(cases))
for k,name in enumerate(['old']+names):
    values=[table[c]['original_periodic_supported'] if name=='old' else table[c]['supported'][name] for c in cases]
    ax.bar(x+(k-2)*.16,values,.15,label=labels[k],color=colors[k])
ax.set_xticks(x,[f'{c}\n{table[c]["parent"]}时刻' for c in cases]);ax.set_ylabel('合格当前时刻数（不是准确率）');ax.legend(ncol=3);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'coverage79.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(12,9),sharey=True)
prev=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
for c,ax in zip(cases,axes.ravel()):
    rr=[json.loads(s) for s in (OUT/c/'features.jsonl').read_text().splitlines()];old=[json.loads(s) for s in (prev/c/'observations_forecasts.jsonl').read_text().splitlines()]
    start=rr[0]['t'];t=np.array([r['t']-start for r in rr]);oldf=[np.sqrt(r['event_activity']['frequency_squared']) if r['event_activity'].get('frequency_squared') is not None else np.nan for r in old]
    ax.plot(t,oldf,'.',ms=2,color=colors[0],label='原160ms局部频谱')
    for k,n in enumerate(['strict7','aligned_strict7']):
        f=[r['features'][n].get('frequency_hz') if r['features'][n]['qualified'] else np.nan for r in rr]
        ax.plot(t,f,'.',ms=2.5,color=colors[2+2*k],label=['原生严格7','对齐严格7'][k])
    ax.set_title(c);ax.set_xlabel('相对当前窗口起点 / s');ax.set_ylabel('亮度周期候选 / Hz');ax.grid(alpha=.2);ax.set_ylim(0,1250);ax.set_xlim(0,t[-1])
axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'frequency_series79.png',dpi=140);plt.close(fig)
rows=[]
for c in cases+['negative5']:
    r=table[c];values=[c,str(r['parent']),str(r['body_supported']),str(r['original_periodic_supported'])]+[str(r['supported'][n]) for n in names]
    rows.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
synthetic=[]
for r in controls['table']:
    synthetic.append('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ['kind','method','n','accepted','accepted_and_within3percent','median_first_qualified_s'])+'</tr>')
layout=controls['layout'];report=OUT/'第七十九阶段逐像素频率与SDK接口诊断.html'
report.write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>79：逐像素周期输入诊断</title>
<style>body{{font:17px/1.7 "Microsoft YaHei",sans-serif;max-width:1150px;margin:35px auto;padding:0 20px;color:#253248}}h1,h2{{line-height:1.3}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd4df;padding:7px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.lead{{padding:18px;background:#fff0d9}}code{{background:#eee;padding:2px 4px}}a{{color:#235eaa}}</style>
<h1>79：逐像素频率没有普遍补齐周期缺失</h1>
<p class="lead">实际调用SDK并修复本机Python缓冲区步长解释后，严格逐像素方法在清晰/侧视有输出，但在3号、1号和2号片段为零。对齐仅在侧视有少量合格时刻补充，不能按覆盖率将它替换成推力输入。本轮未产生新的未来位置预测。</p>
<p>实际RAW {checks['parent']}父/{checks['native10']}原生10ms窗口，九段含两重复截断前缀；六个主要片段38秒，前缀不是独立样本。全部当前特征在两个真实前缀精确相同。使用原78当前NN框与过去已提交CV速度，提取时禁止读取未来标注/成绩。</p>
<h2>当前支持数量</h2><p>合格表示启发式通过，不能当机械RPM准确率。所有三种原生参数与对齐规则事前固定，未按位置标签调参。</p>
<table><tr><th>片段</th><th>时刻</th><th>当前框</th><th>原频谱</th><th>默认7</th><th>严格7</th><th>短3</th><th>对齐严格7</th></tr>{''.join(rows)}</table>
<img src="coverage79.png"><img src="frequency_series79.png">
<h2>已有工具及数值接口问题</h2>
<p>实际复用<a href="https://docs.prophesee.ai/stable/api/cpp/cv/algorithms.html">Prophesee FrequencyAlgorithm</a>。它估计像素闪烁频率，并非已校准桨叶机械转速。<a href="https://arxiv.org/abs/2211.00198">Frequency Cam</a>与作者本地代码已查阅，本轮没有运行其ROS程序。</p>
<p>本机原生记录大小/步长为{layout['record_bytes']}/{layout['native_stride'][0]}字节，numpy视图却为{layout['numpy_stride'][0]}字节。已知多像素287Hz事件中朴素视图{layout['native_rows']}行有{layout['naive_invalid_rows']}行坐标/频率明显无效。copy=True亦复制错位视图。项目适配只按SDK自己的_buffer_info重建有边界的只读视图并立即复制；已知坐标/频率和整包/分块所有输出字段精确控制通过，没有修改或分发SDK二进制。</p>
<p>首轮只在人工Poisson负控制中发现默认宽容差输出请求频带外1212–2553Hz等值，严格断言终止。原协议和失败明细保留；实际RAW前第二版明确筛选60–1200Hz并计排除数，重新冻结。这与步长解释错误是两个问题。</p>
<h2>人工控制：机制成立，真实可靠性仍未知</h2>
<p>静止周期、末段切频均12/12接受且相对误差&lt;3%。已知平移周期中对齐严格7为12/12，原生严格7仅3/12。Poisson、有色随机、突发和空间共同闪烁在本组末时刻均0/12接受；不能据此宣称所有真实背景都能排除。最早接受时间受频率、像素驻留时间和框覆盖影响，不能把20–30ms的静止人工结果当作无人机实测延迟。</p>
<details><summary>完整人工控制表（含失败）</summary><table><tr><th>类型</th><th>方法</th><th>例数</th><th>末时刻接受</th><th>接受且误差&lt;3%</th><th>首次合格中位s</th></tr>{''.join(synthetic)}</table></details>
<h2>目前卡在哪里</h2><p>逐像素需要同一像素持续看到重复周期；投影平移、旋转、遮挡、噪声和多桨混合会破坏条件。固定过去速度只近似补偿平移，不能补偿桨叶自身旋转或未知机体转动。真实无转速真值，当前候选与原频谱相近也不代表正确。四路诊断及160ms回放合算开销较大，当前不能称实时替代。下一步80优先减少原流程重复计算并保持49路输出精确。</p>
<p><a href="protocol.json">冻结协议</a> · <a href="controls79.json">全部控制</a> · <a href="checks79.json">实际RAW检查</a> · <a href="attempt1_range_failure.json">首次失败记录</a> · <a href="../stage78_RAW_separated_forecast_20261021/第七十六至七十八阶段状态隔离与预测修正报告.html">既有位置预测/视频</a></p></html>''',encoding='utf-8')
(OUT/'delivery79.json').write_text(json.dumps(dict(report=str(report),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    no_new_future_forecasts=True,default_promoted=False,mechanical_RPM_validated=False),ensure_ascii=False,indent=2),encoding='utf-8')
print(report)
