import collections,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study29 import OUT,ROOT,CASES,load,save,verify
from select29 import EXPERTS


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');diag=[];fig,axs=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
    for ax,case in zip(axs,CASES):
        rr=[r['frontends']['delayed_weights'] for r in map(json.loads,(OUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines())];tt=np.array([r['t'] for r in rr]);sd=[r['selection_diagnostic'] for r in rr]
        for model in EXPERTS:ax.plot(tt,[d['weights']['all'][model] for d in sd],label=model)
        ax.set(title=case,ylabel='online weight',ylim=(0,1));ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
        counter=[d['updates'] for d in sd]
        diag.append(dict(case=case,frames=len(rr),total_feedback_updates=sum(max(0,n-(counter[i-1] if i else 0)) for i,n in enumerate(counter)),
            feedback_ticks=sum(d['feedback_used'] for d in sd),valid_forecast_ticks=sum(r['valid'] for r in rr),
            selected_expert_counts=dict(collections.Counter(d['chosen_expert'] for d in sd if d['chosen_expert'] is not None)),
            physics_active_ticks=sum(r['feature_diagnostic']['active'] for r in rr),final_weights=sd[-1]['weights']))
    axs[-1].set_xlabel('RAW time (s)');fig.savefig(OUT/'online_weights.png',dpi=150);plt.close(fig);save(OUT/'weight_diagnostics.json',diag)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十九阶段：延迟在线模型权重</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6fa;color:#243342}main{max-width:1300px;margin:auto;background:white;padding:32px}h1{font-size:30px}h2{margin-top:35px}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#b88726}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #dae4ee;white-space:nowrap;text-align:left}th{background:#eef3f7}img{max-width:100%}details{padding:15px;border:1px solid #dae4ee;margin:18px 0}a{color:#176c99}</style><main><h1>第二十九阶段：用到期预测误差调整下一次模型权重</h1><p>2026-10-18 · 同一定位/十五预测器 · 因果反馈 · 先保存预测，再读存档框</p><div class="note"><b>在线混合有开发集收益，物理分支的额外价值仍未证实。</b>0.4s清晰CV21.89→历史自适应混合15.45/全模型15.05px(7/8)；侧视CV127.62→历史94.79/全模型100.17px(12/14)。侧视加物理专家反而逊于只混合历史。</div><h2>1. 机制与来源</h2><p>重用既有paper_bank.py的延迟指数加权规则：保留过去预报，等0.2秒后新位置实际到来，按到期误差更新权重；等待超过0.26秒丢弃，当前观测置信度须&gt;0.5。损失按35px平方归一并截到16，学习率0.08、均匀回流1%，均沿用原代码常数。CV初始0.9、其他均分剩余0.1，未按本轮结果调参。</p><p>历史组为CV、限幅加速度、稳健收缩；全模型组再加自适应姿态/姿态频率专家。比较固定初始权重、在线混合及最大权重硬选。全部专家当前位置/速度和有效性相同，物理轴固定阶段28canonical_static，原十模型系数完全不变。</p><p>反馈参考是新到的当前检测中心，不是评分框或飞控真值；当前反馈只影响从此刻向未来的新预测。用独立粗框端点评分检验其实际作用。这个模式已有<a href="https://doi.org/10.1006/jcss.1997.1504">Freund与Schapire的在线指数权重研究</a>和<a href="https://riverml.xyz/0.17.0/api/overview/">River EWARegressor</a>等实现，不称新算法。未安装/执行River，实际复用本地现成权重规则。</p><h2>2. 全部目标与失败</h2>'''
    selected=['cv','trusted_acc','adaptive_axis_freq','fixed_history_mix','adaptive_history_mix','fixed_all_mix','adaptive_all_mix','adaptive_pick']
    for case in CASES:
        text+=f'<details open><summary>{case}</summary>'+table(['模型','名义s','有效/目标','均值FDEpx','中位px','最大px','联合通过'],[[r['model'],r['nominal_h'],f"{r['valid_forecasts']}/{r['present_targets']}",fmt(r['mean_fde_px']),fmt(r['median_fde_px']),fmt(r['max_fde_px']),r['joint_success']] for r in a['table'] if r['case']==case and r['model'] in selected])+'</details>'
    text+='<div class="note warn">收益不是所有跨度都成立：侧视0.2s CV81.74px，全模型混合82.13、历史混合82.65、硬选85.01，均未改善；清晰0.1s CV10.20，全模型10.36略差。短期到期误差不必然代表更长跨度的正确模型，不能把0.4秒的局部提升推广到所有任务。</div><h2>3. 权重如何变化</h2><img src="online_weights.png">'
    text+=table(['片段','时刻','有效预报','实际反馈更新','物理生效','硬选次数'],[[r['case'],r['frames'],r['valid_forecast_ticks'],r['total_feedback_updates'],r['physics_active_ticks'],r['selected_expert_counts']] for r in diag])
    text+='<p>物理特征缺失时，物理专家本来回退CV，因此它的高权重也可能只是重复CV，不能当作推力测量有效的证据。图中跨断轨会重置权重，更新总数已按每段累加。重叠预测与检测噪声相关，未声称满足理想在线学习的理论界或概率置信保证。</p><h2>4. 边界与接下来</h2><p>合成可预测专家增权、无到期样本不更新、权重正且归一、缺测重置、前缀一致、原系数完全保持通过；真实两段各80时刻前缀重复一致。每个模型有相同覆盖，评分使用存档近似整机框，而不是内部到期损失。开发数据少、无独立姿态/RPM真值，仍需跨录像框评测。</p><p>下一步把原生RAW解码、逐桨频率、固定符号轴、前向推力与在线权重接成独立入口，重新读取短RAW验证观测/频率/十五模型与缓存一一相同，给出可观看视频。完成整合不等于获得新的精度证据。</p><p><a href="protocol.json">协议</a>、<a href="analysis.json">完整逐目标评分</a>、<a href="weight_diagnostics.json">权重诊断</a>、<a href="module_checks.json">检查</a>、<a href="阶段结论.md">结论</a>。无G/yolov10-main访问，旧默认保持。</p></main></html>'
    (OUT/'第二十九阶段延迟在线模型选择报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段29结论\n\n0.4s clear同7/8：CV21.89，固定历史19.99，自适应历史15.45，固定全19.30，自适应全15.05，硬选15.81px。side同12/14：CV127.62，固定历史121.26，自适应历史94.79，固定全124.31，自适应全100.17，硬选101.16。在线加权局部有效；侧视物理专家不如只用历史。\n\n失败：side0.2s CV81.74→自适应全82.13/历史82.65/硬选85.01，clear0.1s CV10.20→全10.36。0.2秒到期排名不保证0.4秒正确。所有原十模型系数/有效性不变，当前检测只在过去预测到期后作在线反馈，存档框仅事后评分。非新算法，无独立RPM/姿态真值，尚非盲测。\n',encoding='utf-8');print('Stage29 report rendered',flush=True)


if __name__=='__main__':main()
