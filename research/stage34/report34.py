import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study34 import OUT,ROOT,CASES,METHODS,load,save,verify


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');summary=[];fig,axs=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
    for case in CASES:
        c=load(OUT/case/'completion.json')
        for method in METHODS:summary.append(dict(case=case,method=method,frames=c['frames'],**c['methods'][method]))
    for ax,case in zip(axs,['side6','train3']):
        rr=list(map(json.loads,(OUT/case/'sensing.jsonl').read_text(encoding='utf-8').splitlines()))
        for method in METHODS:ax.step([r['t'] for r in rr],[int(r['methods'][method]['four_frequencies']) for r in rr],where='post',label=method)
        ax.set(title=case,ylabel='four frequencies internally valid');ax.grid(alpha=.2);ax.legend()
    axs[-1].set_xlabel('RAW time (s)');fig.savefig(OUT/'frequency_availability.png',dpi=150);plt.close(fig);save(OUT/'availability_summary.json',summary)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三十四阶段：转子占位与频率预测</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1300px;margin:auto;background:white;padding:32px}h1{font-size:30px}img{max-width:100%}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#b88726}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;white-space:nowrap;text-align:left;border-bottom:1px solid #dae4ee}th{background:#eef3f7}details{padding:15px;border:1px solid #dae4ee;margin:18px 0}a{color:#176c99}</style><main><h1>第三十四阶段：清出旧轨迹名额后，频率和预测有无改善</h1><p>2026-10-18 · 真实因果RAW频率重算 · 同一当前几何 · 清晰2s+侧视5s+train3 6s</p><div class="note"><b>修复了可用性瓶颈，几乎没有改善未来位置。</b>侧视四频内部有效72→130时刻，train3 0→47；但侧视0.4s全模型混合135.25→135.37px略差，train3 209.49→209.45几乎不变。不能用频率有值率代替预测精度。</div><h2>1. 单独改变了什么</h2><p>旧RotorSensing总共只能保留四条轨迹，旧未匹配轨迹可占位300ms，当前新候选会因满槽无法出生。本轮对未匹配、且距最后实测至少80ms的旧轨迹，仅在当前有新候选等待时腾位；代价、匈牙利和频率算法均沿用原实现。新ID清空旧频率，整机频率平方参考重置，物理特征/参数也随校准epoch重建，位置/历史与模型选择器保留。80ms来自旧重捕获参数，没有按本轮结果搜索。</p><p>每个RAW截止只消费该时刻已冻结的当前桨盘观测，双分支用相同真实RAW历史估频。1300时刻原频率结果和全部15预测，逐项回归12/29/31；六历史专家在两分支完全相同。这个实验隔离频率关联，未改变当前检测，所以不会补回train3后9个漏定位。</p><h2>2. 可用性与代价</h2>'''
    text+=table(['片段','方法','时刻','当帧四轨迹','四频有效','物理生效','提前退役','物理重置','新ID'],[[r['case'],r['method'],r['frames'],r['four_visible_ticks'],r['four_frequency_ticks'],r['physics_active_ticks'],r['retirements'],r['physics_resets'],r['new_ids']] for r in summary])
    text+='<img src="frequency_availability.png"><p>新ID周转更大：侧视20→38，train3 9→16；保守的连续性和新观测接入有取舍。内部频率门槛不是真实RPM精度，新增频率可能仍有谐波、错ID与运动补偿误差；没有转速真值，不称测量更准确。</p><h2>3. 相同目标上的未来评分</h2>'
    for case in CASES:
        text+=f'<details'+(' open' if case!='clear6' else '')+f'><summary>{case}</summary>'+table(['方法','模型','名义s','有效/目标','均值FDEpx','最大px','联合通过'],[[r['frontend'],r['model'],r['nominal_h'],f"{r['valid_forecasts']}/{r['present_targets']}",fmt(r['mean_fde_px']),fmt(r['max_fde_px']),r['joint_success']] for r in a['table'] if r['case']==case and r['model'] in ['cv','adaptive_axis_freq','adaptive_all_mix']])+'</details>'
    text+='<div class="note warn">各片段评分窗口与阶段29全5/12秒不同。这里clear0.4s只有2/3有效、side5/6、train3 3/12；不要拿本轮均值与更长窗口均值相减。train3单物理208.52→207.20px也未解决其转向和定位中断。双方覆盖相同，全部预测保存后才读取存档框。</div><h2>4. 取舍</h2><p>不把此分支直接升级旧默认。占位修复可作为下一步关联候选，但应保留新ID/校准变动的显式通知；更频繁重置会使在线物理参数更难稳定。先解决整机当前观测与桨盘分组，再验证频率真实性，不能靠放宽四频门槛宣布完成。</p><p>下一步针对事件画面中“整机还在，但不再有可靠四盘”的状态，测试已有光流/事件特征跟踪补测；只用原始ON/OFF计数，不用重建G，区分当前图像证据与纯外推。强转向的未知未来控制也应保留短跨度/多假设输出，不承诺0.4秒必然准确。</p><p><a href="protocol.json">冻结协议</a>、<a href="analysis.json">完整未来评分</a>、<a href="availability_summary.json">可用性/ID代价</a>、<a href="阶段结论.md">结论</a>。无G/yolov10-main访问，旧默认和src保留。</p></main></html>'
    (OUT/'第三十四阶段转子占位频率与预测报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段34结论\n\n原当前几何不改，真实RAW重算双分支频率。1300个原频率/15模型完全回归，六历史专家相同。侧视四频72→130、生效150→177；train3四频0→47、生效0→59；clear保持。\n\n新ID侧视20→38/train3 9→16，物理校准重置23/11次。side0.4s全混合135.25→135.37px，train3 209.49→209.45，未形成有意义预测收益。可用性修复不等于RPM/姿态精度或未来运动改善。当前漏定位和转向误差仍在。\n\n不改旧默认；下步整机事件特征/光流补测，保留独立当前证据门槛及失败保护，再谈转速真实性和物理贡献。\n',encoding='utf-8')
    # Association-only stage33 report keeps its narrower scope.
    old=ROOT.parent/'output/stage33_rotor_slot_association_20261018';b=load(old/'analysis.json')
    text33='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第三十三阶段关联占位</title><style>body{font:17px/1.8 "Microsoft YaHei",sans-serif;max-width:1200px;margin:35px auto;padding:20px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd}.scroll{overflow:auto}</style><h1>第三十三阶段：旧转子轨迹占位的独立关联消融</h1><p>只重播缓存桨盘，历史事件输入为空，未执行频率估计。原2299时刻ID/几何完整回归；新增仅在新候选等待且旧ID未匹配80ms时腾位，matched ID不退役，新ID不继承频率/推力参考。</p>'
    text33+=table(['片段','方法','时刻','前端四桨','同时四轨迹','当前未全接住','提前退役','新ID'],[[r['case'],r['method'],r['frames'],r['frontend_four'],r['four_visible'],r['unrepresented_current'],r['retired'],r['new_ids']] for r in b['table']])
    text33+='<p>侧视四轨迹49→95，train3 1→7，clear435保持；ID周转侧视40→82、train3 9→16。这里没有真ID/RPM评测，不能声称身份正确或转速准确。实际频率与预测后验验证见<a href="../stage34_slot_frequency_forecast_20261018/第三十四阶段转子占位频率与预测报告.html">阶段34</a>，可用性增加而预测几乎不变。</p><p>查阅<a href="https://github.com/tryolabs/norfair/blob/master/norfair/tracker.py">Norfair作者出生/失效实现</a>，实际复用本地RotorSensing与SciPy，未执行Norfair，不称新关联算法。<a href="protocol.json">协议</a>、<a href="analysis.json">结果</a>、<a href="module_checks.json">检查</a>。无G/yolov10-main访问。</p></html>'
    (old/'第三十三阶段转子占位关联报告.html').write_text(text33,encoding='utf-8')
    (old/'阶段结论.md').write_text('# 阶段33\n\n2299时刻关联原ID/几何精确回归；四轨迹side49→95、train3 1→7、clear435保持。ID周转side40→82，存在代价。本轮不执行频率，不报RPM/预测增益。实际RAW后验见34，预测几乎不改善。\n',encoding='utf-8');print('Stages33/34 reports rendered',flush=True)


if __name__=='__main__':main()
