import html,json
import numpy as np
from study48 import *
def main():
    verify();a=load(OUT/'analysis.json');summary=[]
    for case in CASES:
        rr=rows(OUT/case/'forecasts.jsonl');completion=load(OUT/case/'completion.json')
        for method in METHODS:
            previous={name:0 for name in prior.PHYSICS};counts={name:0 for name in prior.PHYSICS};allowed={name:0 for name in prior.PHYSICS}
            for r in rr:
                q=r['frontends'][method]
                for name,d in q['feature_diagnostic']['models'].items():
                    current=d['updates'];counts[name]+=current-previous[name] if current>=previous[name] else current;previous[name]=current;allowed[name]+=bool(d['update_allowed'])
            c=completion['methods'][method];summary.append(dict(case=case,method=method,frames=len(rr),physics_active=c['physics_active'],resets=c['physics_resets'],associated_dropped=c['association_dropped'],frequency_model_actual_updates=counts['adaptive_axis_freq'],frequency_update_allowed_ticks=allowed['adaptive_axis_freq'],axis_only_actual_updates=counts['adaptive_axis_only']))
    table=[r for r in a['table'] if r['frontend'] in ['legacy_rotors','slotfix_rotors','associated_slotfix'] and r['model'] in ['cv','trusted_acc','adaptive_axis_only','adaptive_axis_freq','adaptive_history_mix','adaptive_all_mix'] and r['nominal_h']==.4 and r['case']!='negative5']
    save(OUT/'corrected_learning_summary.json',dict(table=summary,correction='completion.methods.updates was an unincremented initialization placeholder, not a measured zero. Actual update counts reconstructed from saved per-frame feature_diagnostic.models counters. Original frozen source/results retained.'))
    for r in a['common']:
        if r['case']=='negative5':continue
        assert r['n']==next(x['valid_forecasts'] for x in a['table'] if x['case']==r['case'] and x['frontend']==METHODS[0] and x['model']=='cv' and x['nominal_h']==r['nominal_h'])
    def tab(rr,cols):return '<table><tr>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[c],5) if isinstance(r[c],float) else r[c]))+'</td>' for c in cols)+'</tr>' for r in rr)+'</table>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第四十八阶段整机位置与桨叶物理贡献</title><style>body{font:16px/1.8 Arial;max-width:1380px;margin:30px auto;color:#233e4c}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px;border:1px solid #ccd8df}</style><h1>第四十八阶段：更准位置接入桨叶物理后，仍缺稳定额外收益</h1><p><b>姿态/转速/推力模块已在同一更准整机位置上重新对照，不只停在YOLO中心预测。</b>13秒正例650个20ms时刻，加1秒空场50，共700。位置/速度/覆盖与45 body_first固定一致，三个传感器策略均逐项保持六历史系数。用34真实RAW已保存10ms逐桨频率，在相同当前cutoff取样；没有用将来频率。当前轴/位置更新20ms，所以不拿它与老10ms系统或其它窗口均值直接比较。</p><p>复用原28法向、27投影推力/RLS与29延迟模型权重，只比较原转子ID、80ms换槽、换槽加整机框关联。关联框扩大25%，仅约束现有测量转子，绝不造出桨叶/法向/RPM。所有15预测先保存，再评分确切未来稀疏框。</p><h2>实际可用与学习</h2>'''+tab(summary,['case','method','frames','physics_active','resets','associated_dropped','frequency_model_actual_updates','frequency_update_allowed_ticks','axis_only_actual_updates'])+'''<p>clear100中75物理生效，真实参数更新58。side250中生效75→88，换槽20次物理重置，换槽频率模型实际更新0；train3 300中0→30生效，10次重置、实际更新仍0。<b>可用率上升未形成连续足够长的学习窗口。</b>旧模型每次重置有0.6秒预热，还要满足四可见/质量/到期反馈，这比“有没有一帧姿态”更决定在线参数能否学起来。</p><p>completion里的updates字段是初始占位，脚本未递增，不能把它当清晰更新0。本表从保存的每帧model诊断计数重算并另存corrected_learning_summary，原冻结结果保留。</p><h2>0.4秒确切未来端点</h2>'''+tab(table,['case','frontend','model','present_targets','valid_forecasts','mean_fde_px','joint_success'])+'''<p>clear仅2/3有效，CV14.3077→姿态+频率14.2240px，极小且样本少；side同5/6，CV124.93→124.36，但纯历史收缩81.91更好；train3同11/12，CV137.05→136.69，纯历史收缩110.32更好。全部模型/方法覆盖完全一致，共同目标就是同样本。关联去掉side85/train3 34个槽内外转子记录，但生效数没变，未来误差也未获得稳定额外提升。空场50当前/预报零。</p><h2>现在具体卡在哪里</h2><p>整机定位恢复，不等于桨叶频率/法向持续可观测。频率换槽会改变归一化参考，当前做法重置整个物理模块，连不依赖RPM参考的轴向模型也丢掉已学状态；另一个限制是轴向模型为了旧受控消融仍共享频率有效门槛。下一步应先研究/验证标定参考变化时的参数与协方差坐标变换、或轴向独立准入，而非只继续加检测框。</p><p>这里是从原生RAW缓存做的同状态贡献试验，尚未把新NN/共享内存入口接成完整实时物理系统。输出是有效像素动力学，不是真质量、推力系数或风速识别。没有独立RPM/姿态真值；不能把框误差当成论文的机械频率误差。</p><p>依据<a href="https://arxiv.org/abs/2604.18289">Relative State Estimation using Event-Based Propeller Sensing</a>的椭圆/频率作为推力输入与位置校正思路，执行本地独立像素适配，不是作者代码或其3D结果复现。<a href="corrected_learning_summary.json">真实学习计数</a> · <a href="analysis.json">全部未来与共同目标评分</a> · <a href="protocol.json">冻结协议</a></p></html>'''
    (OUT/'第四十八阶段整机位置与桨叶物理贡献报告.html').write_text(text,encoding='utf-8');(OUT/'阶段结论.md').write_text('700时刻同更准位置重评物理：clear58更新、side换槽20重置/train3 10重置导致实际频率模型0更新。可用增加但物理额外收益小，未胜纯历史收缩。六历史/覆盖逐项保持；更新占位字段已依逐帧诊断重算。下步标定参考变换/轴向独立准入，非新算法声明。',encoding='utf-8');print('Stage48 physics report complete',flush=True)
if __name__=='__main__':main()
