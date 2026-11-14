import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study28 import ROOT,OUT,CASES,METHODS,load,save,verify


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');diag=[];fig,axs=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
    for ax,case in zip(axs,CASES):
        rows=list(map(json.loads,(OUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines()));tt=np.array([r['t'] for r in rows])
        for method in METHODS:
            rr=[r['frontends'][method]['feature_diagnostic'] for r in rows];mm=[r['models']['adaptive_axis_freq'] for r in rr];counters=[m['updates'] for m in mm]
            normals=[m['normal'][1] if m['normal'] is not None else np.nan for m in mm]
            ax.plot(tt,normals,label=method,lw=1,alpha=.85)
            diag.append(dict(case=case,method=method,frames=len(rows),active_ticks=sum(r['active'] for r in rr),
                down_when_active=sum(bool(r['active'] and m['normal'] is not None and m['normal'][1]>0) for r,m in zip(rr,mm)),
                total_update_events=sum(max(0,n-(counters[i-1] if i else 0)) for i,n in enumerate(counters))))
        ax.axhline(0,color='black',lw=1,ls='--');ax.set(title=case,ylabel='normal y (+down)');ax.grid(alpha=.2);ax.legend()
    axs[-1].set_xlabel('RAW time (s)');fig.savefig(OUT/'normal_comparison.png',dpi=150);plt.close(fig);save(OUT/'axis_diagnostics.json',diag)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十八阶段：轴向符号与传播消融</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6fa;color:#243342}main{max-width:1250px;margin:auto;background:white;padding:32px}h1{font-size:30px}h2{margin-top:35px}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#b88726}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #dae4ee;white-space:nowrap;text-align:left}th{background:#eef3f7}img{max-width:100%}details{padding:15px;border:1px solid #dae4ee;margin:18px 0}a{color:#176c99}</style><main><h1>第二十八阶段：法向为什么会指向图像下方</h1><p>2026-10-18 · 原法向/禁止角速/固定符号静态融合/当帧法向 · 同1699时刻</p><div class="note"><b>定位了符号延续的问题，预测收益仍很小。</b>侧视原分支260个生效时刻中110个向下；只禁止角速仍110，取消跨时刻法向正负翻转后为0。0.4s姿态频率预测129.13→126.55px，CV127.62px；不是精确姿态已恢复的证明。</div><h2>1. 四个分支</h2><p>original逐项重现阶段27；no_rate只在原滤波每步前后把角速度清零，未来保持当前轴；canonical_static用原每帧椭圆/圆反投影与多桨一致性生成测量，再用相同协方差融合，取消跨时刻正负翻转与旋转外推；instantaneous直接使用同一当帧测量、不平滑。两新分支缺观测时只短时保持，轴新鲜度仍160ms。所有推力/学习参数、六历史预测和位置有效性完全不变。</p><p>原weak_normal就使用图像向上和弱透视非负深度分量先验；新分支让时间融合也遵守该先验。原逐时刻选择“离上一个法向更近”的正负号，会在近侧视的横向方向跳变时选取向下解。只关旋转不能消除这条符号路径。</p><h2>2. 先验自洽与学习</h2>'''
    text+=table(['片段','轴方法','时刻','物理生效','生效时向下','整段实际更新'],[[r['case'],r['method'],r['frames'],r['active_ticks'],r['down_when_active'],r['total_update_events']] for r in diag])
    text+='<img src="normal_comparison.png"><div class="note warn">“向下次数为0”是符合所用先验，不是姿态真值准确率。图像向上先验不适用于任意相机滚转、倒飞或所有三维绕视线投影。单目椭圆仍有方向/镜像歧义，新分支不能宣称解决了三维符号可观测性。</div><h2>3. 相同目标上的未来误差</h2>'
    for case in CASES:
        text+=f'<details open><summary>{case}</summary>'+table(['轴方法','模型','名义s','有效/目标','均值FDEpx','最大px','联合通过'],[[r['frontend'],r['model'],r['nominal_h'],f"{r['valid_forecasts']}/{r['present_targets']}",fmt(r['mean_fde_px']),fmt(r['max_fde_px']),r['joint_success']] for r in a['table'] if r['case']==case and r['model'] in ['cv','adaptive_axis_only','adaptive_axis_freq']])+'</details>'
    text+='<p>清晰0.4s7/8：原姿态频率20.71、no_rate/canonical_static20.65、当帧20.68px，CV21.89。侧视12/14：原129.13、no_rate129.04、canonical_static126.55、当帧126.51，CV127.62。修正先验冲突有局部作用，但误差主要仍由位置历史、预测有效覆盖、频率/姿态可用性及未知未来控制限制。</p><h2>4. 复用与边界</h2><p>查阅<a href="https://www.jstatsoft.org/article/view/v089i04">libDirectional原始论文</a>、<a href="https://github.com/libDirectional/libDirectional">作者MATLAB仓库</a>和<a href="https://users.aalto.fi/~ssarkka/pub/ctvmf-conf.pdf">S²参考向量滤波论文</a>。这些已有方法专门处理球面/轴对称不确定性，本轮未运行它们；实际复用本地weak_normal、原测量一致性和协方差计算，只做简单可解释消融，不称新方向滤波算法。</p><p>所有1699个original完整记录与阶段27相同；各变体全部六历史系数相同、有效性相同，前缀和符号/单位向量检查通过。稀疏开发框端点按确切未来时刻评分，无密集ADE/真实RPM/姿态真值。下一步检验已有延迟在线模型加权，观察能否自动减少固定选错模型的损失；不得用未来评分框调权。</p><p><a href="protocol.json">协议</a>、<a href="analysis.json">完整评分</a>、<a href="axis_diagnostics.json">轴诊断</a>、<a href="module_checks.json">检查</a>、<a href="阶段结论.md">结论</a>。无G，未访问执行yolov10-main，历史src和默认保持。</p></main></html>'
    (OUT/'第二十八阶段轴向符号与传播报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段28结论\n\n原法向侧视生效时110次向下；no_rate仍110，canonical_static/instantaneous为0。主要符号延续路径可通过自洽当帧先验排除，不是只关角速就解决。\n\n侧视0.4s同12/14：CV127.62，original129.13，no_rate129.04，canonical_static126.55，当帧126.51px。清晰7/8：CV21.89，original20.71，canonical20.65px。没有强精度改善证据。遵守向上先验不等于真实姿态准确，任意相机滚转/倒飞仍不适用。\n\n1699个original完整记录精确回归27，六历史模型全部保持；参数未根据28输出调整。下一步已有延迟权重模型选择；保持独立框评估，不把在线检测代理误差当真值。\n',encoding='utf-8');print('Stage28 report rendered',flush=True)


if __name__=='__main__':main()
