"""Render isolated natural-layout results without changing frozen algorithms."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study26 import OUT,ROOT,CASES,METHODS,load,save,verify


def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def fmt(x):return '—' if x is None else f'{x:.2f}'


def main():
    verify();a=load(OUT/'analysis.json');det=load(OUT/'detection_analysis.json');diagnostics=[]
    fig,axs=plt.subplots(3,1,figsize=(13,9),constrained_layout=True)
    for ax,case in zip(axs,['clear6','side6','review4']):
        rows=list(map(json.loads,(OUT/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()))
        times=np.array([r['t'] for r in rows]);dist=[];matched=[];changed=0;partial=0
        for r in rows:
            b=r['methods']['neighbor20']['observation'];l=r['methods']['layout20']['observation'];d=r['methods']['layout20']['layout']
            delta=float(np.linalg.norm(np.asarray(b['center'])-l['center'])) if b['detected'] and l['detected'] else np.nan
            dist.append(delta);matched.append(bool(d.get('matched')));changed+=bool(np.isfinite(delta) and delta>1e-6)
            partial+=bool(d.get('matched') and d.get('matched_count',4)<4)
        ax.plot(times,dist,color='#317b9e',label='center difference (not accuracy)')
        ix=np.flatnonzero(matched);ax.scatter(times[ix],np.zeros(len(ix)),s=6,color='#e68d2e',label='layout match')
        ax.set(title=case,ylabel='pixels');ax.legend(loc='upper right');ax.grid(alpha=.2)
        diagnostics.append(dict(case=case,center_changed_ticks=changed,layout_matches=sum(matched),partial_match_ticks=partial))
    axs[-1].set_xlabel('RAW time (s)');fig.savefig(OUT/'layout_effect.png',dpi=140);plt.close(fig);save(OUT/'layout_diagnostics.json',diagnostics)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十六阶段：真实片段布局独立消融</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1250px;margin:auto;background:white;padding:32px}h1{font-size:30px}h2{margin-top:35px}.note{background:#edf6fb;border-left:5px solid #3787a7;padding:18px}.warn{background:#fff4dd;border-color:#bd8c28}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{white-space:nowrap;text-align:left;padding:8px;border-bottom:1px solid #dae4ee}th{background:#eef3f7}img,video{max-width:100%}details{padding:15px;border:1px solid #dae4ee;margin:15px 0}a{color:#176c99}</style><main><h1>第二十六阶段：四点布局在真实片段中的独立作用</h1><p>2026-10-18 · 四段35秒/1750时刻 · 同一20ms前端 · 无周期增强</p><div class="note warn"><b>未形成自然侧视预测的稳定提升。</b>受控删减中的局部收益不能直接推广：真实侧视发布409→411，标签覆盖仍13/14，条件中心误差27.97→28.81px；0.4s收缩加速度FDE71.32→74.77px。保留原默认。</div><h2>1. 本轮隔离了什么</h2><p>阶段19同时变化周期增强与布局，不能只凭其结果判断布局。本轮两路均使用原neighbor20计数/几何，只给layout20加阶段19有界布局。连续3次实际非退化四桨建立模型，距完整观测300ms失效，部分匹配不续期。预测仍是阶段23六模型，两前端各自因果运行。</p><p>1750个neighbor20当前观测与阶段21逐项相同；全部基线预测记录与阶段23相同。没有放宽尺度/期限来追求分数。既有stage11拟合/取向、匈牙利关联、stage19模板与stage22评分直接复用。</p><h2>2. 当前检测</h2>'''
    text+=table(['片段','前端','发布/时刻','有机覆盖','条件误差px','联合通过','无机标签发布','布局匹配'],[[r['case'],r['method'],f"{r['published']}/{r['frames']}",f"{r['present_detected']}/{r['present_labels']}",fmt(r['conditional_error_px']),r['joint_success'],f"{r['absent_published']}/{r['absent_labels']}",r['layout_matches']] for r in det['table']])
    text+='<h2>3. 未来位置</h2><p>参考存档粗框中心；按过去锚点在确切目标时间求位置，实际跨度为名义值再加0–25ms。FDE是图像像素误差，不是ADE/米制轨迹。均值须连同有效覆盖和最大误差一起阅读。</p>'
    for case in ['clear6','side6','review4']:
        text+=f'<details'+(' open' if case=='side6' else '')+f'><summary>{case}</summary>'
        text+=table(['前端','模型','跨度s','有效/目标','均值FDE','最大FDE','联合通过'],[[r['frontend'],r['model'],r['nominal_h'],f"{r['valid_forecasts']}/{r['present_targets']}",fmt(r['mean_fde_px']),fmt(r['max_fde_px']),r['joint_success']] for r in a['table'] if r['case']==case and r['model'] in ['cv','huber_acc','trusted_acc']])+'</details>'
    text+='<h2>4. 卡点</h2><img src="layout_effect.png"><p>图是模型对输出的影响，不是真值误差。清晰主要四点齐全，模板没有增加信息；侧视匹配36/600时刻、train4仅5/500，长期的几何塌缩/分组失败仍未解决。两点约束只能支持局部近似形变，不能凭一帧无限恢复三维姿态。稀疏标签仍可能错过短时有益窗口，本轮同时保留受控部分观测证据，不宣判布局思路无效。</p><p>下一步回到核心物理问题：在相同位置/速度状态上单独比较姿态、相对频率平方及在线误差修正，避免将定位变化与动力学贡献混在一起。</p>'
    if (OUT/'cli_smoke/forecasts.mp4').exists():text+='<h2>5. RAW独立入口</h2><video controls src="cli_smoke/forecasts.mp4"></video><p>绿色当前框、蓝色CV、橙色研究预测；视频按源50fps，不代表计算实时。<a href="cli_smoke/regression.json">入口回归</a>。</p>'
    text+='<p><a href="protocol.json">冻结协议</a>、<a href="analysis.json">完整端点评分</a>、<a href="detection_analysis.json">当前检测</a>、<a href="layout_diagnostics.json">影响诊断</a>、<a href="阶段结论.md">结论</a>。无G，未访问或执行yolov10-main；旧src/default与历史结果保留。</p></main></html>'
    (OUT/'第二十六阶段真实布局消融报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段26结论\n\n同一neighbor20，不加周期增强，只加300ms有界布局。清晰当前8/8及8.72px保持；侧视13/14保持、27.97→28.81px，发布409→411；train4 3/5及39.92px保持；负例0/400。\n\n侧视名义0.4s同12/14有效，CV131.00→130.09px，Huber64.66→65.69，收缩71.32→74.77。没有稳定升级证据。1750当前观测和所有原六模型记录与阶段21/23精确回归。阶段25受控删减支持短时局部收益；这里验证其自然泛化仍不足。\n\n完整四桨不足使模板可用时间少，局部相似模型不能推断无限侧飞。下一步隔离姿态/转速/自适应动力学贡献。旧默认不改；参考为稀疏近似整机框，无RPM/姿态真值。\n',encoding='utf-8')
    print('Stage26 report rendered',flush=True)


if __name__=='__main__':main()
