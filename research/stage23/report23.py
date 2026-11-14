"""Robustness result table; all-arm validity remains identical to stage22."""
import collections,json
from study23 import OUT,MODELS,FRONTENDS,CASES,load,verify,save


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');diagnostic=[]
    for case in CASES:
        rr=list(map(json.loads,(OUT/case/'forecasts.jsonl').read_text().splitlines()))
        for front in FRONTENDS:
            counter=collections.Counter(r['frontends'][front]['robust_diagnostic']['reason'] for r in rr)
            diagnostic.append(dict(case=case,frontend=front,reasons=dict(counter)))
    save(OUT/'fit_diagnostics.json',diagnostic)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十三阶段：稳健加速度预测</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6fa;color:#243342}main{max-width:1250px;margin:auto;padding:32px;background:white}h1{font-size:30px;line-height:1.3}h2{margin-top:36px;border-bottom:2px solid #dce5ed}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#c58b26}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;white-space:nowrap;border-bottom:1px solid #dce4ec}th{background:#edf2f7}details{border:1px solid #dce4ec;padding:15px;margin:18px 0}summary{font-weight:bold;cursor:pointer}a{color:#176c99}</style><main><h1>第二十三阶段：降低历史加速度的离群过冲</h1><p>2026-10-18 · 同一份定位/有效性 · 六预测器 · 先预测再读存档框评分</p><div class="note"><b>局部收益明确，但还不是统一升级。</b>native20侧视名义0.4s，原历史加速度平均293.03px、最大2720.76px；Huber拟合降至64.66/189.85px。native10清晰段仍有过冲，必须继续保留。</div><h2>1. 消融设计</h2><p>hold、CV、原历史加速度完全复用阶段22；新增bounded_acc只给原加速度加3000px/s²范数上限，huber_acc只换SciPy Huber拟合，trusted_acc再用近似拟合方差收缩并限幅。它们的CV位置/速度、观测、有效发布时刻完全相同，所以本轮模型对照不存在通过删预测提高均值的问题。</p><p>Huber门槛1.345×原6px测量噪声、至少6点且160ms支撑；用加权设计矩阵和噪声下限估计方差，正部收缩max(0,1−方差/加速度²)。该协方差是未校准的近似，不是实飞置信区间。3000来自项目既有像素加速度限制，不是物理真值，跨尺度仍有局限。</p>'''
    for case in ['clear6','side6','review4']:
        text+=f'<details'+(' open' if case=='side6' else '')+f'><summary>{case}全部目标评分</summary>'
        for h in [.1,.2,.4]:
            rows=[r for r in a['table'] if r['case']==case and r['nominal_h']==h]
            text+=f'<h3>名义{h:.1f}s，实际最多再长25ms</h3>'+table(['前端','模型','有效/有机目标','均值FDEpx','中位px','最大px','联合通过'],[[r['frontend'],r['model'],f"{r['valid_forecasts']}/{r['present_targets']}",fmt(r['mean_fde_px']),fmt(r['median_fde_px']),fmt(r['max_fde_px']),r['joint_success']] for r in rows])
        text+='</details>'
    text+='<h2>2. 结果怎样解释</h2><p>native20清晰0.4s，CV24.66px，Huber19.87px，收缩限幅18.87px；native20侧视CV131.00px，单限幅100.49px、Huber64.66px、收缩限幅71.32px。稳健损失和限幅的作用可以分开看到。</p><div class="note warn">native10清晰0.4s，CV20.84px，原历史加速度56.01px，Huber仍57.48px，收缩限幅49.48px。不是所有坏预测都由一个离群点造成；拟合偏差、错误定位历史和真实机动都可能存在。train4名义0.4s只有一个有效目标，不能拿其小误差宣称泛化。</div><p>所有结果都在已经用于开发的片段上，下一步应冻结一个实验配置，在尚未用于这轮调参的录像检查，并保留CV。没有在本轮输出后修改参数。</p><h2>3. 等价、因果与实现来源</h2><p>四前端×1750时刻共7000条原hold/CV/历史加速度记录，与阶段22逐项完全一致；所有新模型使用相同有效性。解析二次轨迹、离群点影响、短支持拒绝、前缀一致与限幅不变量通过。评分直接执行冻结阶段22评分函数，仅注入本轮输出路径和模型列表。</p><p>执行<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html">SciPy官方least_squares</a>的Huber损失；没有实现新稳健回归算法。推理未读标签，评分参考存档粗框中心，未用检测器未来输出。没有质量/风速/RPM/姿态真值、没有G/yolov10-main访问，旧src与默认保持。</p><p><a href="protocol.json">协议</a>、<a href="analysis.json">逐目标评分</a>、<a href="fit_diagnostics.json">拟合诊断</a>、<a href="module_checks.json">检查</a>、<a href="阶段结论.md">阶段结论</a>。</p></main></html>'
    (OUT/'第二十三阶段稳健加速度报告.html').write_text(text,encoding='utf-8');print('Stage23 report rendered')


if __name__=='__main__':main()
