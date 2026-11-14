"""Summarize the completed angular-phase prototype without promoting it."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'output/shared_angular_phase_20261027'
audit=json.loads((out/'audit.json').read_text('utf-8'))
summary=json.loads((out/'raw_directional/summary.json').read_text('utf-8'))
doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>五圆共享平面与批量空间相位</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:980px;margin:35px auto;padding:0 22px;background:#f3f6f8;color:#233443;line-height:1.8}section{background:white;padding:20px 25px;border-radius:10px;margin:18px 0}h1{font-size:28px}h2{font-size:20px}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px}a{color:#226591}.note{background:#fff3d5;padding:12px}</style>
<h1>五圆共享平面 → 批量空间相位</h1><p>2026-10-27 · 独立实验模块 · 未更改默认预测输入</p>
<section><h2>为什么做这一步</h2><p>只看像素明暗周期，难以区分桨叶旋转、背景闪烁和倍频。这里让五圆模型负责共同几何，检查“事件发生的角度”和“时间”是否符合旋转关系。用批量复数矩和FFT替代逐事件的重型相位更新。</p><p>只测当前实际支持的模型槽，隐藏桨不填频率；侧视平面病态时拒绝空间相位输出，保留原来的侧视时间周期路线。没有重建灰度/G通道。</p></section>
<section><h2>当前结果</h2><p>相同两秒RAW、60个源时刻：6次四路空间相位同时合格，共47个当前逐桨频率。与同一模型槽匹配的SDK像素周期，47/47处于同倍频的10%范围内，平均相对差约1.10%。</p><p class="note">两者来自同一事件源，没有独立RPM真值。1.10%是两种事件估计之间的差异，不是真实转速误差；尚未证明改善位置预测。</p><img src="phase_tracks.png" alt="各模型槽的当前相位频率与集中度"><p>曲线空档表示当前证据不足；160ms滚动窗口可能产生滞后，平滑不等于变向反应更快。</p></section>
<section><h2>你提出的“机体不动、平面波动”</h2><p>合成控制中让机体中心保持固定、四桨平面共同摆动，设定的100Hz桨叶相位仍估计约99.977–99.980Hz。共同几何变化没有直接被当作转速变化。随机事件时序被拒绝，隐藏槽没有填值。这个实验只检查几何/相位解耦，不证明力学上悬停平衡。</p></section>
<section><h2>新增反例修复：照明闪烁不能算旋转</h2><p>非均匀静态圆环随照明闪烁、某一亮区很强时，旧峰噪/相位集中度/扇区门槛会误接纳4路。加入正反频率至少3:1的能量判据后，反例0路接纳；同一真实RAW的47个合格结果全部保留。原理是静态图案共同闪烁的复数频谱正反对称，真正旋转的空间矩应偏向一个方向。</p><p>3:1是固定实验判据，不是已校准概率，也不能排除所有复杂干扰。</p></section>
<section><h2>计算可行性</h2><table><tr><th>方案</th><th>模块均值ms</th><th>四路合格时刻</th></tr><tr><td>本地作者核心，串行、全像素</td><td>344.39</td><td>5</td></tr><tr><td>作者核心，25%像素、关闭重复姿态求解</td><td>87.15</td><td>2</td></tr><tr><td>批量空间相位，25%像素，当前运行</td><td>16.11</td><td>6</td></tr></table><p>这是模块工程试验，初始化与计算条件不同，不能当成论文公平性能排名。完整时间链、变向精度与真实RPM仍未评分。</p><p>批量模块单线程环境不同运行均值9.81/16.77ms，本次16.11ms、P9546.95ms，运行负载影响明显。数学等价地移除逐事件排序后，60时刻输出相同；48k事件×15交替微测3.389→2.069ms，只是这个操作的收益。</p></section>
<section><h2>原作者核心的接口问题</h2><p>严格完整椭圆入口在本段零初始化，因为当前输入多是合格残弧。把这些拟合形状明确作为初始化假设后，并发小试有一次本地access violation；串行小试完成。确切根因尚未定位，不把“串行成功”写成已证明根因。所有试验独立进程结束，默认RAW预测不受影响。</p></section>
<section><h2>接下来</h2><p>先做可见槽变化时的逐桨基准归一化：即使各桨转速未变，从四桨变成三桨，直接求可见转速平方均值也可能跳变。需要去掉这种可见集合偏差，再判断新空间相位是否给推力和变向预测带来额外收益。未证明价值前不急着增加GPU复杂度。</p></section>
<section><h2>来源与可复现记录</h2><p>原理参照 <a href="https://arxiv.org/abs/2603.09235">HelixTrack</a> 的空间相位、<a href="https://arxiv.org/abs/2610.31384">CoHelix</a> 的共享几何。新模块为自写NumPy/SciPy实现，无作者DLL依赖，不宣称原理首创。</p><p><a href="audit.json">排序移除/周期相互印证审计</a> · <a href="controls_directional_final/controls.json">最新合成控制</a> · <a href="raw_directional/summary.json">当前RAW模块结果</a> · <a href="raw_directional/protocol.json">输入与源哈希</a> · <a href="../../event_drone_forecast/docs/SHARED_ANGULAR_PHASE.md">方法与运行说明</a> · <a href="../three_continuation_20261026/index.html">上一批已完成的默认预测结果</a></p></section></html>'''
(out/'index.html').write_text(doc,'utf-8')
(out/'summary.json').write_text(json.dumps(dict(module_result=summary,audit=audit,production_default_changed=False,
    independent_RPM_truth=False,forecast_advantage_proven=False),indent=2),'utf-8')
print(out/'index.html')
