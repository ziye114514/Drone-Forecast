"""Report both successful wiring and remaining small/negative gains."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from local_period_tracks import LocalPeriodTracks
BASE=ROOT.parent/'output/source_force168_20261105'


def read(path):return json.loads(path.read_text('utf-8'))
def lines(path):return [json.loads(s) for s in path.read_text('utf-8').splitlines()]


if __name__=='__main__':
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fixed=read(BASE/'native_selection_fixed/scores.json');info=read(BASE/'native_selection_fixed/summary.json')
    single=read(BASE/'single_observer_relativefix/scores.json');native=read(BASE/'native_cache_tracked/scores.json');speed=read(BASE/'native_cache_tracked/summary.json')
    rows=lines(BASE/'native_cache_tracked/predictions.jsonl');track=LocalPeriodTracks()
    # The later unsupported-gauge restart fix must leave this measured run intact.
    for r in rows:assert track.step(r['t'],r.get('local_temporal_phase') or {})==r['local_source_tracking']
    (BASE/'native_trace_current_code_check.json').write_text(json.dumps(dict(frames=len(rows),current_tracker_reproduces_saved_trace=True,
        later_change='unsupported_normalized_gauge_restart',affects_this_run=False,
        current_tracker_sha256=hashlib.sha256((ROOT/'core/local_period_tracks.py').read_bytes()).hexdigest()),indent=2),'utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    clips=['pure_side','pure_train2'];x=np.arange(len(clips));width=.35
    axes[0,0].bar(x-width/2,[info[c]['force_inputs']['legacy_algebra'] for c in clips],width,label='正确解析的原输入')
    axes[0,0].bar(x+width/2,[info[c]['force_inputs']['tracked_algebra'] for c in clips],width,label='局部来源续接')
    axes[0,0].set_xticks(x,['纯桨叶侧视97时刻','纯桨叶另一录像67时刻']);axes[0,0].set_ylabel('具备当前输入资格的时刻数');axes[0,0].set_title('输入连续性改善，不是检测AP／真RPM检出率');axes[0,0].legend()
    period=lines(BASE/'native_selection_fixed/pure_side_periods.jsonl')
    axes[0,1].plot([r['t'] for r in period],[r['continued']['relative_squared_proxy'] for r in period],c='#3379b8')
    axes[0,1].set_title('区域内频率变化形成的相对输入');axes[0,1].set_xlabel('源时刻 / s');axes[0,1].set_ylabel('归一化变化代理，不是机械RPM');axes[0,1].grid(alpha=.2)
    records=native['arms']['native_actual']['records'];zero=native['arms']['zero_same_observations']['records']
    for values,color,label in [(records,'#29945d','完整桨叶输入'),(zero,'#ac7d24','同观测／数值置零')]:
        v=[r for r in values if r['valid']];axes[1,0].plot([r['query_t'] for r in v],[r['error_px'] for r in v],'.-',c=color,label=label)
    axes[1,0].set_title('正式入口原生缓存：误差曲线几乎重合');axes[1,0].set_xlabel('查询时刻 / s');axes[1,0].set_ylabel('0.4s终点误差 / px');axes[1,0].legend()
    for section,marker,label in [('fixed','o','原9起点'),('chronological_extension','s','后续6起点')]:
        d=single['pure_side'][section]['arms'];a=d['legacy_exact']['records'];b=d['tracked']['records']
        pair=[(r,s) for r,s in zip(a,b) if r['valid'] and s['valid']]
        axes[1,1].plot([r['query_t'] for r,s in pair],[r['error_px']-s['error_px'] for r,s in pair],marker+'-',label=label)
    axes[1,1].axhline(0,c='gray',linestyle=':');axes[1,1].set_title('单观测器相对旧入口：正值改善，负值退化');axes[1,1].set_xlabel('查询时刻 / s');axes[1,1].set_ylabel('误差减少 / px');axes[1,1].legend()
    fig.savefig(BASE/'source_force_results.png',dpi=150);plt.close(fig)
    def fmt(x):return f'{x:.3f}'
    table=''
    for clip,section,name in [('pure_side','fixed','纯桨叶侧视：原9起点'),('pure_side','chronological_extension','同一录像后续6起点'),('pure_train2','fixed','纯桨叶另一录像7起点')]:
        a=single[clip][section]['arms'];table+=f"<tr><th>{name}</th><td>{fmt(a['legacy_exact']['error']['mean'])}</td><td>{fmt(a['zero_matched']['error']['mean'])}</td><td>{fmt(a['tracked']['error']['mean'])}</td></tr>"
    variant=''
    for clip,name in [('clear','NN辅助清晰'),('side','NN辅助侧视'),('train2','NN辅助另一录像'),('pure_side','纯桨叶侧视9点'),('pure_side_extension','纯桨叶后续6点'),('pure_train2','纯桨叶另一录像')]:
        a=fixed[clip]['arms'];variant+=f"<tr><th>{name}</th><td>{fmt(a['zero_algebra']['error']['mean'])}</td><td>{fmt(a['zero_innovation']['error']['mean'])}</td><td>{fmt(a['tracked_innovation']['error']['mean'])}</td></tr>"
    whole_old=np.mean([r['error_px'] for section in single['pure_side'].values() for r in section['arms']['legacy_exact']['records'] if r['valid']])
    whole_new=np.mean([r['error_px'] for section in single['pure_side'].values() for r in section['arms']['tracked']['records'] if r['valid']])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阶段168：连续桨叶输入与二力接线</title>
<style>body{{font:17px/1.8 system-ui,"Microsoft YaHei";max-width:1120px;margin:32px auto;padding:0 20px;color:#24354a;background:#fafbfd}}h1,h2{{line-height:1.4}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{border:1px solid #d4dfeb;padding:8px;text-align:right}}th:first-child{{text-align:left}}pre{{background:#e8f1fa;padding:18px;white-space:pre-wrap}}.note{{padding:18px;background:#fff1cf;border-left:5px solid #b18225}}a{{color:#225f97}}</style>
<h1>连续局部桨叶输入已接正式纯桨叶入口，收益仍很小</h1>
<p>2026-11-05 · 阶段168 · 本轮优化自己的桨叶／可见性／二力模块，没有改NN、没有融合旧NN完整预测。正式候选不需要NN。</p>
<div class="note">单观测器同输入原9起点35.635→35.528px，但后续6点35.530→35.666px；15点合并{whole_old:.3f}→{whole_new:.3f}px，几乎持平。原生缓存正式入口9点35.527px，同观测数值置零35.593px，直接数值作用约0.065px。小收益保留，不当作稳定创新成功或30px达标。</div>
<h2>改的是自己的哪部分</h2><ol>
<li><b>从每帧选主频，改成局部区域续接。</b>复用当前已经检验的局部网格和竞争频率，用已有匈牙利关联连接机体相对位置。允许不同区域拥有不同频率，只在至少2个连续当前区域上计算各自频率变化。区域不是电机，网格数不当桨数。</li>
<li><b>避免换主导区域假造转速变化。</b>对持续区域的频率增量取中位，再形成归一化变化输入；新区域和消失区域不直接贡献阶跃。只选当前存在的已验证候选，不除2伪造缺失基频；倍频歧义仍明确保留。断档超过.12s或来源全换则转新参考。</li>
<li><b>四／三／部分二桨路径保留。</b>只有无法恢复完整结构的侧线/混合输入替换为上述局部续接。几何补点不补RPM，隐藏桨力始终归未知项。</li>
<li><b>接入同一个动力学观测器。</b>保留原RelativeRotorForceObserver的位置/位移观测和未知力状态，新的同源输入先改变当前已知贡献；位置创新再更新未知项。可见集合变化只转移分账，避免每帧直接用d=a−k*u吞掉新桨叶变化。固定k=200，避免与放大尺度混淆。</li></ol>
<p>新增开关<code>--rotor-track-local-source</code>，要求<code>--rotor-local-phase</code>；原默认保留。输出另存实际selected_frequency_hz、来源及normalized_proxy标志，归一化代理不再冒充真实机械RPM。长时间漂移超出代理范围会重建参考，不永久失锁。</p>
<h2>输入连续性与预测精度分开报告</h2>
<p>在相同已存观测中，纯桨叶侧视当前合格输入81/97→94/97，另一录像21/67→36/67。NN辅助侧视86/132→111/132、另一段3/141→12/141、清晰83/120→100/120。这些是输入资格数量，不是无人机检测召回率、桨叶分割AP或真实转速准确率。</p>
<p><b>纠正此前回放解析：</b>旧纯桨叶辅助函数把全局中位数读数标成局部一阶模板；实际live bank选择局部时应读local_temporal_phase.frequency_hz。当前解析已按来源修复，因此不能再宣传原误记63/2到94/36的倍数。旧167输出保留，并注明其纯桨叶输入归属限制；正式旧观测器在本轮逐帧复原，偏差0px。</p>
<table><tr><th>单观测器：0.4s源时刻误差</th><th>旧入口</th><th>同资格数值置零</th><th>连续桨叶输入</th></tr>{table}</table>
<p>侧视新输入相对同资格置零控制：原9点约0.097px改善，后续6点约0.119px改善。但相对旧完整入口后续仍退化，说明输入数值虽有小作用，整体可靠性尚未提高到目标水平。另一录像仍约99.8px，原位置参考/跟踪问题没有被周期输入自动解决。所有样本均为历史开发录像，非独立盲测。</p>
<img src="source_force_results.png" alt="输入资格、相对周期代理及逐点误差对比">
<h2>正式入口运行与速度</h2>
<p>完整运行原生EventCD NPY包→GPU桨叶支持→几何/匹配→连续周期→原单观测器，60个原生窗口、60次当前区域、56次有效预测/力输入、无NN。源窗口来自已有缓存，实际时间戳保留；与旧冷启动包边界约差20us，不能伪装为完全相同的RAW切窗。</p>
<p>这次9个固定终点平均35.527px；同一批当前观测的零输入回放35.593px，原预测复原偏差0px。控制是冻结观测回放，未冒充第二次完整RAW测速。</p>
<ul><li>缓存稳态吞吐29.93FPS（排除首包），P95单包54.29ms，不能称稳定30FPS。</li><li>含首包的处理段60包/3.80s≈15.79FPS；worker/模型启动另约22.66s，总调用26.63s。</li><li>CPU模型容量39.25FPS是算子统计，不能当端到端帧率。</li><li>E盘RAW调用在形成可观察输出前无进展，已终止，原因未定位；本轮没有RAW解码吞吐或真实30Hz发布时间证明。</li></ul>
<h2>额外未知力滤波的正例和负例都保留</h2>
<p>首轮另试保留未知力均值/协方差，用带噪声的当前净加速度创新更新，避免直接配平。它在纯桨叶原9点改善较多，但在后续时段和另一段NN观测上退化；额外滤已滤过的加速度会引入滞后，不能直接全面部署。</p>
<table><tr><th>同输入实验</th><th>原净加速度</th><th>额外未知力滤波／桨叶置零</th><th>额外未知力滤波／连续桨叶</th></tr>{variant}</table>
<p>纯桨叶原9点35.635→34.705px中，大部分来自未知项平滑；同资格零输入34.785px，实际桨叶数值约0.080px。后续却35.530→38.829px，因此这条候选保留研究，不把它升为默认或只报最好的9点。</p>
<h2>当前还卡在哪里</h2><ol>
<li><b>连续性已改善，真正的力学映射还弱。</b>当前实验固定默认像素尺度200，没有宣称量出了牛顿推力。来源稳定让后续辨识有更长数据，但局部混合可能含谐波/不同桨的切换，仍需后续速度证据确认。</li>
<li><b>隐藏力可能快变。</b>把隐藏桨归未知项是正确分账，但不能强假设整个未知项慢常。新输入响应要在同一个观测器中与位置创新共存，不能再用额外平滑掩盖迟滞。</li>
<li><b>另一录像主要仍是位置参考问题。</b>约99.8px不能靠当前约0.1px的周期作用解决；需要在自己的桨叶支撑与五圆/部分结构上稳定整体参考，而非升级NN来替代。</li>
<li><b>下一步应验证有效响应的后续信用。</b>用已到期自身速度，检验新周期响应能否解释之后的速度变化；可见性/来源分开记参数，不能只凭过去拟合就持续放大k。零输入与同资格对照继续保留。</li></ol>
<h2>机制检查与复现</h2><p>不同固定转速区域不会因主导切换造变化；合成20%同源阶跃可保留当前输入作用；只有倍频而无基频时不会除2补造；缺当前证据不称当前推力；可见集合增减不假加速；未知力每帧可更新；归一化参考越界可恢复。真实输入前缀重放一致。旧单观测器两录像复原偏差0px。正式缓存入口启动前快照和结束守卫通过，之后参考越界恢复修复在该60包上逐帧复原跟踪输出。</p>
<p>方法参考作者<a href="https://github.com/ros-event-camera/frequency_cam">FrequencyCam</a>、<a href="https://arxiv.org/abs/2603.09235">HelixTrack</a>和<a href="https://github.com/ialzugaray/haste">HASTE</a>对频率/几何/假设续接的处理；这里复用仓库已有组件自写关联与力学接线，没有声称完整复现或首次发明这些通用方法。</p>
<p><a href="single_observer_relativefix/scores.json">单观测器逐点结果</a> · <a href="native_selection_fixed/scores.json">首轮全部分支与负例</a> · <a href="native_cache_tracked/scores.json">正式缓存入口精度</a> · <a href="native_cache_tracked/summary.json">完整时钟</a> · <a href="native_trace_current_code_check.json">当前跟踪代码对60包复原</a> · <a href="../../event_drone_forecast/docs/SOURCE_FORCE168_PROTOCOL_20261105.md">协议</a></p>
<pre>当前实验选项（默认仍关闭）：
run_rotor_only.py ... --rotor-local-phase --rotor-track-local-source
run_fast_structure.py --rotor-only ... --rotor-local-phase --rotor-track-local-source
纯桨叶原生GPU仍使用既有gpu-python、rotor-gpu-front/native-gpu等选项。
本轮没有训练、改权重、上传GitHub或更新发布ZIP，两处yolov10与RAW保持。</pre></html>'''
    (BASE/'index.html').write_text(page,'utf-8');print('Saved report:',BASE/'index.html')
