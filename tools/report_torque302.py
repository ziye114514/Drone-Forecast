"""Measured close-out for differential-thrust/visibility torque research."""
from pathlib import Path
import sys,json,datetime,html
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path.insert(0,str(ROOT))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle


def read(path):return json.loads((WORK/path).read_text('utf8'))


if __name__=='__main__':
    OUT=WORK/'output/torque302_20261109';OUT.mkdir(exist_ok=False)
    result=read('output/factory300_20261109/results.json');joint=read('output/innovation299_20261109/results.json')
    final=read('output/torque301_20261109/results.json');partition=read('output/partition297_20261109/results.json')
    calibration=read('output/torque296_20261109/results.json');polarity=read('output/polarity295_20261109/results.json')
    temporal={n:read(f'output/temporal298_20261109/recording{n}/results.json') for n in ('15','16')}
    rows=[json.loads(s) for s in (WORK/'output/motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl').read_text('utf8').splitlines()]
    r=min(rows,key=lambda r:abs(r['t']-26.499735));center=np.array(r['structure']['current_geometric_center'])
    fig,axes=plt.subplots(1,2,figsize=(12,5));ax=axes[0];weights=[];freqs=[]
    n=np.array(r['structure']['axis_xy']);n/=np.linalg.norm(n)
    for i,(g,rate) in enumerate(zip(r['geometry']['rotors'],r['rates'])):
        p=np.array([g['cx'],g['cy']])-center;f=rate['frequency_hz'];w=(f/300)**2/4;weights.append(w);freqs.append(f)
        ax.plot([0,p[0]],[0,p[1]],color='#78889d',lw=1)
        ax.add_patch(Ellipse(p,g['major'],g['minor'],angle=g['angle'],facecolor='#cddcff',edgecolor='#345f9c'))
        ax.annotate('',p+n*w*100,p,arrowprops=dict(arrowstyle='->',color='#267a41',lw=2))
        ax.text(p[0],p[1]+18,f'{f:.1f} Hz',ha='center',fontsize=9)
    ax.add_patch(Circle((0,0),8,color='#64799b'));ax.annotate('',n*sum(weights)*100,(0,0),arrowprops=dict(arrowstyle='->',color='#142b4c',lw=3))
    ax.set_aspect('equal');ax.set_xlim(-110,110);ax.set_ylim(70,-125);ax.grid(alpha=.2)
    ax.set_title(f'Actual clear frame: {r["t"]:.6f}s\nUnequal parallel contributions; net direction is common normal')
    ax.set_xlabel('x relative to five-circle centre [px]');ax.set_ylabel('image y [px]')
    axes[1].bar(range(1,5),np.array(weights)/np.mean(weights),color=['#3566a8','#82a2cd','#4e7db8','#a9bfdc'])
    axes[1].axhline(1,color='gray',linestyle='--');axes[1].set_xticks(range(1,5));axes[1].set_xlabel('Anonymous current rotor regions')
    axes[1].set_ylabel('f_i squared / mean(f_i squared)');axes[1].set_title('Relative force PROXIES are unequal\nSame coefficient prior; not measured Newton thrust')
    fig.tight_layout();fig.savefig(OUT/'unequal_thrust.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2));names=['13','15','16'];xx=np.arange(3)
    for label,data,offset,color in [('Native RAW option',result['cases'],-.18,'#547da6'),('CPU joint-phase input',joint['cases'],.18,'#d29647')]:
        differences=[]
        for n in names:
            v=data[n];base=v['control']['error']['mean'];new=v['torque' if 'torque' in v else 'innovation']['error']['mean'];differences.append(base-new)
        bars=axes[0].bar(xx+offset,differences,.34,label=label,color=color)
        axes[0].bar_label(bars,labels=[f'{d:+.4f}' for d in differences],fontsize=8,padding=4)
    axes[0].set_xticks(xx,names);axes[0].axhline(0,color='gray',lw=1);axes[0].set_ylim(-.025,.14)
    axes[0].set_ylabel('Baseline mean FDE minus new mean FDE [px]');axes[0].set_xlabel('Previously seen recordings');axes[0].legend(fontsize=8)
    axes[0].set_title('Incremental contribution is very small\nWeak visible-outline references, not independent COM truth')
    axes[1].bar(xx,[result['source_counts'][n]['informative_drive_frames'] for n in names],color='#547da6')
    axes[1].set_xticks(xx,names);axes[1].set_ylabel('Packets with measured same-source drive change');axes[1].set_title('Native-input bottleneck: only one usable change\n181 source cutoffs per recording')
    fig.tight_layout();fig.savefig(OUT/'increment_and_input.png',dpi=150);plt.close(fig)
    table=[]
    for n in names:
        base=result['cases'][n]['control']['error'];native=result['cases'][n]['torque']['error'];cpu=joint['cases'][n]['innovation']['error'];combined=final['cases'][n]['tangent_plus_native']['score']['error']
        table.append(f'<tr><td>{n}</td><td>{base["n"]}/25</td><td>{base["mean"]:.6f}</td><td>{native["mean"]:.6f}</td><td>{cpu["mean"]:.6f}</td><td>{combined["mean"]:.6f}</td></tr>')
    phase_table=[]
    for n in ('15','16'):
        old=partition['cases'][n];new=temporal[n]
        before=34 if n=='15' else 39;after=sum(new['ring_histogram'][str(k)] for k in (2,3,4))
        phase_table.append(f'<tr><td>{n}</td><td>{before}→{after}</td><td>{old["longest_current_block"]}→{new["longest_current_torque_block"]}</td><td>{old["best_gain_zscore"]:.3f}→{new["best_gain_zscore"]:.3f}</td><td>{new["CPU_ms_mean"]:.0f}ms/包</td></tr>')
    body='''<!doctype html><meta charset="utf-8"><title>不等桨推力、力矩与预测：本轮结果</title>
<style>body{font:17px/1.75 system-ui,Microsoft YaHei,sans-serif;max-width:1120px;margin:38px auto;padding:0 22px;color:#223047;background:#f7f9fc}h1,h2{color:#183854}section{background:white;padding:20px 26px;margin:22px 0;border-radius:12px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;border-bottom:1px solid #dce4ef;text-align:right}td:first-child,th:first-child{text-align:left}code{background:#eaf0f6;padding:2px 5px}a{color:#195eac}.note{border-left:4px solid #d2933c;padding-left:15px}pre{white-space:pre-wrap;background:#edf2f8;padding:14px;font-size:14px}</style>
<h1>不等桨推力→力矩→未来姿态：已补链路，精度收益仍弱</h1>
<p>2026-11-09 · 研究阶段294–302 · RAW原生ON/OFF、五圆/侧视与可测＋不可测力主线，无NN。0.4秒30px与持续30Hz目标均未达成。</p>
<section><h2>1. 这次实际改了什么</h2><p>原程序已有逐桨f²求和，当前缺的是空间分布如何影响未来姿态。本轮加入匿名逐桨力矩代理、可见性变化时的未知力矩交接、带宽不确定力学初值的角响应观察器，并把差动贡献接入未来法线积分。机头朝向不参与；当前平移位置/速度/加速度不改。</p>
<pre>每个真实当前来源：Tᵢ代理 = cᵢ fᵢ²（cᵢ默认相同先验，实际Tᵢ随fᵢ不同）
合推力：F = (ΣTᵢ)n；力矩：τ = Σ rᵢ × (Tᵢn)
力矩改变角运动 → n(t+h)改变 → 未来平移力方向改变
消失或新出现的来源：交接基线至未知角响应；不制造差动力矩跳变
隐藏桨推力仍属于未知合力；隐藏力矩属于未知角响应，二者单位不同</pre>
<img src="unequal_thrust.png"><p class="note">图中频率来自原清晰片段实际逐区域观测。同样的桨盘法线下，四条推力大小不同，瞬时合力仍沿法线。f是事件周期代理；叶片数、谐波、CT与惯量没有独立标定，不能把图中代理称为真实RPM或牛顿推力。</p>
<p>对弱透视、局部近似各向同性的转动惯量，投影法线角加速度代理与 −ΣTᵢrᵢ 有关。初始角响应由现有图像力增益K、桨半径R与机架尺度L得到 γ≈4KR/L²；这是圆饼惯量先验，不是测出的无人机参数。随后只用已经到达的姿态观测校正。另设完全相同、仅去掉桨叶差动输入的角观察器，取两者差值，避免把额外姿态平滑算成桨叶贡献。非桨盘法向的外扰项不随法线旋转；包含隐藏桨的估计法向响应仍沿原有分工随法线，隐藏力矩另归角响应。</p></section>
<section><h2>2. 同一起点实测：还没有达到有效的大幅改善</h2><p>统一10.8–15.6秒、间隔0.2秒、0.4秒FDE；三段都是已经看过的开发数据。未改变评分起点和参考。以下为像素均值。</p>
<table><tr><th>录像</th><th>可评分/固定查询</th><th>原基线</th><th>原生轻量差动力矩</th><th>旧CPU联合相位＋差动力矩</th><th>原切向记忆289＋原生力矩</th></tr>''' + ''.join(table)+'''</table>
<img src="increment_and_input.png"><p class="note">原生15号仅改善约0.005px；CPU联合相位15号约0.110px，其余两段不足0.01px且13轻退。弱事件轮廓参考不能证实如此小的改进具有统计或实际意义。组合列较大的收益主要来自上轮289，不能归功于本轮力矩。没有挑每段最佳版本替换统一方法。</p>
<p>公共工厂实际GPU共跑三段各181包，开关两臂共1086次模型更新。默认预测与冻结基线最大差2.51e−12px；两臂当前p/v/a与旧状态最大差2.01e−11。说明差异来自未来积分，旧基线没有被暗改。此次为保存原生事件的回放，不是独立新录像，也不是新的实时RAW吞吐认证。</p></section>
<section><h2>3. 现在具体卡在哪里</h2><p><b>最直接的卡点：差动输入没有持续出现。</b> 原生入口三段181包中，分别仅0、16、3包有当前逐桨周期；真正可连续比较同一来源变化的只有0、1、0包。大量整机池化频率不能拆成四个频率，也不能冒充逐桨力矩。这解释了轻量接入后为什么几乎不移动预测。</p>
<p><b>未知力矩与响应系数仍未分开。</b> 本轮先消去每段的初始角度、角速度和未知恒定角加速度，再尝试学习差动响应；三段加旧清晰片段均未可靠辨识。即使清晰四桨短段可看见不等频率，若频差基本不变，它仍能被未知平衡力矩抵消，不能据此学习惯量/推力系数。没有强行设隐藏力矩为零来造出“已学会”。</p>
<p><b>可见集合变化会打断学习，已修复了一部分。</b> 以可测变化＋未知项的交接替代整段重建后，15号连续窗口4→20帧；16号9→12帧。缺桨和重新出现均不产生瞬时力矩脉冲；持续可见来源的真实变化保留。</p>
<p><b>进一步用过去频率初始化当前拟合，改善了输入但没有兑现精度。</b> 每个空间来源只增加一个历史频率假设，选择只看当前早半窗，晚半窗仍按原0.5集中度/一致性、9扇区等门槛独立验证；不直接延用历史频率充当前测量。</p>
<table><tr><th>录像</th><th>≥2当前环包数</th><th>交接后的最长学习帧数</th><th>响应增益信噪代理</th><th>CPU研究耗时</th></tr>'''+''.join(phase_table)+'''</table>
<p>信噪代理未过预设2倍噪声门槛，而且噪声本身未独立标定，不能当正式置信度。更慢的新相位入口接同一力矩模型后，15/16为48.458、44.848px，未优于旧联合相位输入，故不推荐拿它拖慢默认入口。分开ON/OFF相位的诊断也未增加多桨覆盖，未采用。</p></section>
<section><h2>4. 已能怎样调用，下一步只攻什么</h2><p>公共<code>run_rotor_candidate.py</code>新增显式<code>--rotor-torque-innovation</code>，默认关闭。它复用当前原生逐桨周期，不启动CPU作者相位DLL、不增加NN、额外椭圆搜索或原始事件副本。需要原有GPU Python与本地几何DLL参数。可与上轮<code>--rotor-tangential-memory</code>组合；组合在固定缓存回放中核验，尚未做组合的独立真实RAW帧率认证。</p>
<p>下一单项应是<b>持续的、空间归属明确的逐桨周期变化</b>：对已经确认的转轴维护连续相位状态，当前观测修正、遮挡增加不确定性，重新可见时独立复锁。保留源集合交接，禁止把过期/池化频率作为新当前转速。验收先看同源有效差动变化能否覆盖变向前，而后只做一次冻结同起点预测对照；没有提前量就如实停在未证实，不继续调大γ追分。整体框中心/FPS/新NN均不作为这条研究的替代成果。</p>
<p>若周期始终无法得到可信的逐桨对应和共同谐波口径，RAW单目数据不能凭位置误差唯一分离四个CT、质量、惯量与外力矩。本项目可以学习有效响应，但当前尚未证明差动推力能稳定改善新录像预测，不能写成已完成物理标定或普适创新成功。</p></section>
<section><h2>5. 复用来源、代码与证据</h2><p><a href="https://arxiv.org/html/2604.18289v1">Relative State Estimation using Event-Based Propeller Sensing</a>已利用逐桨转速平方差推进姿态，并以椭圆观测修正；本项目的“差动力矩→姿态”物理链条不作为首创。<a href="https://arxiv.org/html/2603.09235v1">HelixTrack论文</a>与<a href="https://github.com/radimspetlik/helixtrack">作者项目</a>用于环几何/相位观测启发与本地作者CPU核心；不是现成轨迹预测器。<a href="https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf">Faessler等旋翼动力学模型</a>提供刚体力/力矩背景。</p>
<p>新逻辑为本项目实现；未复制/改动作者头文件、SDK安装、原RAW/权重、两处yolov10或已发布NN包。隐藏推力始终归未知合力，角运动另有未知力矩；没有新安装、训练或上传。</p>
<p><a href="../factory300_20261109/results.json">实际GPU公共工厂与基线</a> · <a href="../torque301_20261109/results.json">最终/组合/零输入报告修正</a> · <a href="../innovation299_20261109/results.json">旧联合相位输入的增量</a> · <a href="../torque296_20261109/results.json">可辨识性与清晰四桨对照</a> · <a href="../partition297_20261109/results.json">可见集合交接</a> · <a href="../temporal298_20261109/recording15/results.json">新相位15</a> · <a href="../temporal298_20261109/recording16/results.json">新相位16</a>。</p></section>'''
    (OUT/'index.html').write_text(body,'utf8')
    summary=dict(local_date='2026-11-09',last_completed_stage=302,report=str(OUT/'index.html'),
        goal30px30Hz_achieved=False,default_promoted=False,NN_used=False,new_default_force_model=False,
        native_actual_factory_results={n:{a:v['error'] for a,v in result['cases'][n].items()} for n in names},
        joint_input_results={n:{a:v['error'] for a,v in joint['cases'][n].items() if a in ('control','innovation')} for n in names},
        current_native_source_counts=result['source_counts'],
        prior_uniform_disk_inertia_not_measured=True,known_input_not_total_force=True,
        independent_RPM_pose_COM_truth_available=False,all_jobs_finished=True)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf8')
    print(str(OUT/'index.html'),flush=True)
