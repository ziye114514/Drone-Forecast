"""Deliver CPU observation results separately from trajectory/FPS results."""
from pathlib import Path
import json,numpy as np,html
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output';OUT=BASE/'cpu_resume235_20261108'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    chirp=json.loads((BASE/'chirp231_20261108/real.json').read_text('utf-8'))['records']
    pose=json.loads((BASE/'pose234_20261108/results.json').read_text('utf-8'))['records']
    c0=sum(r['result']['constant_rate_control']['qualified'] for r in chirp)
    c1=sum(r['result']['qualified'] for r in chirp)
    p0=sum(r['result']['control']['qualified'] for r in pose)
    p1=sum(r['result'].get('arms',{}).get('phase_only',{}).get('qualified',False) for r in pose)
    p2=sum(r['result'].get('arms',{}).get('joint_pose',{}).get('qualified',False) for r in pose)
    gained=[r for r in pose if not r['result']['control']['qualified'] and r['result'].get('arms',{}).get('joint_pose',{}).get('qualified')]
    lost=[r for r in pose if r['result']['control']['qualified'] and not r['result'].get('arms',{}).get('joint_pose',{}).get('qualified')]
    fig,ax=plt.subplots(figsize=(7.4,5.2));plotted=0
    for r in pose:
        b=r['result']['control'];a=r['result'].get('arms',{}).get('joint_pose',{})
        if 'validation_concentration' not in b or 'concentration' not in a.get('validation',{}):continue
        new=bool(a.get('qualified') and not b['qualified']);kept=bool(a.get('qualified') and b['qualified'])
        ax.scatter(b['validation_concentration'],a['validation']['concentration'],s=85 if new else 32,
            color='#df7d26' if new else '#148e78' if kept else '#8799ae',edgecolor='white',linewidth=.6)
        if new:ax.annotate(f"{r['recording']} / {r['t']:.1f}s",(b['validation_concentration'],a['validation']['concentration']),xytext=(6,6),textcoords='offset points',fontsize=9)
        plotted+=1
    ax.plot([0,1],[0,1],color='#9eabb8',ls='--',lw=1);ax.axhline(.5,color='#d06754',ls=':');ax.axvline(.5,color='#d06754',ls=':')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='固定几何：后半窗相位集中度',ylabel='联合几何：后半窗相位集中度',
        title=f'{plotted} 个可评分候选；橙色为新增通过（并非独立真实电机数）')
    ax.grid(alpha=.15);fig.tight_layout();fig.savefig(OUT/'phase_scatter.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.3));x=np.arange(len(gained));width=.24
    values=[[],[],[]]
    for r in gained:
        values[0].append(r['result']['control']['validation_concentration'])
        values[1].append(r['result']['arms']['phase_only']['validation'].get('concentration',0))
        values[2].append(r['result']['arms']['joint_pose']['validation']['concentration'])
    for i,(name,color) in enumerate([('固定几何/转速','#8799ae'),('作者仅相位更新','#659ac6'),('作者几何+相位更新','#148e78')]):
        ax.bar(x+(i-1)*width,values[i],width,label=name,color=color)
    ax.set_xticks(x,[f"录像{r['recording']} / {r['t']:.1f}s" for r in gained]);ax.axhline(.5,color='#d06754',ls='--',label='集中度门槛（还需相位/方向/覆盖门槛）')
    ax.set(ylim=(0,1),ylabel='冻结模型在后半窗的相位集中度',title='新增候选：只用早半窗拟合，验证事件不更新模型')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'new_candidates.png',dpi=145);plt.close(fig)
    metrics=[]
    for n in ['13','15','16']:
        score=json.loads((BASE/f'fov227_20261108/recording{n}/score.json').read_text('utf-8'))
        metrics.append(dict(recording=n,source_horizon_s=.4,mean_px=score['error']['mean'],available=score['available'],version=227))
    for n in ['14','11']:
        source=json.loads((BASE/'credit202_20261108/scores.json').read_text('utf-8')) if n=='14' else json.loads((BASE/'regional212_20261108/scores11.json').read_text('utf-8'))
        score=source[n] if n=='14' else source['adaptive'];metrics.append(dict(recording=n,source_horizon_s=.4,mean_px=score['error']['mean'],available=score['available'],version=202))
    summary=dict(trajectory_metrics_unchanged_this_CPU_batch=metrics,trajectory_gain_proven=False,controlled_FPS_test=False,GPU_used=False,
        chirp=dict(hypotheses=len(chirp),constant_qualified=c0,accelerating_qualified=c1),
        joint_pose=dict(hypotheses=len(pose),constant_qualified=p0,author_phase_only=p1,author_joint_pose=p2,
            gained_vs_constant=len(gained),lost_vs_constant=len(lost),counts_are_not_unique_motors=True),
        force_input_enabled=False,independent_RPM_truth_available=False)
    (OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    table=''.join(f'<tr><td>{r["recording"]}</td><td>{r["mean_px"]:.2f}</td><td>{r["available"]}</td><td>{r["version"]}</td></tr>' for r in metrics)
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CPU继续优化：几何与相位联合确认</title>
<style>body{{font:16px/1.75 'Microsoft YaHei',sans-serif;background:#f0f4f8;color:#21334b;margin:0}}main{{max-width:1050px;margin:auto;padding:28px}}section{{background:white;padding:24px 30px;border-radius:12px;margin:20px 0}}h1,h2{{line-height:1.4}}.note{{padding:16px;background:#fff1d6}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dbe4ed}}a{{color:#176eb4}}code{{background:#edf1f5;padding:3px}}</style><main>
<h1>CPU继续优化：先修旋转观测的几何误差</h1><p>2026-11-08 · 未启动GPU · 未修改原始RAW、NN发布包或旧基线</p>
<section><h2>当前预测结果没有被新试验改写</h2><table><tr><th>录像</th><th>0.4s源时间平均误差 / px</th><th>评分点数</th><th>已测版本</th></tr>{table}</table>
<p class="note">这些是既有固定起点回放结果，本轮CPU观测试验尚未降低轨迹FDE。多数参考为固定8/4原生事件轮廓中心，不是独立人工飞行真值；不同已测版本已分别列出。30px未稳定达到，GPU仍由用户使用，30FPS没有空闲条件下的验收结论。</p></section>
<section><h2>固定转速确有局限，但仅加入加速度没有真实收益</h2><p>新实现只在早半窗拟合连续转速变化，再冻结角阶数、频率、加速度和方向验证后半窗，保留原0.5集中度/方向与9扇区门槛。6组二/三叶恒速、加速、减速合成信号通过，8组非周期环拒绝；打乱后半窗不改变拟合参数。</p>
<p>真实录像全部{len(chirp)}个已保存候选，固定转速通过{c0}个、加速度模型通过{c1}个：没有新增通过，丢失一个原候选。因此没有接入主预测，也没有降低阈值制造提升。不能把合成成功当作实际推力或轨迹增益。</p></section>
<section><h2>复用作者代码后：联合几何找回两个候选</h2><p>在本地重新编译已有HelixTrack作者核心，显式关闭GPU路径；修复旧适配器硬编码已不存在的DLL路径，支持早半窗相位与方向初始化。作者头文件不改，二进制不放进公开发布包。方法参考<a href="https://arxiv.org/html/2603.09235v1">HelixTrack原文</a>。</p>
<p>同样{len(pose)}个冻结候选，固定方法{p0}个通过、作者仅相位更新{p1}个、作者联合几何更新{p2}个；对固定方法新增{len(gained)}个、丢失{len(lost)}个。两个新增来自录像12和11。固定形状误差会影响旋转确认，这次有直接观测证据。</p><img src="new_candidates.png"><img src="phase_scatter.png">
<p class="note">41是“位置/运动假设”，不是41个独立电机；同一物理桨盘可能有多个假设。初始几何、方向和频率只来自早半窗，作者滤波器不读取后半窗，最后用冻结状态评价；所有环还需有限椭圆、FOV及原门槛检查。没有独立RPM真值，没有证明牛顿推力、检测AP或0.4s轨迹误差改善。录像11新增集中度约0.513，离门槛很近，不能称稳定锁定。</p></section>
<section><h2>新增了可复用的观测模块，未强行反馈推力</h2><p><code>joint_ring_phase.observe_ring</code>接收原生事件、早半窗桨盘几何和运动初始化、本机作者DLL，输出独立验证结果与模型推断的圆锥/转轴投影。接口检查几何和速度时间，禁止用验证半窗的信息初始化；与已保存观测逐字段一致。返回<code>force_input_enabled=False</code>，现有主线保持原样。</p>
<p>下一步应检查跨帧连续锁定、信号是否仍对应同一物理桨、重复候选去重和失锁后的未知力转移，再做同起点“启用/禁用”轨迹对照。只有那时才能说桨叶观测收益转化成了预测收益。</p></section>
<section><h2>文献约束与复现</h2><p>10月22日预印本<a href="https://arxiv.org/abs/2610.31384">CoHelix</a>已经联合估计四桨共享几何与各自旋转状态；它要求初始桨盘标注、方向和初始转速。可借鉴共享约束，但不能把它直接当成RAW自主冷启动实现，本轮也未声称复现它。</p>
<p>原始数据：<a href="../chirp231_20261108/real.json">加速度对照</a>、<a href="../pose234_20261108/results.json">全部联合几何候选</a>、<a href="../joint235_20261108/checks.json">接口/时间检查</a>。<a href="../precision_delivery_20261108/index.html">上一轮完整轨迹报告</a>；<a href="../../event_drone_forecast/docs/HANDOFF_20261108_CPU235.md">本轮交接</a>。</p></section></main></html>'''
    (OUT/'index.html').write_text(doc,'utf-8');print(OUT/'index.html')
