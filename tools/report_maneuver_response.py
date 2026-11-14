"""Local report from frozen evidence; no new forecasting or model selection."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from diagnose_maneuver_horizon import OUT,SPECS


def load(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from compare_final_pipeline import reference_at
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':155})
    scored=load(OUT/'scores.json');deadline=load(OUT/'scores_deadline.json');warm=load(OUT/'warm_context.json')
    frozen=load(OUT/'frozen_side_axis_rebase/summary.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,(name,t,dimension,label) in zip(axes,[('side',117.632157,0,'当前已有制动线索：可修复'),('other',43.432899,1,'后半窗明显向下转弯')]):
        spec=next(r for r in SPECS if r[0]==name)
        labels=sorted(load(ROOT.parent/'output'/spec[2]/'labels.json')['records'].values(),key=lambda r:r['t'])
        selected=[r for r in labels if t-.35<=r['t']<=t+.55 and r.get('box')]
        ax.plot([r['t'] for r in selected],[(r['box'][dimension]+r['box'][dimension+2])/2 for r in selected],'k.-',label='人工框中心（离线参考）')
        for tag,color,title in [('control','#c78526','原计算'),('stable_side_rebase','#187b62','历史坐标修正')]:
            source=rows(OUT/f'{name}_{tag}.jsonl');r=min(source,key=lambda r:abs(r['t']-t))
            ts=[r['t']]+[r['t']+h for h in (.1,.2,.4)]
            values=[r['coefficient']['position'][dimension]]+[r['physical'][str(h)][dimension] for h in (.1,.2,.4)]
            ax.plot(ts,values,'o--',ms=4,color=color,label=title)
        ax.axvline(t,color='gray',lw=1);ax.axvline(t+.2,color='gray',lw=.7,ls=':')
        ax.set_title(label);ax.set_xlabel('源时刻 / s');ax.set_ylabel(('水平x' if dimension==0 else '向下y')+' / px');ax.legend(fontsize=8)
    fig.savefig(OUT/'two_error_types.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    sets=[scored['side']['comparison'],scored['side']['actual_RAW']['same_new_observations'],deadline['side']['actual_RAW']['same_new_observations']]
    before=[r['old_matched_error']['mean'] for r in sets];after=[r['new_matched_error']['mean'] for r in sets];x=np.arange(3)
    axes[0].bar(x-.18,before,.36,label='同观测旧计算');axes[0].bar(x+.18,after,.36,label='历史坐标修正')
    axes[0].axhline(30,color='gray',ls='--');axes[0].set_xticks(x,['原冻结观测\n26起点','新RAW观测\n25起点','截止等待RAW\n25起点'])
    axes[0].set_ylabel('0.4秒均值 / px');axes[0].legend(fontsize=9)
    for xx,b in zip(x,after):axes[0].text(xx+.18,b+1,f'{b:.2f}',ha='center')
    for tag,color,label in [('control','#c78526','原计算'),('candidate','#187b62','历史坐标修正')]:
        r=scored['side']['revisions'][tag]['records']
        axes[1].plot([r['t'] for r in r],[r['revision_px'] for r in r],color=color,label=label,lw=1)
    axes[1].set_xlabel('源时刻 / s');axes[1].set_ylabel('相邻时刻对同一未来目标的修订 / px');axes[1].legend(fontsize=9)
    axes[1].set_title('修订包含合理的新信息，不全是抖动')
    fig.savefig(OUT/'improvement.png');plt.close(fig)
    actualrows=[]
    for key,folder,label,prefix in [('plain','RAW_side_axis_rebase8s','侧视，修正',114.),('deadline','RAW_side_deadline_rebase8s','侧视，修正+截止等待',114.),
        ('other','RAW_other_axis_rebase6s','另一段，修正',None),('otherdeadline','RAW_other_deadline_rebase6s','另一段，修正+截止等待',None)]:
        s=load(OUT/folder/'summary.json');timing=rows(OUT/folder/'timings.jsonl')[1:]
        chosen=[r for r in timing if prefix is None or r['t']<prefix]
        capacity=1000/np.mean([r['total_service_ms'] for r in chosen])
        actualrows.append(f"<tr><td>{label}</td><td>{'固定前4秒' if prefix else '完整6秒'}</td><td>{capacity:.2f}</td><td>{s['paced_fresh_observation_rate']:.2f}</td><td>{s['timing_ms']['completion_lag_ms']['p95']:.1f}</td><td>{s['nn_frames']}</td></tr>")
    wa=warm['original_same_origins'];wb=warm['control_on_warm_observations']
    rawother=scored['other']['actual_RAW']['same_new_observations'];deadother=deadline['other']['actual_RAW']['same_new_observations']
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>独立侧视分支、未来变向与可修复误差</title>
<style>body{{font:17px/1.75 'Microsoft YaHei',sans-serif;color:#182b40;background:#f1f5fa;max-width:1160px;padding:0 22px;margin:35px auto}}section{{background:white;border-radius:12px;padding:22px 28px;margin:20px 0}}h1{{font-size:30px}}h2{{font-size:23px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border-bottom:1px solid #dce4ee;padding:9px;text-align:left}}img{{max-width:100%}}a{{color:#175a9a}}code{{background:#edf2f7;padding:2px 5px}}.note{{background:#fff0cc;padding:12px}}</style>
<h1>独立侧视输入、未来变向与可修复误差</h1><p>2026-10-29 · 工作候选已更新，原发布ZIP保持冻结。目标仍为0.4秒均值&lt;30px与稳定低延迟30FPS，尚未全面达到。</p>
<section><h2>先回答侧视分支</h2><p>侧视不必先恢复四个桨心或建立五圆，也不必给桨编号。当前整机轴线约束图像推力方向，来自桨叶的周期作为相对有效推力线索，未知合力由随后位置反馈修正。新修正明确允许没有四桨模板的轴线分支。</p><p>需要区分“提供线索”和“唯一得到四桨总推力”：单个主峰可能代表某一桨，显著性/可见比例变化会使主峰换人；因此整体周期仍需在线增益、口径连续性和不确定性。逐桨归属只在确实做逐桨反演时需要，不再作为侧视准入门槛。当前没有把周期、事件数或补出的桨心称作真实RPM/牛顿推力。</p></section>
<section><h2>0.4秒误差：两种情况确实都存在</h2><p>另一录像43.432899秒，之后前/后两个0.2秒平均速度约(-318,-48)与(-196,+181)px/s，后半窗向下转弯。这说明存在预测发出后的运动变化；但不能仅凭位置标注，证明之前的姿态或原始事件里绝无提前线索。</p><p>侧视117.632157秒，起点前已拟合出约1000px/s²的制动斜率，原算法却因位置一致性失败而不用它。原因是NN框重锁时出现小的尺寸/中心参考跳变；旧代码没把跳变当速度，却还把跳变前后的中心混在加速度检查中。这是可修复的线索丢失，不能归咎于未来不可预测。</p><img src="two_error_types.png"><p class="note">过去框的二次拟合外推方差很大，不能据此给“不可预测误差占比”。半窗速度变化也可能是可预测的持续加速度。所有未来标签均在预测冻结后用于诊断，未进入运行预测器。</p></section>
<section><h2>已经落实的算法改善</h2><p>新增位置一致性历史的参考坐标修正：当前强NN发生小的形状重锁，已有刚体位移可靠、整机侧轴稳定时，将旧一致性坐标统一平移到当前参考。当前框、共享位置状态、原始流速度、周期输入和预测起点保持原样；没有增设传统并行预测器。沿用6px位置尺度、12px最大修正、9px²位移方差与既有侧轴稳定门槛，没有按预测误差搜索参数。</p><table><tr><th>同一批原观测</th><th>起点</th><th>原0.4秒均值</th><th>修正后</th><th>P95</th></tr><tr><td>自采侧视</td><td>26/26</td><td>34.60px</td><td>31.62px</td><td>72.54→60.86px</td></tr><tr><td>另一录像</td><td>15/15</td><td>59.79px</td><td>59.79px</td><td>不变</td></tr><tr><td>FRED 7原观测</td><td>原样保留</td><td colspan="3">所有预测逐项不变</td></tr></table><p>主要收益集中在117.63秒一个制动起点：104.21→26.64px；它有明确机制，但不是跨场景泛化证明。相邻时刻对同一未来目标的预测修订P95从50.24降到37.81px；修订也包含合理新信息，不能全部称作抖动。</p><p>已接入<code>--target-candidate</code>，<code>--no-position-history-rebase</code>可关闭。机制检查包含已知700px/s²运动、静止形变重锁、无需四桨的侧轴、原位置/轨迹观测不变。小幅内部纹理运动可能低于现有位置噪声，未宣称所有桨叶内部运动都能被区分。</p><img src="improvement.png"></section>
<section><h2>完整RAW成绩与同输入成绩分别记录</h2><table><tr><th>实际新RAW</th><th>原起点保留</th><th>同次新观测旧计算</th><th>修正后</th></tr><tr><td>侧视，修正</td><td>25/26</td><td>38.40px</td><td>36.05px</td></tr><tr><td>侧视，修正+截止等待</td><td>25/26</td><td>34.42px</td><td>31.57px</td></tr><tr><td>另一段，修正</td><td>15/15</td><td>{rawother['old_matched_error']['mean']:.2f}px</td><td>{rawother['new_matched_error']['mean']:.2f}px</td></tr><tr><td>另一段，修正+截止等待</td><td>15/15</td><td>{deadother['old_matched_error']['mean']:.2f}px</td><td>{deadother['new_matched_error']['mean']:.2f}px</td></tr></table><p>新观测中的旧/新计算逐项复用相同检测、几何和周期，候选回放复原RAW误差为0。独立RAW重跑受NN答复应用时序等影响，输入观测不同，因此旧另一段59.79px与这次65–66px不能直接归因于历史坐标修正。侧视两次都少了原来的一个起点，不能把31.57px直接当完整26起点结果或宣布稳定达标。</p></section>
<section><h2>速度、异步稳定性与前置历史</h2><p>额外显式选项<code>--deadline-nn</code>利用当前33ms截止余量，等待已经提交的NN答复，同时预留最近下游计算，上限半周期。等待计入耗时，未新增网络或预测分支；现有请求调度随观测改变，实测侧视NN次数72→85，不能说调用次数不变。超时请求≤16.67ms，操作系统实际调度可能使等待更久，实测最大18.61ms。</p><table><tr><th>本机短测</th><th>窗口</th><th>处理能力FPS</th><th>整段当前新观测Hz</th><th>完成滞后P95 ms</th><th>主NN调用</th></tr>{''.join(actualrows)}</table><p>整8秒存在无人机缺席，52.41/43.44的整段吞吐不作检测FPS。截止等待默认关闭：侧视一次变好，但另一段没有改善且仍有数百毫秒滞后。30FPS处理能力、有效观测率与准时完成是不同要求。</p><p>固定增加108–110秒过去历史，仍评价原26起点：保留{wa['new_available']}/26，均值{wa['new_matched_error']['mean']:.2f}px；同一warm观测旧计算{wb['new_matched_error']['mean']:.2f}px。110秒前只有{warm['observed_body_before110']}个合格整机输出、{warm['gain_updates_before110']}次增益更新，不能称作充分暖机。遇到6个超过200万事件共享槽的包（最大2193664），既有有界队列回退保留全部事件，没有截断；但复制与处理大包带来约{warm['summary']['timing_ms']['completion_lag_ms']['p95']:.0f}ms的P95滞后，不作为实时改进。</p></section>
<section><h2>现在还卡在哪里</h2><p>已确定并修复一处“当前制动线索被丢掉”的问题，原26起点均值还差约1.62px到30px；另一段仍较差。当前没有足够独立姿态/转速真值，把每个大误差唯一归成新控制或估计不稳。接下来沿独立侧视分支改已有线索的及时使用和连续性，并检查异步NN时序、大包传输造成的延迟；不靠补四桨身份、删掉难起点或缩短0.4秒来达标。</p><p>前一交接把117秒称作“出画”过于笼统：人工框在117.3–118.2秒仍完整位于画内，最左约34.5px，实际接近边缘并制动/转向，不能把这一段全视作出画后的不可观测。</p></section>
<section><h2>记录与来源</h2><p><a href="scores.json">原与新RAW评分</a> · <a href="scores_deadline.json">截止等待RAW评分</a> · <a href="frozen_side_axis_rebase/summary.json">固定输入与原轨迹复原</a> · <a href="mechanism_checks.json">机制检查</a> · <a href="diagnosis.json">过去位置参考与局限</a> · <a href="warm_context.json">前置历史结果</a> · <a href="source_snapshot/manifest.json">源码清单</a></p><p>相关已有研究：<a href="https://github.com/noahcao/OC_SORT">OC-SORT官方项目</a>讨论观测与运动估计，<a href="https://stonesoup.readthedocs.io/en/latest/stonesoup.models.transition.html">Stone Soup官方运动模型</a>包含Singer模型。本批复用本项目现有几何、位移与反馈代码，没有复制上述项目代码，未宣称该修正为首创。</p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True);manifest=snap/'manifest.json'
    names=['run_fast_structure.py','forecast_runtime.py','position_history_gauge.py','shared_image_motion.py',
        'tools/probe_same_window_motion.py','tools/probe_maneuver_rebase.py','tools/diagnose_maneuver_horizon.py',
        'tools/score_maneuver_batch.py','tools/check_position_history_gauge.py','tools/report_maneuver_response.py','docs/SIDE_RESPONSE.md']
    if manifest.exists():
        for name,digest in load(manifest).items():assert hashlib.sha256((snap/name).read_bytes()).hexdigest()==digest,name
    else:
        records={}
        for name in names:
            raw=(ROOT/name).read_bytes();p=snap/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);records[name]=hashlib.sha256(raw).hexdigest()
        manifest.write_text(json.dumps(records,indent=2),'utf-8')
    print(str(OUT/'index.html'))


if __name__=='__main__':main()
