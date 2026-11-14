"""Report matched rotor-only accuracy trials, with explicit scope limits."""
from pathlib import Path
import json,html,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output';OUT=BASE/'rotor_accuracy243_20261108'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);joint={};periods={};completions={}
    for n in ['13','15','16']:
        p=BASE/f'joint244_20261108/recording{n}/results.json'
        if not p.exists():continue
        joint[n]=json.loads(p.read_text());folder=BASE/f'joint242_20261108/recording{n}_10_16.01'
        periods[n]=[json.loads(s) for s in (folder/'periods.jsonl').read_text().splitlines()]
        completions[n]=json.loads((folder/'completion.json').read_text())
    main=json.loads((BASE/'mainline239_20261108/results.json').read_text())['results']
    trust=json.loads((BASE/'trust241_20261108/results.json').read_text())['results']
    speed=json.loads((BASE/'projection237_20261108/checks.json').read_text())
    published=json.loads((OUT/'publication_scores.json').read_text())
    live={n:json.loads((BASE/folder/'summary.json').read_text()) for n,folder in [('rolling','available_gpu246_20261108/raw13'),('iterator','available_gpu248_20261108/raw13_iterator'),('cooperative','available_gpu252_20261108/raw13_cooperative')]}
    gain=json.loads((OUT/'gain_diagnosis.json').read_text())
    result=dict(joint=joint,phase=completions,mainline=main,trust=trust,speed=speed,
        live=live,publication_scores=published,gain_diagnosis=gain,
        NN_used=False,metric_thrust_or_RPM_truth_available=False,weak_references=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),'utf-8')
    fig,axes=plt.subplots(len(joint),1,figsize=(10,3.1*max(1,len(joint))),squeeze=False)
    for ax,(n,r) in zip(axes[:,0],joint.items()):
        for arm,label in [('control','原候选'),('joint_overlay','新相位替换'),('joint_geometry','相位＋法线')]:
            rows=[s for s in r['scores'][arm]['records'] if s['valid']]
            ax.plot([s['t'] for s in rows],[s['error_px'] for s in rows],'-o',ms=3,label=label)
        ax.axhline(30,color='#b85e4d',ls='--',lw=1);ax.set(title=f'录像{n}：相同位置/运动输入、固定0.4s源起点',ylabel='FDE / px',xlabel='源时间 / s');ax.legend(ncol=3);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'matched_errors.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(len(periods),1,figsize=(10,2.4*max(1,len(periods))),squeeze=False)
    for ax,(n,rows) in zip(axes[:,0],periods.items()):
        ax.step([r['t'] for r in rows],[r['meta'].get('late_qualified_before_temporal_confirmation',0) for r in rows],where='post',label='当前后半窗通过')
        ax.step([r['t'] for r in rows],[r['meta'].get('qualified_current_rings',0) for r in rows],where='post',label='连续确认后可入已观测项')
        ax.set(title=f'录像{n}：候选≠已测力，失锁立即撤销当前分量',ylabel='环数（非GT）',xlabel='源时间 / s',ylim=(-.1,4.2));ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'phase_continuity.png',dpi=145);plt.close(fig)
    table=''
    for n,r in joint.items():
        values=[r['scores'][k] for k in ['control','joint_overlay','joint_geometry']]
        table+=f'<tr><td>{n}</td>'+''.join(f'<td>{v["error"]["mean"]:.3f}</td>' for v in values)+f'<td>{values[0]["available"]}/25</td><td>{r["joint_input_frames"]}/181</td><td>{r["joint_normal_frames"]}</td></tr>'
    mt=''
    for n,r in main.items():
        v=r['scores'];mt+=f'<tr><td>{n}</td>'+''.join(f'<td>{v[k]["error"]["mean"]:.3f}</td>' for k in ['control','no_measured_period','no_plane_axis'])+'</tr>'
    lt=''
    for n,v in live.items():
        lt+=f'<tr><td>{n}</td><td>{v["frames"]}</td><td>{v["steady_pipeline_fps"]:.2f}</td><td>{v["actual_valid_output_rate_hz"]:.2f}</td><td>{v["actual_output_age_ms_median"]:.1f}</td><td>{v["skipped_observations"]}</td></tr>'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶主线：连续相位与预测精度</title><style>
body{{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f2f5f9;color:#193247;max-width:1080px;margin:36px auto;padding:0 22px}}section{{background:white;border-radius:12px;padding:24px;margin:22px 0}}h1{{font-size:30px}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:10px;border-bottom:1px solid #d5dfe7;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;height:auto}}code{{background:#edf2f6;padding:2px 5px}}.flow{{padding:18px;background:#eaf5f2;border-left:4px solid #258570}}</style>
<h1>桨叶主线：连续相位与预测精度</h1><p>2026-11-08 · 236–254 · 保留旧算法和NN发布版；本页为研究线。</p>
<section><h2>仍按用户的检测逻辑推进</h2><div class="flow">RAW原生ON/OFF → 桨叶事件抑制/弧与椭圆 → 五圆模板与四/三/两桨或侧视资格 → 当前可测贡献 u ＋ 未知力 d → 位置预测 → 已到达时刻的位置偏差修正状态。</div>
<p>四桨用当前环；三桨可用近期非退化模板推断隐藏位置，但隐藏频率不算已测；两桨/近侧视只提供当前轴/法线和合格的局部周期。头尾无关。质量、尺度、隐藏桨、重力、风阻等无法独立分离的部分仍由未知项承担。现有频率只提供图像域相对输入，不能称牛顿推力或已知机械RPM。</p></section>
<section><h2>这次真正新增的链路</h2><p>使用过去的原生桨盘几何和平移作初值；只在当前早半窗联合拟合几何和相位，冻结后用后半窗验证。连续确认后才进入当前可测项，失锁只保留几何用于复锁。重复环不重复求和；任意短末包也按来源时间选较早初值。</p>
<p>作者CPU核心来自HelixTrack本机私有构建，不是完整论文复现；未改作者头文件，未把DLL放进公开发布包。模型推断圆锥不是独立测量，未把其中心冒充机体质心。此分支目前仅为精度研究：CPU平均每窗约0.38–1.54秒，未接入实时默认。当前周期要连续确认，当前平面方向可由两条不重复的后半窗确认圆锥提供，两项资格分开。</p>
<table><tr><th>录像</th><th>原候选</th><th>新相位替换</th><th>再修正法线</th><th>评分起点</th><th>当前输入窗</th><th>新法线窗</th></tr>{table}</table><p>单位px；0.4s源时间FDE均值。新的相位在缺失时回退原桨叶信号；输入归属改变由已有未知力转移处理，避免制造假加速度。</p><img src="matched_errors.png"><img src="phase_continuity.png"></section>
<section><h2>主线哪个部分当前确实有用</h2><p>固定相同位置和运动观测，仅撤去某一物理特征：不是更换检测器，也未使用NN。</p><table><tr><th>录像</th><th>完整主线</th><th>去掉当前周期输入</th><th>去掉平面方向约束</th></tr>{mt}</table><p>方向几何在15/16有明确作用，当前周期幅值增量仍弱；11也显示几何并非总有收益。不能以此声称五圆方法已普适胜出。</p></section>
<section><h2>真正修复的运行时问题</h2><p>249短片段调用栈捕捉到一次2.948秒读包，其中SDK在空HAL队列重复调用_run/get_latest_raw_data/decode约658060次。官方安装头文件明确poll_buffer=0表示无数据。新增工作区内CooperativeRawReader扩展，仅在空队列sleep0.5ms让出CPU，再沿用原SDK解码/结束与flush；未改SDK安装文件。</p><p>原SDK两读法、扩展读法均检查181包、11714141个事件，x/y/p/t和切分时刻逐包哈希相同。252完整RAW的读包最长14.79ms，未再出现约3秒长尾。候选入口默认改为cooperative，原rolling入口保留可作对照。</p><table><tr><th>本次6秒读法</th><th>处理帧</th><th>流程FPS</th><th>有效发布Hz</th><th>年龄中位ms</th><th>跳包数</th></tr>{lt}</table><p>相同RAW/起点/积压上限，独立短运行，启动与文件缓存有差异，不把全部比值归为单一因素或称稳定验收。新联合相位不在这条实时入口内；GPU已获用户使用许可，未记录整个电脑其他程序逐秒负载。</p><p>固定25个发布时间查询：iterator可评分6/25，平均55.21px；cooperative为15/25，平均63.35px。它们不是相同评分集合。共同5起点为60.75→61.18px，不能声称平均预测精度改善。这次实时收益是长尾消除和有效输出连续性，30Hz未达到。</p></section>
<section><h2>保留的优化与未采用的尝试</h2><p>频谱投影改成等价的空间/时间因子相乘，不丢任何事件和时间戳。三个真实事件样本局部计算快3.43–5.82倍，复数投影最大差5.43e-15。这是部件速度，不能当整条RAW流程FPS。</p><p>另测试了依靠已成熟位置误差给平面约束分配信用：15退化约0.99px、13退化约0.43px，16仅改善约0.12px；因此未更改默认预测。保留负结果，避免把单段小幅改善称为整体优化。</p></section>
<section><h2>现在卡在何处</h2><p>新输入覆盖达到13/15/16的12/86/83窗，但三个序列的已观测项增益学习更新仍都是0次；15同源有效输入历史最多6个，16最多9个仍缺少合格激励。未知项本来还允许快速法向响应，频率与未知响应容易互相解释。这是幅值可辨识性与连续性的缺口，不能靠降低8窗门槛或把事件密度当牛顿推力解决。</p><p>优先保留当前几何方向约束，进一步寻找同一可见贡献持续变化对净运动的因果增量，显式检查增益可辨识性；联合几何新法线在16改善约1.52px、在15略退且尾部退，不按录像切换。实时路径尚有几何/前端长尾，30Hz不能只看模型瞬时容量34.56FPS。不要再无目的增加单窗通过数量。</p></section>
<section><h2>结论边界</h2><p>参考来自固定原生ON/OFF轮廓并经过目视核验，不是独立质心/RPM真值；三个序列均已参与开发。源时间误差不包含实时计算延迟。30px与持续30Hz目标仍未验收。用户已明确GPU空闲，之前共享负载下的测速仍不能反推当前独占性能。</p><p><a href="results.json">完整逐起点评分</a> · <a href="../../event_drone_forecast/docs/HANDOFF_20261108_PRECISION255.md">交接文档</a></p></section></html>'''
    (OUT/'index.html').write_text(text,'utf-8');print(str(OUT/'index.html'))
