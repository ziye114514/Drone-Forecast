"""Read completed pilots, draw timing and forecast diagnostics, save report."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from audit_target25 import read,continuity
OUT=ROOT.parent/'output/target25_continuity_20261031'


def load(p):return json.loads(p.read_text('utf-8'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':155})
    fig,axes=plt.subplots(2,2,figsize=(12.4,7),layout='constrained');publish={};runtime={}
    for i,(name,start) in enumerate([('clear',26.),('other',40.)]):
        run=OUT/('RAW_clear_publish4s' if name=='clear' else 'RAW_other_publish6s')
        s=load(run/'summary.json');pub=[json.loads(x) for x in (run/'publications.jsonl').read_text().splitlines()]
        times=np.array([r['t']-start for r in pub]);intervals=np.diff(times)*1000
        axes[i,0].plot(times[1:],intervals,color='#1a7b82',lw=1)
        axes[i,0].axhline(1000/30,color='gray',ls='--');axes[i,0].set_ylabel('实际输出间隔 / ms')
        axes[i,0].set_title(('清晰段' if name=='clear' else '另一段')+'：输出时钟约30Hz')
        ages=np.array([np.nan if r['source_age_s'] is None else 1000*r['source_age_s'] for r in pub])
        valid=np.array([r['valid'] for r in pub]);axes[i,1].plot(times,ages,color='#49647c',lw=1)
        axes[i,1].scatter(times[~valid],ages[~valid],color='#c74e4e',s=13,label='明确无效')
        axes[i,1].axhline(160,color='gray',ls='--');axes[i,1].set_ylabel('使用的源状态年龄 / ms')
        axes[i,1].set_title('有输出不等于有当前新图像');axes[i,1].legend(fontsize=9)
        for ax in axes[i]:ax.set_xlabel('源时钟经过 / s')
        publish[name]=s['publication'];runtime[name]=dict(summary=s,continuity=continuity(read(run)))
    fig.savefig(OUT/'publication_timing.png');plt.close(fig)
    other=load(OUT/'other_center_scores.json');fred=load(OUT/'fred0_paired_scores.json')
    exact=load(OUT/'period_overlap_exactness.json');retime=load(OUT/'other_publication_scores.json')
    sparse=load(OUT/'sparse_clear_diagnostic.json');negative=load(OUT/'net_force_trend_scores.json')
    rows=[]
    for label,data in [('另一段，同15起点',other),('FRED0，同12起点',fred)]:
        pair=data['matched'];rows.append(f"<tr><td>{label}</td><td>{pair['old_matched_error']['mean']:.2f}px</td><td>{pair['new_matched_error']['mean']:.2f}px</td><td>{pair['new_available']}/{pair['original_origins']}</td></tr>")
    summary=dict(user_targets=dict(horizon_s=.4,non_side_mean_px=25.,continuous_hz=30.,side_tolerance='moderately relaxed, no new hard value assigned'),
        actual_publications=publish,runtime=runtime,center_other=other['matched'],center_fred0=fred['matched'],
        exact_period_overlap=exact,rollout_exactness=load(OUT/'rollout_exactness.json'),actual_output_time_score=retime,
        sparse_clear_diagnostic=sparse,negative_net_force_trend=negative,
        all_target_achieved=False,front_view_benchmark_available=False,
        current_source_observations_increased_by_publisher=False)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    pubrows=[]
    for name,title in [('clear','清晰段4秒'),('other','另一段6秒')]:
        p=publish[name];pubrows.append(f"<tr><td>{title}</td><td>{p['outputs']}</td><td>{p['actual_mean_hz']:.3f}</td><td>{p['valid_outputs']}</td><td>{p['missed_deadlines']}</td><td>{p['interval_p95_ms']:.2f}</td></tr>")
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>30Hz输出、中心运动与25px新目标</title>
<style>body{{font:17px/1.8 'Microsoft YaHei',sans-serif;color:#193047;background:#f2f5fa;max-width:1120px;margin:32px auto;padding:0 20px}}section{{background:white;border-radius:12px;margin:20px 0;padding:20px 26px}}h1{{font-size:30px}}h2{{font-size:23px}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{text-align:left;border-bottom:1px solid #dce5ef;padding:9px}}img{{max-width:100%}}a{{color:#155e97}}code,pre{{background:#edf2f7}}pre{{padding:14px;white-space:pre-wrap}}.note{{background:#fff0cc;padding:12px}}</style>
<h1>持续30Hz与非侧视25px：本轮实际进展</h1><p>2026-10-31 · 输入仍为RAW。沿五圆／侧轴、周期有效推力和未知合力继续推进。目标没有全面完成。</p>
<section><h2>新目标与视角口径</h2><p>优先非侧视0.4秒位置误差低于25px，侧视允许适度放宽；均值、P95、覆盖与延迟分别记录。未擅自给侧视指定新硬阈值。原来约59px的“另一段”多数是算法axis_only／insufficient，不能直接称正视；算法模式也不等于人工视角真值。清晰窗口含四桨、部分／侧轴转换，只有稀疏人工框，尚无足够连续的正视评测证明。</p></section>
<section><h2>已落实：中心平移与旋转分开</h2><p>当前图像已拟合出刚体相似变换，原框仍按特征点中位位移移动；特征集中在一侧桨盘时，旋转／尺度变化会拖动框。新增复用变换在机体或近期五圆中心求位移，且中心必须处于当前内点凸包内、≥12内点、内点率≥75%、位移方差≤9px²。不合格回原中位位移，无额外LK或网络。</p><table><tr><th>相邻实际RAW对照</th><th>原</th><th>新</th><th>原起点保留</th></tr>{''.join(rows)}</table><p>另一段得到约0.71px小收益，保留；FRED0约0.008px回退，变化很小，未作足以判定显著性的独立样本评估，未宣称普遍增益。不同RAW异步重跑的观测会变，不能把全部差异唯一归因于新平移。稀疏清晰人工框只作诊断：五个目标跨度0.384–0.409秒，不冒充严格0.4秒基准；其中一个four源时刻约10.52px，仅1个案例。</p></section>
<section><h2>已复用：周期进程与几何重叠</h2><p>复用项目已有PeriodProcess。清晰120帧的当前框、结构、周期和预测数值逐项完全一致，处理能力32.64→34.66FPS，整段滞后P95 123.94→84.19ms。平均吞吐虽够30，仍有单帧超过33ms与积压。不能用平均FPS代替持续准时。</p></section>
<section><h2>已接通：独立30Hz实际时间输出</h2><p>复用TimedPublisher。新增解析积分器沿同一Singer总加速度、法线演化及已学力记忆，将当前状态推进到实际输出时间，再给其后0.1／0.2／0.4秒位置。540条原记录在原三个时距最大差1.14e-13px；没有插值终点或新增传统预测器。源状态／最近真实图像超过160ms就明确无效。</p><table><tr><th>实测</th><th>输出</th><th>实际Hz</th><th>有效预测</th><th>漏输出时隙</th><th>间隔P95 ms</th></tr>{''.join(pubrows)}</table><p class="note">30Hz接口已经跑通，持续有效新观测30Hz尚未全面做到。无效心跳不计预测成功，旧状态推进不计新检测。较轻片段确有97个连续新观测与有效状态、跨度3.20秒，每个都在自身33ms周期内算完；清晰重负载段仍有明显迟到。清晰实测发布中最大有效源年龄136ms，并非当前无延迟图像。</p><img src="publication_timing.png"></section>
<section><h2>时间正确不代表精度已经提高</h2><p>严格按实际输出时间+0.4秒，在预先固定的16个目标中13个有效。同状态同目标，直接沿用旧源时刻0.4秒终点误差72.00px，正确重新积分79.85px，中位数66.21→62.50px、P95 158.29→180.20px。净力模型对更远未来的外推仍不好。不能为降低均值，把终点改回早几十毫秒；也不能拿这里13起点与前面15起点约60px直接比较。</p><p>尝试复用已有合格流曲率的净力变化率、固定.35秒相关时间的临界阻尼延续，没有扫参数。另一段60.159→61.274px，侧视原26起点不变；保留负结果，未接运行入口。当前力变化率不能直接代表未来控制。</p></section>
<section><h2>可直接运行的工作配置</h2><pre>python run_fast_structure.py --raw "event.raw" --output "新的输出目录" --start 26 --duration 4 --sdk-avx2 --structure-guided-nn --realtime30</pre><p><code>--realtime30</code>组合物理候选、严格中心平移、独立周期进程和30Hz定时输出。原<code>--target-candidate</code>仍保留为控制；可单独启用<code>--rigid-box-center</code>、<code>--process-period</code>或<code>--publish-hz 30</code>。原模型和发布ZIP没有修改。</p><p><code>predictions.jsonl</code>保留每次源更新，<code>publications.jsonl</code>保留实际输出及source_t／source_age_s／target_times／valid。两份文件用途不同。</p></section>
<section><h2>下一步卡点</h2><p>先压清晰段几何／周期耗时长尾，同时保持当前弧与周期输入。精度主要卡在变向期缺少可靠先行物理输入、当前加速度不能稳定代表未来0.4秒，以及连续非侧视参考不足。继续优先非侧视25px，侧视保留现有修复；不把未知视角移进侧视组来降低考核，也不把一个10px案例称作整体达标。</p></section>
<section><h2>复查资料与来源</h2><p><a href="summary.json">汇总</a> · <a href="other_center_scores.json">另一段对照</a> · <a href="fred0_paired_scores.json">FRED相邻对照</a> · <a href="period_overlap_exactness.json">周期进程等值</a> · <a href="rollout_exactness.json">同一模型解析积分</a> · <a href="other_publication_scores.json">实际输出时间评分</a> · <a href="sparse_clear_diagnostic.json">稀疏清晰诊断与时距</a> · <a href="net_force_trend_scores.json">变化率负结果</a> · <a href="source_snapshot/manifest.json">当前源码</a></p><p>复用<a href="https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html">OpenCV相似变换</a>、<a href="https://docs.opencv.org/4.x/dc/d6b/group__video__track.html">LK</a>及既有<a href="https://stonesoup.readthedocs.io/en/latest/stonesoup.models.transition.html">Singer运动模型</a>，未复制第三方源码，也未将通用模型称为首创。</p></section></html>'''
    (OUT/'index.html').write_text(page,'utf-8');print(str(OUT/'index.html'))


if __name__=='__main__':main()
