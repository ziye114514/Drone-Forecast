"""Post-inference report; all predictions and test protocols are frozen."""
from pathlib import Path
import sys,json,hashlib,numpy as np,cv2,html
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from score_rotor164 import score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=ROOT.parent/'output';OUT=BASE/'rootfix191_20261107'

def read(path):return json.loads(path.read_text('utf-8'))
def rows(path):return [json.loads(s) for s in path.read_text('utf-8').splitlines()]
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),'utf-8')

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);results={};details={}
    for case,old,new,label,origins in [
        ('side',BASE/'layout176_20261107/integrated/side_fixed.jsonl',BASE/'normal187_20261107/side/normal_legacy_input.jsonl',BASE/'dense_labels_side_110_1195/labels.json',np.arange(110.4,114.0001,.2)),
        ('train2',BASE/'layout176_20261107/integrated/train2_fixed.jsonl',BASE/'normal187_20261107/train2/normal_legacy_input.jsonl',BASE/'dense_labels_train2_40_46/labels.json',np.arange(43.2,44.4001,.2)),
        ('recording15',BASE/'phase185_20261107/recording15/control176.jsonl',BASE/'normal187_20261107/recording15_with_input/normal_legacy_input.jsonl',BASE/'motion181_20261107/reference_review15/references.json',np.round(np.arange(10.8,15.6001,.2),1)),
        ('recording16',BASE/'unseen191_20261107/recording16/control176.jsonl',BASE/'unseen191_20261107/recording16/plane.jsonl',BASE/'unseen191_20261107/reference16/references.json',np.round(np.arange(10.8,15.6001,.2),1))]:
        paired={arm:score(path,label,origins) for arm,path in [('baseline',old),('plane',new)]}
        assert [r['t'] for r in paired['baseline']['records'] if r['valid']]==[r['t'] for r in paired['plane']['records'] if r['valid']]
        for s in paired.values():
            s['reference_is_independent_human_GT']=case in ('side','train2')
            s['reference_is_calibrated_COM']=False
            s['manual_boxes_are_development_references']=case in ('side','train2')
            s['new_temporal_extension_is_not_a_new_recording']=False if case.startswith('recording') else True
        details[case]=paired
        results[case]={arm:{'planned':s['eligible'],'scored':s['available'],**s['error']} for arm,s in paired.items()}
        results[case]['mean_improvement_percent']=100*(1-paired['plane']['error']['mean']/paired['baseline']['error']['mean'])
    dump(OUT/'paired_scores.json',details);dump(OUT/'summary.json',results)
    print(json.dumps(results,indent=2),flush=True)
    fig,axes=plt.subplots(2,2,figsize=(13,7.6))
    for ax,case in zip(axes,('recording15','recording16')):
        pair=details[case];aa={r['t']:r for r in pair['baseline']['records'] if r['valid']};bb={r['t']:r for r in pair['plane']['records'] if r['valid']}
        times=sorted(aa);old=np.array([aa[t]['error_px'] for t in times]);new=np.array([bb[t]['error_px'] for t in times])
        ax[0].plot(times,old,'o-',color='#c07d22',label='176 baseline',markersize=3);ax[0].plot(times,new,'o-',color='#117b68',label='Plane-conditioned unknown force',markersize=3)
        ax[0].axhline(30,color='#b22222',linestyle=':',label='30px target');ax[0].set_title(case+(' (development)' if case.endswith('15') else ' (frozen unseen recording)'));ax[0].set_ylabel('0.4s error (px)');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
        ax[1].bar(times,old-new,width=.12,color=['#117b68' if x>=0 else '#c84f4f' for x in old-new]);ax[1].axhline(0,color='black',linewidth=.7);ax[1].set_title('Every shared origin: positive = improvement');ax[1].set_ylabel('Baseline minus new error (px)');ax[1].grid(alpha=.2)
        for a in ax:a.set_xlabel('Source time (s)')
    fig.tight_layout();fig.savefig(OUT/'all_origins.png',dpi=145);plt.close(fig)
    # Two strongest gains AND two strongest regressions, with the full curve
    # above. This is a diagnostic selection, never the scoring population.
    pair=details['recording16'];aa={r['t']:r for r in pair['baseline']['records'] if r['valid']};bb={r['t']:r for r in pair['plane']['records'] if r['valid']}
    gains=sorted(aa,key=lambda t:aa[t]['error_px']-bb[t]['error_px']);selected=list(dict.fromkeys(gains[:2]+gains[-2:]))
    refs=read(BASE/'unseen191_20261107/reference16/references.json')['records'];selection=[]
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    for ax,t in zip(axes.flat,selected):
        r=aa[t];s=bb[t];im=cv2.cvtColor(cv2.imread(str(BASE/f'unseen191_20261107/reference16/{t:.1f}.png')),cv2.COLOR_BGR2RGB)
        ax.imshow(im);b=refs[f'{t:.1f}']['box'];center=(np.asarray(b[:2])+b[2:])/2
        p=np.array(r['prediction']);q=np.array(s['prediction']);target=np.array(r['target'])
        ax.plot(*center,'+',color='#228b22',markersize=13,label='Current outline centre')
        ax.plot(*p,'o',fillstyle='none',color='#d28b1f',markersize=11,label='Baseline forecast')
        ax.plot(*q,'s',fillstyle='none',color='#007a6b',markersize=10,label='New forecast')
        ax.plot(*target,'x',color='black',markersize=12,label='Future outline reference')
        points=np.vstack([b[:2],b[2:],p,q,target]);lo=points.min(0)-45;hi=points.max(0)+45
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal',adjustable='datalim');ax.set_title(f'{t:.1f}s -> {t+.4:.1f}s: {r["error_px"]:.1f} -> {s["error_px"]:.1f}px',fontsize=11)
        ax.grid(alpha=.13);selection.append(dict(t=t,baseline=r['error_px'],plane=s['error_px']))
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1),h_pad=3,w_pad=2);fig.savefig(OUT/'unseen16_forecasts.png',dpi=145);plt.close(fig);dump(OUT/'overlay_selection.json',dict(rule='Two best gains and two worst regressions; full shared-origin scoring unchanged',examples=selection))
    paths={15:BASE/'phase185_20261107/recording15/control176.jsonl',16:BASE/'unseen191_20261107/recording16/control176.jsonl',17:BASE/'unseen191_20261107/recording17/control176.jsonl',18:BASE/'normal187_20261107/fresh18/recording18_control176.jsonl',19:BASE/'motion181_20261107/holdout/recording19_control176.jsonl'}
    coverage={}
    for n,path in paths.items():
        rr=rows(path);coverage[n]=dict(packets=len(rr),detected_regions=sum(r['body'].get('detected',False) for r in rr),valid_forecasts=sum(r['valid'] for r in rr),current_position_observations=sum(r['motion'].get('current_observation',False) for r in rr),
            mean_events=float(np.mean([r['counts']['events'] for r in rr])),median_period_pixels=float(np.median([r['counts']['period_pixels'] for r in rr])),median_eligible_pixels=float(np.median([r['counts']['eligible_pixels'] for r in rr])))
    dump(OUT/'input_coverage.json',coverage)
    table=''.join(f'<tr><td>{case}</td><td>{r["baseline"]["scored"]}/{r["baseline"]["planned"]}</td><td>{r["baseline"]["mean"]:.2f} → <b>{r["plane"]["mean"]:.2f}</b></td><td>{r["mean_improvement_percent"]:.1f}%</td><td>{r["baseline"]["p95"]:.2f} → {r["plane"]["p95"]:.2f}</td></tr>' for case,r in results.items())
    covtable=''.join(f'<tr><td>{n}</td><td>{r["detected_regions"]}/181</td><td>{r["valid_forecasts"]}/181</td><td>{r["median_period_pixels"]:.0f}</td></tr>' for n,r in coverage.items())
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>根因修复与新录像测试 · 191</title><style>body{{font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#203036;background:#f4f6f6;margin:0}}main{{max-width:1080px;margin:auto;padding:35px 25px 70px}}h1{{font-size:32px}}h2{{margin-top:38px}}section,.card{{background:white;border-radius:10px;padding:20px 25px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{border-bottom:1px solid #dde4e2;text-align:left;padding:10px}}img{{width:100%;height:auto}}code,pre{{background:#eef2f1;overflow:auto}}pre{{padding:14px;white-space:pre-wrap}}.note{{border-left:5px solid #b47c25;padding:10px 18px;background:#fff7e8}}a{{color:#126f66}}li{{margin:8px 0}}.small{{font-size:14px;color:#58646a}}</style><main>
<h1>先修影响精度的根因，再测新录像</h1><p>2026-11-07 · 181–191 · 纯桨叶 ON/OFF 路线 · 没有使用 NN 或训练权重</p>
<div class="card"><b>实质改善：</b>新增开发录像15：61.46 → 52.87px；冻结代码之后的新录像16：51.67 → 48.01px。改的是未知力的方向结构，不是针对录像换阈值。<br><b>范围：</b>0.4s，同起点、同观测、同周期输入；新录像使用事件可见轮廓参考。30px目标和持续RAW30FPS仍未达成。</div>
<section><h2>1. 影响严重的根因，按处理优先级</h2><ol>
<li><b>未知力的延续丢失了机体法线约束——本轮已做出有效修复。</b>旧版让二维未知加速度自由变化；运动开始转向时，已经观测到桨盘平面变化，却没有把它用于未知桨力的方向。现在保留二力模型 a=k·u+d，在内部令 d=e+h·n：隐藏/未标定的法向响应随当前法线n变化，风、重力、阻力及其他残差e不跟着整机一起旋转。h从0开始，由过去位置残差学习；不是预设整机推力，不是已辨识真实质量/牛顿力。</li>
<li><b>周期检测冷启动和连续性——覆盖范围的最严重短板，尚未解决。</b>新录像17仅7包区域、3次位置观测，达不到原有5次观测发布条件；18/19零区域。缺的是前端可用测量，增加后端力学复杂度不能恢复它。运动区域只作提案，做原事件运动补偿再交给原周期检验，及复用现有弱区域聚合，在18均未恢复检测；完整失败保留，没有继续为它调阈值。</li>
<li><b>周期→推力幅值的可信连接不足。</b>局部环相位原型把15的当前合格输入从8/181提高到99/181，但相同预测器的误差变化只有约0.06px；放宽为量纲一致的在线增益学习后也没有收益。本轮改善主要来自几何法线约束，不能归功于精确转速或推力辨识。环相位实现还有约168ms均值的额外开销，不进入实时推荐路径。</li>
<li><b>观测参考、遮挡和未观测未来指令。</b>可见轮廓中心随姿态/出画变化，不等于真实COM。未知力与增益存在相互解释；当前信息不能保证0.4s后没有新指令。这些限制不等于可将现有过冲全部归因于不可预测未来。</li></ol></section>
<section><h2>2. 同源、同起点实际结果</h2><table><tr><th>录像</th><th>评分/计划起点</th><th>均值px：176 → 新模式</th><th>改善</th><th>P95 px</th></tr>{table}</table>
<p>side/train2是旧人工开发框，15已参与方案选择，因此属于开发数据；16在看画面、运行模型前冻结代码和10–16s窗口，所有参数不变。16共25个预设起点，22个有未来可见轮廓参考，另3个未来时刻目标已离开画面，两个方法评分集合完全相同。</p>
<div class="note">15/16参考来自原生20ms ON/OFF的固定计数阈值8/4和面积16规则，参考生成不读预测器输出；全部27帧覆盖图已逐页目视核验。它是可见轮廓中心代理，不是独立人工真值、真实COM、真实姿态/RPM。阈值4/8的中心差最大约7.5px，部分画面有桨叶截断或支架遗漏，均保留。新录像16的改善是跨录像初步证据，不是稳定泛化或统计显著性证明。</div>
<p>评分另修了一处浮点精确时间查询错误：恰好落在左侧锚点、但带微小浮点误差时被漏掉。旧side现在应含113.8s这个原本就计划的坏例：基线由历史17点38.62变为完整18点42.41。预测没有改变。新模式同18点42.07，不能把评分集合变化当算法提升。</p>
<img src="all_origins.png" alt="开发和独立录像的全部同起点误差与正负改善"><p class="small">上图包含每个可评分起点，没有只展示改善帧。左：0.4s误差；右：基线减新模式，红柱代表退化。</p>
<img src="unseen16_forecasts.png" alt="独立录像中两处最大改善和两处最大退化"><p class="small">固定规则选两处最大改善、两处最大退化用于诊断；不是评分样本筛选。橙圈旧预测、绿方框新预测、黑叉未来轮廓参考。当前图像只是展示背景，未来参考未被预测器读取。</p></section>
<section><h2>3. 新录像不能只报成功的一段</h2><table><tr><th>录像</th><th>区域输出包</th><th>有效预测包</th><th>逐包周期合格像素中位数</th></tr>{covtable}</table><p>这些是181包原生输入的输出数量，<b>不是检出率/AP/召回率</b>，因为并未逐包标注目标在场状态，离场包也在分母里。两个后端在16/17的有效性相同；18两后端均0输出；19为已有冻结的176基线0输出，未另测法线后端。17/18/19的黄昏失败完整保留。</p><p>采集元数据提示15/16白天，17–19黄昏；后者同步帧相机亮度记录低。但事件相机抗闪烁、限流、拖尾滤波均关闭，偏置配置相同。不能说是错误硬件滤波导致，也不能仅凭相关性认定低光是唯一原因。当前直接证据是：白天中位合格周期像素242/180，黄昏只有3/9/9，区域密度提案并未等价于可靠周期桨叶证据。</p></section>
<section><h2>4. 为什么选这个修复，而非不断换滤波参数</h2><p>旧视野相关的匹配归一化确有代码不一致，但统一归一化后实际预测更差；前后向匹配校验也没有带来均值收益。更平滑的力/推力变化率先验、局部相位单独接入、仅放开在线增益，同样没有足够收益。这些试验均留在原输出目录，未成为推荐默认。</p><p>有效修复来自同一结构性假设：<b>不可测的桨力仍应受当前机体平面约束。</b>去掉周期输入，录像15仍从61.46改善到53.36；把法线固定不随当前姿态变化，则回到61.37。说明改善与动态几何相关，尚不能证明周期数值有实质作用。h与外部残差仍不唯一，不称真实推力估计。</p><p>防护保留：法线30°异常跳变或过期先重建方向参考，不注入力冲击；运动未确认时不启用未来角速度外推；已有悬停/平面摆动检查最大0.4s伪位移4.67px。可见/隐藏贡献交接保持净加速度和外部项连续。</p></section>
<section><h2>5. 现在如何实际使用，下一步攻哪里</h2><p>新模式已经接入公共RAW入口：<code>--rotor-unknown-force-model plane</code>。旧控制保留为<code>free</code>；API旧默认不被静默覆盖。复现命令与解释见<a href="../../event_drone_forecast/docs/HANDOFF_20261107_ROOTFIX191.md">交接文档</a>。当前推荐研究对比用plane；NN发布包、RAW、权重和两处yolov10-main都没改。</p><p><b>下一优先项：</b>以空间邻域共同周期＋跨包相位连续性取代“每个像素短窗都必须发出足够多事件”的冷启动依赖，但必须用独立后半窗确认，并允许拒绝非周期运动区域。先让弱周期段连续产生真实位置观测，再评估误差；不能通过降低发布门槛、用惯性代替检出，或把隐藏桨频率写回可测项来刷覆盖。已有区域聚合和运动补偿的简单组合已经失败，需要针对事件统计和共同频率混叠作结构改进。</p><p>本轮没有新的完整RAW实时FPS认证；双模型共享GPU的对比运行也不能换算成30FPS。30px/持续RAW30FPS目标均未达成。源码冻结守卫、默认176回放0差、可见性力分区、协方差及工厂参数检查已通过；没有持续运行的实验。</p>
<p class="small">原始数据：<a href="paired_scores.json">逐起点评分</a> · <a href="input_coverage.json">入口覆盖统计</a> · <a href="checks.json">机制检查</a> · <a href="../unseen191_20261107/predecode_freeze.json">独立测试事前冻结</a> · <a href="../unseen191_20261107/recording16/completion.json">录像16真实运行</a> · <a href="../unseen191_20261107/recording17/completion.json">录像17失败</a> · <a href="../bootstrap190_20261107/recording18/completion.json">弱周期恢复负例</a></p></section>
</main></html>'''
    (OUT/'index.html').write_text(doc,'utf-8')
