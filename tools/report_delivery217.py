"""Consolidate actual gains, regressions, latency and failed hypotheses."""
from pathlib import Path
import sys,json,html,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output';sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
OUT=BASE/'precision_delivery_20261108'
def read(p):return json.loads((BASE/p).read_text('utf-8'))
def rows(p):return [json.loads(s) for s in (BASE/p).read_text('utf-8').splitlines()]

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    adaptive=read('credit202_20261108/scores.json');old=read('separated201_20261107/scores_all.json')
    extra=read('regional212_20261108/scores11.json');oldlabels=read('separated201_20261107/scores_old_labels.json')
    metrics=[]
    for n in ['15','14','16','13','11','side','train2']:
        if n=='11':a=extra['plane191'];b=extra['adaptive'];sep=None
        elif n in old:a=old[n]['plane191'];b=adaptive[n];sep=old[n]['separated']['error']['mean']
        else:
            a=dict(error=dict(mean={'side':42.06618801605331,'train2':86.79362736805906}[n]));b=adaptive[n];sep=oldlabels[n]['separated']['error']['mean']
        metrics.append(dict(recording=n,n=b['available'],baseline=a['error']['mean'],adaptive=b['error']['mean'],
            separated=sep,p95=b['error']['p95'],delta=b['error']['mean']-a['error']['mean'],
            reference='人工开发框' if n in ('side','train2') else '固定8/4事件轮廓弱参考（非人工/COM真值）'))
    fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(len(metrics));w=.27
    ax.bar(x-w/2,[v['baseline'] for v in metrics],w,label='191：原方向未知力',color='#8b98ae')
    ax.bar(x+w/2,[v['adaptive'] for v in metrics],w,label='202：快速法向 / 平滑外扰 + 因果信用',color='#2375bd')
    ax.axhline(30,color='#d75539',linestyle='--',label='30px目标')
    ax.set_xticks(x,[v['recording'] for v in metrics]);ax.set_ylabel('0.4秒平均终点误差 / px（越低越好）');ax.set_xlabel('不同录像，不是同一条轨迹切窗')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'paired_errors.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for n,ax in zip(['13','14','15','16'],axes.flat):
        for name,data,color in [('191',old[n]['plane191'],'#8b98ae'),('202',adaptive[n],'#2375bd')]:
            points=[v for v in data['records'] if v['valid']]
            ax.plot([v['t'] for v in points],[v['error_px'] for v in points],'-o',ms=3,color=color,label=name)
        ax.axhline(30,color='#d75539',ls='--',lw=.8);ax.set_title(f'录像{n}：全部可评分起点');ax.set_ylabel('误差 / px');ax.grid(alpha=.2);ax.legend()
    fig.supxlabel('预测起点 / 秒');fig.tight_layout();fig.savefig(OUT/'all_origins.png',dpi=145);plt.close(fig)
    runtimes=[]
    for name,path in [('201：旧几何 / separated','separated201_20261107/raw13'),
        ('210：种子算子 / adaptive','native210_20261108/raw13_adaptive_native'),
        ('215：完整搜索 / adaptive','arc215_20261108/raw13_adaptive_search'),
        ('216：原生不确定度 / adaptive','uncertainty216_20261108/raw13_adaptive_uncertainty'),
        ('218：限制积压 / adaptive','live218_20261108/raw13_latest'),
        ('220：独立解码（更差）','process220_20261108/raw13_latest'),
        ('222：10Hz弧搜索（候选）','multirate221_20261108/raw13_latest')]:
        if not (BASE/path/'summary.json').exists():continue
        d=read(path+'/summary.json');runtimes.append(dict(name=name,path=path,**d))
    tr=''.join(f'<tr><td>{html.escape(v["name"])}</td><td>{v["steady_pipeline_fps"]:.2f}</td><td>{v["steady_pipeline_p95_ms"]:.1f}</td><td>{v["actual_output_age_ms_median"]:.0f}</td><td>{v["actual_valid_output_count"]}/{v["actual_output_count"]}</td><td>{v["actual_valid_output_rate_hz"]:.2f}</td></tr>' for v in runtimes)
    table=''.join(f'<tr><td>{v["recording"]}</td><td>{v["n"]}</td><td>{v["baseline"]:.2f}</td><td>{v["adaptive"]:.2f}</td><td class="{"gain" if v["delta"]<0 else "loss"}">{v["delta"]:+.2f}</td><td>{v["p95"]:.2f}</td></tr>' for v in metrics)
    all_results=dict(metrics=metrics,runtimes=runtimes,targets_achieved=False,NN_used=False,
        regional_statistic208_supersedes194=True,source_time_error_is_not_actual_publication_error=True,
        independent_test_ground_truth_available=False,old_manual_development_boxes=True)
    (OUT/'results.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2),'utf-8')
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>预测精度与实时性：11月8日交付</title>
<style>body{{font:16px/1.75 'Microsoft YaHei',sans-serif;color:#233148;background:#f1f4f8;margin:0}}main{{max-width:1120px;margin:auto;padding:32px}}section{{background:white;padding:24px 32px;margin:20px 0;border-radius:12px}}h1,h2{{line-height:1.35}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dde5ed;text-align:left}}img{{max-width:100%;height:auto}}.gain{{color:#138056}}.loss{{color:#b24a2a}}.note{{background:#fff1d6;padding:16px}}code{{background:#edf1f6;padding:3px}}a{{color:#176eb4}}small{{color:#59697d}}</style><main>
<h1>让桨叶分支产生实际预测增量</h1><p class="note"><b>用户最新说明：GPU正在使用。</b>近期实机运行不是受控独占测速；GPU共用起点未记录，表中FPS、延迟、实际发布误差仅代表当时共享负载下结果。不能据此断言算法退化、GPU单独能力不足或稳定帧率。所有原始数值保留并附条件旁注；GPU空闲前不再运行GPU实验。固定源观测的数值精度对照仍保留。</p><p>2026-11-08 · 纯原生ON/OFF · 五圆/侧视分工 · 可测力 + 不可测力 · 保留原基线</p>
<section><h2>目前做成了什么</h2><p>新增可选 <code>plane-adaptive</code>：快速的法线方向未知推力响应，与缓慢外扰及其变化率分别建模；只有过去预测到期、当前真实观测支持增量后才提高权重。没有用NN、录像编号、未来标签或机型参数选择模型。三段录像的0.4秒均值下降1.6–3.9px，另两段新录像轻退，旧train2仍然差。尚未稳定达到30px。</p>
<img src="paired_errors.png"><table><tr><th>录像</th><th>同起点数</th><th>191 / px</th><th>202 / px</th><th>变化 / px</th><th>202 P95 / px</th></tr>{table}</table>
<p class="note">13/14/15/16/11使用固定原生事件8/4阈值轮廓中心弱参考，已逐页目视检查，但不是独立人工框或重心真值。15用于开发；14/16是随后验证；13先冻结后测试201，随后已用于202开发；12是202首次新录像失败后用于前端开发；11是212全套冻结后才解码的新录像。11对比仅15/25起点可评分，未隐藏缺失，仍显示退化。几像素差异还不能当严格科研结论。</p></section>
<section><h2>你的二力模型如何发挥作用</h2><p><b>a = k·u + d，d = e + h·n。</b>u是当前可信可见桨叶的频率平方变化代理；隐藏桨叶、未知尺度/重力/风/阻力等仍在d。h允许沿当前合格法线快速变化，e及其变化率平滑演变。这里的单位是图像加速度，不声称测得牛顿或真实电机RPM。</p>
<p>202同时保留191和分离响应，使用既有3秒因果信用记忆，初始新模型权重为0。到期前不读取目标；到期后用当前原生观测评价增量，给二力响应争取权重。模型共享同一组桨叶观测，不能作为两个独立传感器融合。旧train2直接换分离响应会86.79→98.64px，因果信用后87.17px；这避免大幅退化，但未让该段变好。</p>
<p>同总噪声、同资格各向同性控制，以及去外扰变化率外推控制都保留在 <a href="../separated201_20261107/scores_all.json">原始对照</a>。没有把加大过程噪声冒充法线建模收益。</p><img src="all_origins.png"></section>
<section><h2>只看桨叶的滤波：修了一个真实漏洞</h2><p>193/194原型每毫秒选“第一个事件”，会给稠密随机噪声制造约960Hz假周期。208改为与时间无关的最小固定hash选择，并去除短块线性包络、采用保守方差界。八组稠密非周期负例0通过，稀疏周期正例通过，NumPy/C++/GPU对照一致。旧194区域计数不再视为可靠收益；主预测没有启用那一版。</p>
<p>211使用固定形状CUDA Graph，在逐像素入口失败时进行区域时间确认，保留原始ON/OFF计数和真实像素。12整段181帧：区域4→28、有效预测0→14，但只有很短一段恢复，不能冒充90%检出率，也尚无独立预测FDE证据。区域周期不确定电机身份：恢复帧的已测推力信用强制清空，全部推力进不可测项。12实际177次补位调用，中位10.34ms/P9513.56ms，仍有20.85ms长尾；这是检验算子耗时，不是整机FPS。</p>
<p>预先冻结的新录像11：99个区域、100个有效预测，区域补位未新增检出；0.4秒均值38.22→39.63px，P95 74.02→69.33px。前期稀疏/快速移动仍是难点。</p></section>
<section><h2>真实RAW速度：吞吐和可用发布必须分开</h2><p>215原生完整弧搜索保留原有随机种子、全部点、筛选顺序、阈值及Python不确定度。216进一步移植相同十次中心差分/MAD/SVD截断计算，65弧接受一致，181帧资格/预测保留，最大数值差1.27e−13。65真实弧接受一致，算子123→46ms；录像13完整181帧几何、状态、预测最大差2.22e−16。该速度优化本身不改变预测精度。</p><table><tr><th>完整配置（有差异）</th><th>吞吐FPS</th><th>P95周期ms</th><th>观测年龄中位ms</th><th>有效发布/总发布</th><th>有效Hz</th></tr>{tr}</table>
<p><b>实际发布时间评分：</b>218在25个固定墙钟源起点中19个可用，0.4秒均值60.90px。保留同样19个起点和丢帧序列、假设计算零延迟才可获得的反事实均值37.96px；它不是真实运行结果。这说明仅看缓存或源时间误差会掩盖严重的延迟损失。220独立解码配置实测更差，不推荐。</p><p class="note">同一RAW窗口，以上是不同完整配置，不能把差异全归因给单一算子。无丢帧路径若低于输入30Hz会持续积压；源时间预测有效不代表发布时仍有效。记录器按实际年龄作判断，绝不把重复外推/补帧当30FPS检测。</p></section>
<section><h2>现在真正卡在哪里</h2><ol><li>未知法向推力响应有可见增量，但缺少足够稳定、可观测的当前转速/法线证据；不能仅凭位置误差唯一分辨隐藏推力、风、阻力和尺度变化。</li><li>逐像素重复门槛对快速跨像素移动目标过严；区域检验只补回局部，不能保证弱光、出画、小目标全部发现。</li><li>几何长尾仍会消耗预测时域。已有原生加速与受限积压；近期GPU共用，先等待空闲做受控分段计时，再判断哪个阶段需要优化，记录真实输出频率、年龄和缺失比例。</li><li>参考只有旧人工开发框和弱事件轮廓，且可用片段少；对新录像的退化必须保留，当前不能声称跨场景30px或优于前沿。</li></ol>
<p>下一步优先让“当前可测周期 + 当前法线”在新序列提供稳定有效输入；区域只有旋转证据时继续更新位置与未知力，不补造四桨RPM。先解决入口与延迟，再扩大完整RAW外推对照，避免继续按录像调参。</p></section>
<section><h2>后续225–230：几何先行与完整性修复</h2><p>早半窗原始ON/OFF几何提出桨心候选，后半窗冻结确认1–4阶角相位；三叶旋转合成例能通过而旧二阶失败，但真实困难11/12仍未合格，不能声称已测得推力或有新增预测收益。八组非周期环和整体闪烁环均拒绝，不是相机FAR标定。</p><p>发现并修复候选路径的FOV资格：旋转椭圆必须用实际投影边界，裁剪环不计完整平面。13/15/16同起点均值分别37.4869→37.5164、48.9461→48.4756、44.6710→44.8508px，局部收益与退化都保留。选择开关<code>--rotor-projected-fov-guard</code>；累计研究入口<code>run_rotor_candidate.py</code>默认启用，原入口默认保留。当前不含未成功的角相位实验。</p></section><section><h2>代码与复现入口</h2><p>旧默认free/176保留；<code>--rotor-unknown-force-model plane-adaptive</code>启用因果响应；<code>--rotor-regional-graph</code>是实验性原生GPU补位，仍需真实泛化评价；<code>--rotor-native-geometry-dll</code>只加载显式指定的本地编译算子。公开NN发布包/ZIP和两处yolov10均未改。</p><p>核心文件：adaptive_response_observer.py、causal_response_credit.py、separated_force_response.py、regional_graph_front.py、cuda_regional_graph.py、native/arc_search215.cpp。完整快照与参数在每次运行目录中。详见 <a href="../../event_drone_forecast/docs/HANDOFF_20261108_CURRENT230.md">进行中交接</a>。</p>
<p>方法参考：<a href="https://arxiv.org/html/2408.06899v3">EEPPR</a>、<a href="https://arxiv.org/abs/2603.08386">DDHF</a>、<a href="https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf">Quadrotor rotor drag</a>。只借鉴机制，未复制作者源码或声称论文完整复现。</p></section></main></html>'''
    (OUT/'index.html').write_text(doc,'utf-8');print(OUT/'index.html')
