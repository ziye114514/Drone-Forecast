"""Build a compact local report from completed mechanism outputs, no inference."""
from pathlib import Path
import hashlib,html,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;BASE=WORK/'output/helix_author_probe_20261025'
OUT=WORK/'output/optimization_run_20261025';OUT.mkdir(exist_ok=True)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
previous=rows(BASE/'RAW_natural_extension/predictions.jsonl');current=rows(BASE/'RAW_shadow_recovery/predictions.jsonl')
old=rows(WORK/'output/stage156_parallel_geometry_20261024/clear/predictions.jsonl')
fixed=read(BASE/'causal_extension/results.json');moving=read(BASE/'adaptive_signed/probe/results.json')
speed=read(BASE/'adaptive_signed/probe_roi_parallel/equivalence.json')
summary=dict(status='ongoing optimization checkpoint; not a paper or completed product',
    publication_clock='independent5Hz live default; invalid heartbeat included',
    default_research_prediction='138 geometry +140 motion +144 damping;156 remains optional',
    no_new_forecast_accuracy_gain_proven=True,independent_RPM_or_force_truth=False,
    main_input='RAW + fixed weights only; current detector geometry initializes local author phase module',
    fixed_roi_extension_all_four=fixed['summary']['new_extension']['all_four_supported'],
    moving_roi_extension_all_four=moving['summary']['new_extension']['all_four_supported'],
    moving_roi_extension_ticks=40,phase_speed_probe=speed,cases={})
cases=[('single_initialization','RAW_natural_extension'),('whole_model_reset','RAW_reanchor'),
       ('partial_reset','RAW_partial_reanchor'),('parallel_recovery','RAW_shadow_recovery')]
for name,directory in cases:
    seq=rows(BASE/directory/'predictions.jsonl')
    obs=[r['auxiliary_observation'] for r in seq]
    exact=all(a['t']==b['t'] and a['body']==b['body'] and a['coefficients']==b['coefficients']
        and a['predictions']==b['predictions'] and a['valid']==b['valid'] for a,b in zip(seq,old)) and len(seq)==len(old)
    case=BASE/directory
    protocol=read(case/'protocol.json');own=protocol['optional_observation_extension']['source_sha256']
    sources_unchanged=all(hashlib.sha256((case/'source_snapshot'/n).read_bytes()).hexdigest()==sha for n,sha in own.items())
    item=dict(frames=len(seq),all_four_supported=sum(r.get('valid',False) for r in obs),
        initialized_times=[r['t'] for r in obs if r.get('initialized_now')],
        reanchor_times=[r['t'] for r in obs if r.get('reanchored_now')],
        module_processing_median_ms=float(np.median([r['processing_ms'] for r in obs])),
        all_baseline_coordinates_exact=exact,phase_sources_unchanged=sources_unchanged,
        total_processing_median_ms=read(case/'completion.json')['processing_median_ms'],
        periods={})
    for lo,hi in [(22,25),(25,26),(26,28),(28,30)]:
        block=[r['auxiliary_observation'] for r in seq if lo<r['t']<=hi]
        item['periods'][f'{lo}-{hi}']=dict(frames=len(block),all_four=sum(r.get('valid',False) for r in block))
    assert exact and sources_unchanged
    summary['cases'][name]=item
summary['parallel_recovery_checks']=read(BASE/'RAW_shadow_recovery/comparison.json')
summary['side_periodic_axis']=read(BASE/'RAW_side_axis/comparison.json')
summary['rotation_increment_replay']=read(BASE/'forecast_increment_attribution/results.json')['summary']
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
for seq,label in [(previous,'Original phase tracks'),(current,'Keep original + validated recovery candidate')]:
    axes[0].step([r['t'] for r in seq],[r['auxiliary_observation'].get('supported_candidates',0) for r in seq],where='post',label=label)
for i in range(4):
    y=[]
    for r in current:
        v=r['auxiliary_observation'].get('rotors',{}).get(str(i),{})
        y.append(v['selected']['phase_hz'] if v.get('valid') else np.nan)
    axes[1].plot([r['t'] for r in current],y,label=f'Visual slot{i} (IDs reset at new epoch)')
for epoch in sorted({r['auxiliary_observation'].get('rotation_proxy_epoch',0) for r in current}):
    seq=[r for r in current if r['auxiliary_observation'].get('rotation_proxy_epoch',0)==epoch]
    proxy=[(r['auxiliary_observation'].get('conditional_rotation_proxy') or {}).get('relative_to_first_supported',np.nan) for r in seq]
    proxy=[x if x is not None else np.nan for x in proxy]
    if np.isfinite(proxy).any():axes[2].plot([r['t'] for r in seq],proxy,label=f'Epoch{epoch} local reference')
for ax in axes:
    ax.grid(alpha=.25);ax.legend(loc='best')
axes[0].set_ylim(-.1,4.1);axes[0].set_yticks(range(5));axes[0].set_ylabel('Current supported candidates')
axes[1].set_ylabel('Supported MODEL phase Hz');axes[2].set_ylabel('Conditional rotation activity')
axes[2].set_xlabel('RAW seconds');fig.suptitle('400 actual RAW+NN steps; observation improvement is NOT flight accuracy or RPM truth')
fig.tight_layout();fig.savefig(OUT/'phase_progress.png',dpi=150);plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
a=summary['cases']['single_initialization'];b=summary['cases']['parallel_recovery']
table=''.join('<tr><td>'+p+'s</td>'+''.join(f'<td>{summary["cases"][key]["periods"][p]["all_four"]}/{summary["cases"][key]["periods"][p]["frames"]}</td>' for key,_ in cases)+'</tr>' for p in a['periods'])
body=f'''<!doctype html><meta charset="utf-8"><title>持续优化：发布调度与桨盘旋转观测</title>
<style>body{{font:17px/1.7 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;background:#f7f8fa;color:#19232d}}h1,h2{{line-height:1.3}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccd4dd;padding:9px 18px}}img,video{{max-width:100%}}code{{background:#edf1f5}}a{{color:#1768ae}}</style>
<h1>持续优化：发布调度与桨盘旋转观测</h1><p>2026-10-25 · 进行中的工程检查点 · 未发布GitHub</p>
<section><h2>这次实际改了什么</h2><p>live改为独立5Hz定时发布；缺新鲜观测时输出无效心跳。6/4/2/1秒RAW片段发布30/20/10/5次，有效16/16/9/0次，迟到P95约1–2ms。解决发布节拍，不等于连续有效5Hz。</p>
<p>桨盘通道直接运行用户下载的HelixTrack原作者核心，增加正反初始方向假设、跟随当前模型的事件采样区、独立当前像素周期核验及复锁。原作者仓库不改动，代码与DLL限本地，不进源码候选包。</p>
<p>四轮各400个实际RAW时刻的原位置预测坐标与156对应片段完全一致。这一轮改善旋转观测和调度，<strong>尚未证明提高未来位置精度、真实RPM或推力精度</strong>。1600时刻是同一8秒开发录像的四轮试验，不是独立数据。</p></section>
<section><h2>旋转跟踪的具体结果</h2><p>先前1.2秒观察探针：后0.8秒，四桨同时支持从固定ROI的0/40次升至移动ROI的25/40次。无抽稀的解析ROI预筛选+四桨并行，480个相位状态数值最大差0；单次开发探针处理中位323.8→97.0ms。硬件/负载未控制，不是固定倍数承诺，也不是20ms实时。</p>
<p>之后接入真正RAW+NN主入口，从22.86秒<strong>当前</strong>四椭圆自动初始化，不输入旧框、旧几何、标签或飞控。单次初始化8秒四桨支持156/400；整套重置145/400、局部重置149/400，都不采用。保留原轨迹并行核验恢复候选后161/400，增加5个时刻。统计的是算法条件支持次数，<strong>不是四电机正确率或轨迹精度</strong>。</p>
<table><tr><th>录像时间</th><th>原相位轨迹</th><th>整套重置</th><th>局部重置</th><th>并行候选</th></tr>{table}</table>
<p>并行版保留的原轨迹全部相位状态数值最大差0、有效性不变。原通道仍合格时优先保留；恢复候选必须得到新的原生相位、当前像素周期及位置唯一性支持。没有把推测的第四桨当观测，未取得电机ID真值；恢复范围有限，仍为本机可选实验，没有升主预测默认。</p>
<img src="phase_progress.png"><p>模型相位Hz与机械旋转Hz必须区分。相对Σf²仅在四路当前支持且像素谐波签名未变时输出；各身份段分别归一化，不能当牛顿推力。</p></section>
<section><h2>实际RAW与原预测叠图</h2><video controls preload="metadata" src="../helix_author_probe_20261025/RAW_shadow_recovery/phase_and_forecast.mp4"></video>
<p>绿色为原位置预测，黄色是当前相位+像素周期均支持的模型轮廓。50fps是离线录像时间轴，不能作为实时性能证明。未画出的桨表示当前证据不够。</p>
<img src="../helix_author_probe_20261025/RAW_shadow_recovery/native_26.66.png"></section>
<section><h2>侧视不再只等待四桨</h2><p>复用项目141原有周期像素线轴算法，提取为无状态接口并共享同一SDK频率包。侧视110—111.2秒60个实际RAW时刻：四桨相位0，当前整机周期49/60、周期投影线轴21/60。其余时刻明确证据不足；原位置预测逐项保持，整链中位73.23ms。</p>
<video controls preload="metadata" src="../helix_author_probe_20261025/RAW_side_axis/phase_and_forecast.mp4"></video>
<p>青色是当前周期支撑的无向图像轴，不是飞行速度方向、两根物理转轴或完整有向3D姿态。没有由一个混合频率虚构四电机总推力。</p></section>
<section><h2>旋转增量有没有帮助预测</h2><p>在同一161个合格起点做已到期误差反馈的因果重放，只用当前同一NN框反馈，不使用人工标签。三个数字依次为原预测/同门控纯偏差校正/加入旋转增量：</p>
<table><tr><th>时域</th><th>原预测px</th><th>同门控纯偏差px</th><th>旋转增量px</th></tr><tr><td>0.1s</td><td>4.395</td><td>4.294</td><td>4.247</td></tr><tr><td>0.2s</td><td>7.454</td><td>7.468</td><td>7.527</td></tr><tr><td>0.4s</td><td>16.650</td><td>16.650</td><td>16.650</td></tr></table>
<p>0.1s有微小增量，0.2s反而退化，0.4s未通过过去损失门控。没有一致收益，暂不接入主入口、不继续扫权重。这里是检测器自洽分析，不能与全400帧或人工框的历史分数跨口径比较。</p></section>
<section><h2>当前还卡在哪里</h2><p>桨盘严重重叠或压扁时，四路独立旋转证据仍会不足；高相位集中度本身可能给出错误低频解，必须联合当前像素周期。质量、尺度、桨叶数/谐波、推力系数与风仍未分别可辨识。作者核心计算也偏重：复锁版整链中位{b['total_processing_median_ms']:.1f}ms/原生20ms时刻，观测模块中位{b['module_processing_median_ms']:.1f}ms；当前是离线研究旁路。</p>
<p>下一步应把合格旋转信号的增量与纯位置偏差校正做同起点比较；若没有超出位置分支的收益，不提高物理融合权重。进入缺桨段时先使用可信部分证据与显式缺失，禁止凑四桨强行算总推力。</p></section>
<section><h2>停止继续调的负结果</h2><p>连续时间Q跨录像结果混合；ByteTrack弱框关联没有增加有效输出；已到期轨迹残差的幅度增益略退。它们均不升默认。直接均匀抽稀到每桨每20ms1000事件虽降耗时，却破坏跟踪，已放弃。</p>
<p>整套重置和局部重置均无净收益，默认关闭；并行保留恢复候选仅有小幅条件支持收益。状态保留/再关联已有工程与论文基础，如<a href="https://docs.nvidia.com/deepstream/text/DS_plugin_gst-nvtracker">NVIDIA目标管理与再关联</a>及<a href="https://github.com/noahcao/OC_SORT">OC-SORT作者项目</a>。本次不是直接移植它们的实现，也不主张首次提出这些机制。</p>
<p>本轮所有录像都是已知开发录像，人工框规模未增加；不能宣称泛化、SOTA、真实推力或首次性。</p></section>
<p><a href="summary.json">原始汇总JSON</a> · <a href="../../event_drone_forecast/docs/OPTIMIZATION_JOURNAL_20261025.md">持续记录</a> · <a href="../../WORKLOG.md">WORKLOG</a> · <a href="https://github.com/radimspetlik/helixtrack">HelixTrack原项目</a> · <a href="https://github.com/FoundationVision/ByteTrack">ByteTrack原项目</a></p>'''
(OUT/'index.html').write_text(body,encoding='utf-8')
print(json.dumps(summary['cases'],ensure_ascii=False),flush=True)
