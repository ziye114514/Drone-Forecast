"""Report measured full-pipeline speed, retained accuracy and remaining latency."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,html,json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parent
OUT=WORKSPACE/'output/final_pipeline_speed_20261027'
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'tools'))
from compare_final_pipeline import score


def main():
    load=lambda name:json.loads((OUT/name).read_text('utf-8'))
    side=load('side_comparison.json');clear=load('clear_comparison.json');prefilter=load('side_prefilter_comparison.json')
    practical=load('side_async_prefilter/summary.json')
    rows=[json.loads(s) for s in (OUT/'side_async_prefilter/predictions.jsonl').read_text('utf-8').splitlines()]
    labels=WORKSPACE/'output/dense_labels_side_110_1195/labels.json'
    truths=sorted(json.loads(labels.read_text('utf-8'))['records'].values(),key=lambda z:z['t'])
    practical_score=score(rows,truths)
    (OUT/'side_async_prefilter/manual_score.json').write_text(json.dumps(practical_score,ensure_ascii=False,indent=2),'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    x=np.arange(2);w=.32
    for offset,key,label,color in [(-w/2,'reference','Reference','#8994a5'),(w/2,'optimized','Equivalent compute','#147d92')]:
        values=[case['summaries'][key]['steady_capacity_fps'] for case in [side,clear]]
        axes[0].bar(x+offset,values,w,label=label,color=color)
        for at,value in zip(x+offset,values):axes[0].text(at,value+.5,f'{value:.1f}',ha='center')
    axes[0].axhline(30,color='#b85436',ls='--',label='30 FPS target')
    axes[0].set(xticks=x,xticklabels=['Side110-112s','Clear26-28s'],ylabel='Full pipeline capacity (FPS)',ylim=(0,34),title='Same outputs, short paired RAW runs')
    axes[0].legend(fontsize=8,loc='upper left')
    t=np.array([r['t'] for r in rows]);cost=np.array([r['cost']['service_ms'] for r in rows])
    axes[1].plot(t,cost,color='#147d92',lw=1,label='Compute + decode wait')
    axes[1].axhline(1000/30,color='#b85436',ls='--',label='33.3ms budget')
    axes[1].set(xlabel='Source time (s)',ylabel='Service before output write (ms)',title='Async + decoder filter: spikes remain')
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/'measured_pipeline.png',dpi=150);plt.close(fig)
    body=side['manual_scores']['optimized'];physical=body['physical_future_error_px']['0.4']
    table=''
    for label,case in [('侧视110–112s',side),('清晰26–28s',clear)]:
        old,new=[case['summaries'][k] for k in ['reference','optimized']]
        table+=f'<tr><td>{label}</td><td>{old["steady_capacity_fps"]:.2f}</td><td>{new["steady_capacity_fps"]:.2f}</td><td>{old["timing_ms"]["total_service_ms"]["median"]:.2f}→{new["timing_ms"]["total_service_ms"]["median"]:.2f}ms</td><td>{case["source_frames"]}个逐项相同</td></tr>'
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>最终流水线FPS优化结果</title><style>
body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;background:#f3f6f9;color:#223447}}section{{background:white;padding:22px 28px;margin:18px 0;border-radius:10px}}h1{{font-size:30px}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dae1e8}}img{{max-width:100%}}.note{{background:#fff1d5;padding:12px 16px}}a{{color:#15657f}}code{{background:#edf2f6}}</style>
<h1>把改动落实到完整流水线FPS</h1><p>2026-10-27 · 原生RAW→整机观测→五圆几何→周期代理→二力预测 · 本机短时实测</p>
<section><h2>这次取得的是速度收益</h2><p>两段各60个源时刻，整机框、几何、周期、五圆状态及未来预测逐项相同，平均处理能力上升。默认启用等价计算优化，<code>--reference-compute</code>保留原路径。没有降低输入更新率、少算原生事件、放松资格门槛或把重复输出当新检测。</p><table><tr><th>片段</th><th>原计算FPS</th><th>优化FPS</th><th>每帧总服务耗时中位</th><th>最终输出对照</th></tr>{table}</table><p>计时包含消费者RAW等待、检测、几何、预测、输出写入，稳态不含一次性RAW定位/模型启动；未使用profiler或预览。清晰段反向运行顺序，仍有收益，但每段仅2秒，存在设备时钟、调度和写盘波动，不能把百分比当稳定保证。</p><img src="measured_pipeline.png" alt="完整流水线FPS与实际慢帧"></section>
<section><h2>优化具体落在哪里</h2><ol><li>曲率拆分提前检查全部子弧共有的父轮廓必要条件；第一条弧失败时，停止无用的兄弟弧拟合。</li><li>复用已经算过的弧残差，细线分位数使用等价线性插值，原函数保留。</li><li>每帧KLT直接使用同样的事件占用灰度，只在实际请求神经检测/预览时生成原生彩色图；神经网络帧哈希核验。</li><li>提前启动有界RAW预取，与GPU模型初始化重叠。侧视首次结果{side['startup']['reference']['first_result_after_s']:.2f}→{side['startup']['optimized']['first_result_after_s']:.2f}s，这是启动耗时收益，不是稳态FPS。</li></ol><p>可选<code>--prefilter-decode</code>将同一个SDK降噪器移到解码进程，原顺序/规则不变，另加两个固定共享缓冲，几何仍使用原事件。同步侧视60帧再次逐项相同；主线程服务中位43.43→35.79ms，平均FPS17.57→17.53没有进一步改善，P95还较高，因此保留实验开关。</p></section>
<section><h2>检测精度没有拿候选数来冒充</h2><p>侧视既有20个人工锚点，{body['detected_anchors']}个有当前整机观测；中心误差中位{body['box_center_error_px']['median']:.2f}px，IoU中位{body['box_iou']['median']:.3f}，与原计算完全相同。0.4s物理预测在{physical['n']}个参考目标上的平均误差{physical['mean']:.2f}px，同样未变。</p><p>这些是以前用过的开发标注，锚点与源时刻最多差16.68ms；不是全数据集召回/mAP，也不是物理重心/桨心真值。清晰26–28s没有独立人工框，不能给该段报检测精度。当前没有新增最终精度收益。</p><p>清晰段保留18个four、3个three、5个partial_two时刻，侧视保持轴线/退化语义。没有为了加速取消用户五圆模型，也没有把两个事件块强行命名为两台电机。</p></section>
<section><h2>实际异步输出与30FPS差距</h2><p>降噪解码重叠+异步NN在110–114s处理{practical['frames']}个源时刻，{practical['fresh_image_observations']}个当前整机观测、{practical['valid_forecasts']}个有效预测。平均处理能力{practical['steady_capacity_fps']:.2f}FPS，实际当前观测{practical['paced_fresh_observation_rate']:.2f}Hz；总服务耗时中位{practical['timing_ms']['total_service_ms']['median']:.2f}ms、P95 {practical['timing_ms']['total_service_ms']['p95']:.2f}ms。</p><p class="note">中位数接近33.3ms预算，慢帧仍造成积压：完成延迟中位{practical['timing_ms']['completion_lag_ms']['median']/1000:.2f}s，未达到稳定、低延迟30FPS。所有源帧保留，不能把积压隐藏掉。异步回包时机与同步对照不同，不能把上述FPS都解释为同观测加速。</p><p>该次异步输出单独评分：{practical_score['manual_anchor_count']}个人工锚点中{practical_score['detected_anchors']}个有观测，中心误差中位{practical_score['box_center_error_px']['median']:.2f}px、IoU中位{practical_score['box_iou']['median']:.3f}；它包含更长时间段，不能与上面2秒直接推导精度变化。</p></section>
<section><h2>下一处值得优化的地方</h2><p>异步组合中，几何平均{practical['timing_ms']['structure_ms']['mean']:.1f}ms、定位{practical['timing_ms']['localize_ms']['mean']:.1f}ms、周期{practical['timing_ms']['period_engine_ms']['mean']:.1f}ms，几何与间歇慢帧仍是主要预算。下一步优先把周期处理和几何的独立部分重叠、减少重复扫描，以及给昂贵的重发现独立慢速调度，保持失锁恢复。</p><p>较长的4秒异步片段0.4s预测平均误差{practical_score['physical_future_error_px']['0.4']['mean']:.2f}px，共{practical_score['physical_future_error_px']['0.4']['n']}个开发参考目标；检测中心较准并不意味着推力/扰动预测已经准确。该范围比2秒对照更长，不能直接用两组误差之差评价算法改动，但这一绝对误差说明物理预测仍有明显缺陷。</p><p>上轮逐像素预测式谐波仍是实验观测，没有硬塞每帧流程。其固定25%像素CUDA组件36–61ms，当前直接增加会拖慢流水线；周期像素增多或合格单弧增加不能证明最终精度提升。后续应只在不合格/失锁ROI触发，合格局部弧仅更新可观测方向。</p></section>
<section><h2>复现与交接</h2><p><a href="../../event_drone_forecast/docs/FINAL_PIPELINE_SPEED.md">改动、入口和口径</a> · <a href="side_comparison.json">侧视逐项对照与人工评分</a> · <a href="clear_comparison.json">清晰段逐项对照</a> · <a href="side_prefilter_comparison.json">解码降噪对照</a> · <a href="side_async_prefilter/summary.json">异步完整耗时</a> · <a href="side_async_prefilter/manual_score.json">异步独立评分记录</a> · <a href="../predictive_harmonics_20261027/index.html">谐波研究记录</a></p><p>每次RAW实际代码保存在各运行的source_snapshot，旧函数、旧控制、失败实验保留。已上传预览ZIP字节不变，本轮没有重新打包或发布GitHub。WORKLOG和latest_handoff已更新。</p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    local=[p for p in re.findall(r'(?:href|src)="([^"#]+)"',report) if '://' not in p]
    assert all((OUT/p).is_file() for p in local)
    frozen=WORKSPACE/'output/github_release_final_20261027'
    archive=frozen/'event-drone-forecast-v0.4.0-alpha.1.zip';digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest=='ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'
    old=frozen/'event-drone-forecast/src/event_drone_forecast/position'
    same=[p.name for p in old.glob('*.py') if p.read_bytes()==(ROOT/'src/event_drone_forecast/position'/p.name).read_bytes()];assert len(same)==7
    paths=['run_fast_structure.py','fast_raw_reader.py','fast_structure_front.py','fast_geometry_compute.py','native_render.py',
        'tools/check_final_compute.py','tools/compare_final_pipeline.py','tools/profile_geometry_budget.py','tools/profile_raw_pipeline.py',
        'tools/report_final_pipeline.py','tools/prepare_github_release.py']
    snapshot=OUT/'final_source_snapshot';snapshot.mkdir(exist_ok=True);hashes={}
    for name in paths:
        data=(ROOT/name).read_bytes();ast.parse(data,filename=name);dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    checks=dict(frozen_ZIP_sha256=digest,position_core_files_unchanged=same,links_checked=len(local),
        exact_source_samples_side_clear=120,exact_prefilter_side_samples=60,all_RAW_runs_unchanged=True,
        default_equivalent_compute=True,prefilter_default=False,new_detection_accuracy_gain=False,stable30FPS=False,current_jobs=[],sources=hashes)
    (OUT/'artifact_checks.json').write_text(json.dumps(checks,indent=2),'utf-8')
    checkpoint=WORKSPACE/'output/checkpoints/latest_handoff.json';handoff=json.loads(checkpoint.read_text('utf-8'))
    handoff.update(updated_utc=datetime.now(timezone.utc).isoformat(),current_work_state='Completed full-pipeline equivalent-compute optimization and actual RAW paired timings; no jobs',
        current_jobs=[],current_running_jobs=[],execution_status='Completed current optimization batch; broader research incomplete; no background jobs',latest_research_report=str(OUT/'index.html'))
    handoff['full_pipeline_compute_optimization']=dict(report=str(OUT/'index.html'),method=str(ROOT/'docs/FINAL_PIPELINE_SPEED.md'),
        side_reference_optimized_fps=[side['summaries'][k]['steady_capacity_fps'] for k in ['reference','optimized']],
        clear_reference_optimized_fps=[clear['summaries'][k]['steady_capacity_fps'] for k in ['reference','optimized']],
        exact_same_outputs_side_clear=120,side_manual_center_median_px=body['box_center_error_px']['median'],
        async_prefilter_fps=practical['steady_capacity_fps'],async_fresh_Hz=practical['paced_fresh_observation_rate'],
        default_equivalent_compute=True,reference_flag='--reference-compute',prefilter_optional=True,
        stable30FPS=False,accuracy_improved=False,async_0_4s_mean_px=practical_score['physical_future_error_px']['0.4']['mean'],
        next='Overlap independent period/geometry work or remove repeated scans; control latency spikes. Trigger costly harmonic rediscovery only on loss/unqualified ROI; local arcs constrain observable directions. Preserve same-output control and frozen ZIP.')
    checkpoint.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
    marker='## 2026-10-27 完整流水线等价计算与FPS优化';worklog=WORKSPACE/'WORKLOG.md'
    if marker not in worklog.read_text('utf-8'):
        with worklog.open('a',encoding='utf-8') as stream:
            stream.write('\n\n'+marker+'\n\n'
                '- 先剖析实际完整RAW，发现几何/过滤/定位与串行RAW寻址+GPU启动。不是用组件候选数当精度。新增fast_geometry_compute：父轮廓必要条件提前拒绝、失败子弧短路、残差复用、等价分位插值；原函数保留。native_render直接KLT占用图、神经请求时才生成原BGR，提前有界预取与模型启动重叠。\n'
                '- 两段各2s/60时刻，同步NN固定观测、清晰段反向顺序；最终框/几何/周期/模型/预测全部逐项相同。side13.7066→17.5733FPS，clear14.7780→22.1882FPS；总服务中位56.27→43.43、56.41→45.45ms，短测含调度/写盘抖动，不承诺固定加速比例。side首次结果35.11→19.59s。\n'
                '- side20个人工开发锚点19有观测，中心中位5.9216px、IoU中位.6235；0.4s9个参考目标平均37.815px，前后不变。clear26–28s无独立人工框，不能称最终精度改善。120相同时刻之外另prefilter60时刻也相同。\n'
                '- 可选--prefilter-decode在子进程原顺序同SDK过滤，增加两个固定共享缓冲；几何仍原事件。side同步总服务中位43.43→35.79ms，但均值17.57→17.53FPS、P95更差，保留可选不默认。异步+prefilter4s：120源/115当前整机/111有效预测，24.7729FPS/当前23.0222Hz，中位30.49ms/P9591.11ms，完成延迟中位1.126s，未达稳定低延迟30FPS。\n'
                '- 默认启用等价计算，--reference-compute恢复原路径。原位置7文件、已发布ZIP哈希保留，不修改保护目录。报告output/final_pipeline_speed_20261027/index.html；各RAW代码快照及最终快照齐备，WORKLOG/latest_handoff已更新，无后台作业。下一步独立周期/几何重叠、减少重复扫描/慢帧；昂贵谐波只触发式重发现，未入默认、没有新精度收益。\n')
    print(json.dumps(dict(report=str(OUT/'index.html'),async_manual={k:v for k,v in practical_score.items() if k!='anchors'},checks=checks),ensure_ascii=False))


if __name__=='__main__':main()
