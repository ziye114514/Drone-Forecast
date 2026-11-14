"""Collect the bounded pilot, including regressions and failed attempts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,html,json,py_compile,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/five_circle_fast30_20261026'


def load(p):return json.loads(p.read_text('utf-8'))
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')
def main():
    names=['side_capacity_v2','side_thread_bounded','side_async','side_async_fast','clear_async',
        'clear_physical_only','clear_native_roi','clear_prefetch','clear_final','clear_release',
        'clear_process','clear_release_v2','side_release']
    summaries={name:load(OUT/name/'summary.json') for name in names}
    final={name:summaries[name] for name in ['clear_release_v2','side_release']}
    accuracy=load(OUT/'side_release/accuracy.json');controlled=load(OUT/'mechanism/controlled.json')
    rows={name:[json.loads(s) for s in (OUT/name/'predictions.jsonl').read_text('utf-8').splitlines()] for name in final}
    audit={}
    for name in final:
        protocol=load(OUT/name/'protocol.json')
        audit[name]=dict(recorded_sources_match=all(hashlib.sha256((OUT/name/'source_snapshot'/p).read_bytes()).hexdigest()==sha for p,sha in protocol['sources'].items()),
            no_traditional_prediction_in_runtime=protocol['physical_only'] and all(not r['baseline'] and not r['cv'] for r in rows[name]),
            strictly_increasing_t=bool(np.all(np.diff([r['t'] for r in rows[name]])>0)),
            no_event_from_future=all(r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6) for r in rows[name]),
            native_video=load(OUT/name/'render_check.json'))
        assert all(audit[name][k] for k in ['recorded_sources_match','no_traditional_prediction_in_runtime','strictly_increasing_t','no_event_from_future'])
    original=WORK/'output/flow_acceleration_20261026/actual_side_headless/source_snapshot/src/event_drone_forecast/position'
    baseline_files={p.name:p.read_bytes()==(ROOT/'src/event_drone_forecast/position'/p.name).read_bytes() for p in original.glob('*.py')}
    assert baseline_files and all(baseline_files.values())
    for f in ['run_fast_structure.py','fast_raw_reader.py','fast_structure_front.py','fast_force_response.py','five_circle_model.py','live_decode.py','forecast_mainline.py']:
        py_compile.compile(str(ROOT/f),doraise=True)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,(name,data) in zip(axes[0],rows.items()):
        t=np.array([r['t'] for r in data]);ms=[r['cost']['service_ms'] for r in data]
        ax.plot(t[1:],ms[1:],label='Compute + decoder wait',lw=1.3);ax.axhline(1000/30,color='red',ls='--',label='30Hz budget')
        ax.set(xlabel='Native source time [s]',ylabel='Service time [ms]',title=name);ax.legend(fontsize=8)
    data=rows['side_release'];t=[r['t'] for r in data]
    axes[1,0].plot(t,[r['force']['effective_gain_px_s2'] for r in data],label='Effective image gain')
    axes[1,0].set(xlabel='Native source time [s]',ylabel='Gain [px/s2]',title='Online gain is NOT validated Newton thrust');axes[1,0].legend(fontsize=8)
    hs=['0.1','0.2','0.4'];x=np.arange(3)
    for shift,name in [(-.24,'cv'),(0,'kinematic'),(.24,'physical')]:
        axes[1,1].bar(x+shift,[accuracy['scores'][h][name]['mean'] for h in hs],width=.24,label=name)
    axes[1,1].set(xticks=x,xticklabels=hs,xlabel='Horizon [s]',ylabel='Mean endpoint error [px]',title='14 sparse development targets per horizon');axes[1,1].legend(fontsize=8)
    fig.savefig(OUT/'results.png',dpi=140);plt.close(fig)
    body=[]
    for name,s in final.items():
        body.append(f'<tr><td>{name}</td><td>{s["frames"]}</td><td>{s["steady_capacity_fps"]:.2f}</td><td>{s["paced_fresh_observation_rate"]:.2f}</td><td>{s["timing_ms"]["total_service_ms"]["p95"]:.1f}</td><td>{s["current_force_inputs"]}</td><td>{html.escape(str(s["modes"]))}</td></tr>')
    experiments=''.join(f'<tr><td>{name}</td><td>{s["steady_capacity_fps"]:.2f}</td><td>{s["fresh_image_observations"]}/{s["frames"]}</td></tr>' for name,s in summaries.items())
    score=''.join(f'<tr><td>{h}s</td><td>{accuracy["scores"][h]["cv"]["mean"]:.2f}</td><td>{accuracy["scores"][h]["kinematic"]["mean"]:.2f}</td><td>{accuracy["scores"][h]["physical"]["mean"]:.2f}</td></tr>' for h in hs)
    videos=''.join(f'<h3>{name}</h3><video controls preload="metadata" poster="{name}/poster.jpg" src="{name}/results.mp4"></video><p><a href="{name}/summary.json">完整性能</a> · <a href="{name}/protocol.json">源码与运行参数</a></p>' for name in final)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>五圆物理主线与30FPS小试</title><style>body{{font:16px/1.7 system-ui,Microsoft YaHei;max-width:1120px;margin:40px auto;padding:0 20px;color:#203040;background:#f8fafc}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #ccd5df;padding:8px;text-align:left}}video,img{{width:100%;max-height:720px}}.note{{padding:16px;background:#fff1d6;border-left:5px solid #e4a22a}}code{{background:#e7edf4}}</style>
    <h1>五圆物理主线：移除实时传统预测，向30FPS推进</h1>
    <p>本轮按用户要求，把五圆/可见性切换、桨叶周期和总扰动作为主线。实时快入口默认只运行物理预测；传统方法在保存输出后离线计算对照。最小位置/速度滤波用于力学积分初始状态，仍然保留。原v7源代码与结果未删除。</p>
    <div class="note"><b>结果：接近30FPS，尚未稳定达标；物理预测精度仍未优于对照。</b> 两个最终小段共150次源更新、141次当前图像观测、121次合格力学代理输入。处理能力约28FPS，真实新观测约25–26.5FPS；计入缺失观测和积压后不能声称30FPS实时。速度测试不含视频编码，下面视频为事后按相同原始事件渲染。</div>
    <h2>实际改动</h2><ol><li>从默认快循环移除传统加速度拟合、尺度外推和多套预测输出；对照只在离线分析执行。</li><li>ROI原生ON/OFF阈值、合格残弧反推桨心、五圆四/三桨及退化轴线。整机扁长不能抢先否决当前四桨。<code>axis_only</code>表示仅得到轴、可见数量未知，不能称作已识别二桨。</li><li>四桨使用当前频率平方均值；部分可见/整机池化均标明代表转速假设。不是测得的真实RPM/牛顿推力。</li><li>力学为<code>a=k*u_current+d_slow</code>；到期位置残差慢校准参数，当前合格姿态/周期立即使用。输入激励不足冻结增益。默认200是像素加速度先验。</li><li>NN异步10Hz，当前事件图KLT对齐迟到NN；30Hz尝试更新结构/物理状态。原生SDK ROI过滤、时间后缀零拷贝、固定像素周期采样、双共享槽独立RAW解码；不跳源帧、不降低所选像素的时间分辨率。</li></ol>
    <h2>最终实际性能</h2><table><tr><th>片段</th><th>源帧数</th><th>平均计算能力 FPS</th><th>实际新观测 Hz</th><th>服务P95 ms</th><th>力学输入</th><th>结构状态</th></tr>{''.join(body)}</table>
    <p>计算能力为1000/平均服务时间，服务含消费者解码等待、计算、日志；独立解码耗时不重复相加。原始定位/SDK初始化另外记录。完成滞后仍有数百毫秒，截止时刻33.3ms内比例不足。不能仅用中位数或源切片率宣布达标。</p>
    <img src="results.png" alt="耗时、增益和误差图">
    <h2>精度与科学边界</h2><p>侧视110–113s，先保存预测再读已暴露的人工框中心参考；每时域14个5Hz共同起点。当前p/v固定，传统加速度/尺度仅离线重建；不是完整v7光流系统对比，不是盲测，框中心不是质心/姿态/推力真值。</p>
    <table><tr><th>时域</th><th>CV均误差 px</th><th>离线传统均误差 px</th><th>物理均误差 px</th></tr>{score}</table>
    <p>因此本轮证明的是用户主线已独立接通、计算路径已精简；尚未证明力学先兆能提高真实轨迹预测。清楚片段60时刻仅16次确认四桨，快前端仍比历史完整前端容易失锁。三桨补全代码存在，但最终150时刻没有成功的三桨状态，不能声称本批三桨验证成功。</p>
    <h2>现在卡在哪里，下一步做什么</h2><ol><li><b>结构资格与几何耗时：</b>当前几何每帧约十几毫秒且存在尖峰。下一步把完整五圆/残弧建模放入低频后台，前台基于同一刚体模板做当前姿态更新；保留当前证据检查，不能把模板推点标成观测。不要恢复传统并行预测。</li><li><b>可靠增益：</b>侧视增益仅4次更新到约794，清楚段没有合格更新、仍为默认200；不是学出了推力系数。需要有足够姿态/频率变化的段落，检查频率谐波/区域污染以及图像投影与总扰动的耦合。优先比较输入增量能否解释随后位移，而非继续调最终融合权重。</li><li><b>稳态与启动：</b>队列仍会积压；要以新观测率和完成滞后判断30FPS。先解决几何重复计算与尖峰，再决定硬件。</li></ol>
    <p>受控方程实验：悬停不胡乱辨识增益，激励轨迹真值600时最终{controlled['changing_gain_final']:.2f}，当前响应在变向后{controlled['current_response_after_onset_s']:.3f}s出现。这只是已知输入/受控位置的机制检查，不是事件感知精度证明。</p>
    <h2>查看预测</h2>{videos}
    <h2>复用来源</h2><p>主线参考<a href="https://arxiv.org/abs/2604.18289">原始桨叶感知状态估计论文</a>。本轮另核对<a href="https://github.com/radimspetlik/effqpdet">effqpdet官方C++仓库</a>的异步突发、频率及椭圆思路；依赖与绝对路径改造较多，未声称已编译作者项目。本轮实际用的是本机Metavision SDK以及已有几何/周期代码。</p>
    <details><summary>全部有结果版本与失败保留</summary><table><tr><th>版本（开发顺序，非严格单因素试验）</th><th>能力FPS</th><th>当前图像观测</th></tr>{experiments}</table><p>首次SDK33333µs起点未对齐失败保留；<code>side_physical_final</code>暴露实验协方差复用对line条件为空的错误，失败保留、已修复，该近似开关默认关闭。线程限制与协方差复用没有可靠整链收益，不单列为达标证据。离线评分初次KF一致性断言发现弱置信度处理差异，后改为固定实际p/v再比较加速度，未重写在线预测。</p></details>
    <p><a href="batch_summary.json">机器可读汇总</a> · <a href="artifact_checks.json">源码、时序、基线核验</a></p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    save(OUT/'artifact_checks.json',dict(final=audit,original_position_core_unchanged=baseline_files,syntax_passed=True))
    save(OUT/'batch_summary.json',dict(final=final,accuracy=accuracy,controlled=controlled,
        failures=['SDK start alignment','experimental line condition None (fixed, option off)','offline control weak-confidence mismatch (p/v now held fixed)'],
        research_complete=False,stable_30fps=False,force_advantage_proven=False,real_time_traditional_branch=False,current_jobs=[]))
    print(json.dumps(dict(report=str(OUT/'index.html'),baseline_files=len(baseline_files),checks=True),ensure_ascii=False))


if __name__=='__main__':main()
