"""Save existing evidence and a usable handoff without changing goal status."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=PROJECT/'output/native_plane165_20261104'


def marked(path,key,text):
    source=path.read_text('utf-8');start=f'<!-- {key}:start -->';end=f'<!-- {key}:end -->'
    block=start+'\n'+text.strip()+'\n'+end
    if start in source and end in source:
        a=source.index(start);b=source.index(end,a)+len(end);source=source[:a]+block+source[b:]
    else:source+='\n\n'+block+'\n'
    path.write_text(source,'utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=0);args=parser.parse_args()
    report=OUT/'index.html';handoff=ROOT/'docs/HANDOFF_20261105_NATIVE165.md'
    assert report.exists() and handoff.exists()
    source=(ROOT/'README.md').read_text('utf-8')
    latest='''<!-- native165-latest:start -->
> **最新阶段165（2026-11-05）：** 纯RAW ON/OFF桨叶主线保留。加入相对位移相关观测、有限延迟模式、碎片重定位保护与FOV内部匹配；侧视固定源查询均值38.65→35.63px，但P95有退化。另一录像同64包时序185.61→101.22px，全67包121.64→99.59px，仍远未30px；真实发布误差更高。RAW有效输出约20–24Hz，不把缓存30FPS当实时成功。目标未完成，发行ZIP未更新。见[报告](../output/native_plane165_20261104/index.html)与[最新交接](docs/HANDOFF_20261105_NATIVE165.md)。下方条目是历史。
<!-- native165-latest:end -->'''
    begin='<!-- native165-latest:start -->';end='<!-- native165-latest:end -->'
    if begin in source:
        a=source.index(begin);b=source.index(end,a)+len(end);source=source[:a]+latest+source[b:]
    else:source=source.replace('# Event Drone Forecast','# Event Drone Forecast\n\n'+latest,1)
    (ROOT/'README.md').write_text(source,'utf-8')
    data_path=PROJECT/'output/checkpoints/latest_handoff.json';d=json.loads(data_path.read_text('utf-8'))
    d.update(utc=datetime.now(timezone.utc).isoformat(),latest_numbered_stage=165,
        current_status_document=str(handoff),latest_report=str(report),research_objective_status='incomplete',
        execution_status='Stage165 evidence saved; actual RAW, fixed-input reference repair and clear-view speed work. Goal30px/30Hz incomplete.',
        next_action='Read stage165 final handoff. Retain relative observer + explicit support-reference/border fixes and native history ownership. Attack remaining rigid rotor correspondence bias, shared-image error correlation and RAW processing long tails; do not rerun rejected weak-pixel/hybrid/Matern sweeps.')
    if args.jobs==0:d['last_completed_stage']=165;d['completed']=sorted(set(d.get('completed',[])+[164,165]))
    progress=d.setdefault('stage165_progress',{})
    progress.update(directory=str(OUT),running_jobs=args.jobs,goals_achieved=False,
        cold_relative_source_error_px=35.634695712186925,warm_relative_source_error_px=37.84419779964097,
        cold_relative_source_P95_worsened=True,
        train2_same_schedule_source_mean_px=[185.6072811282153,118.05395129969205,107.83361226969588,101.21653174785855],
        train2_all_packets_source_mean_px=[121.63534033237923,99.59414677569475],
        pending='Remaining current-reference/correspondence bias and event-period force ambiguity; fullRAW sustained30Hz and30px still unmet.',
        rejected_not_default=['shared-plane LK','uniform exposure-time shift','Matern force derivative','CPU/GPU hybrid histogram','weak regional templates with no native support gain'])
    for name in ['raw_relative_bounded','raw_train2_low_queue','raw_clear_low_queue','raw_clear_sparse_history']:
        p=OUT/name/'summary.json'
        if p.exists():progress[name]=json.loads(p.read_text('utf-8'))
    progress['retained']=['OpenCV native match preprocessing','partial-entry velocity initialization','native axis workspace',
        'NVRTC spatial kernels and persistent buffers','sparse accepted-pixel rotor rate assignment','EventCD byte window join',
        'remote dense materialization and bounded queue','previous masked rotor images owned in sparse history buffers',
        'optional relative displacement observer','optional support-reference guard and interior FOV matching']
    data_path.write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')
    table=[]
    for name in ['raw_relative_bounded','raw_train2_low_queue','raw_clear_low_queue','raw_clear_sparse_history']:
        s=progress.get(name)
        if not s:continue
        table.append(f'|{name}|{s["frames"]}|{s.get("actual_valid_output_count","见trace")}|{s["steady_pipeline_fps"]:.2f}|{s.get("actual_valid_output_rate_hz",0):.2f}|')
    final='''## 本批补齐结果与当前状态

- 另一录像同实际64包时序的源查询：185.607→118.054（碎片守卫）→107.834（显式参考更换）→101.217px（内部边缘匹配）；P95 495.608→198.326→186.642→189.973px。均值逐步改善，但不是每个统计都单调改善，结果全部保留。
- 全部67包：121.635→99.594px，P95 198.682→188.407px。原控制按实际时序重放逐帧中心最大差0px，完成了真实失效复现；这是缓存质量试验，不是RAW30Hz证明。
- 新参考更换在原7维或相对9维状态中只平移位置（及克隆位置），为共同偏移增加36px²不确定性，不改v、d、k。它是机体参考点假设，不是证明质心可见；旋转时偏移的动态仍未完整可观测。
- FOV内部匹配已实现，已知(3,-4)px合成例由旧(3,0)恢复约(3,-4.001)。必须先模糊原支持再裁内部；内部模板用零均值归一化，否则会产生人工边界偏差。默认关闭，--rotor-border-motion明确启用。
- 原生历史缓冲已改为只拥有原本参与匹配的桨叶像素，仍保留原生大小平面及掩膜；共享槽覆盖安全不撤销。4对真实输入的匹配结果与共同运动输入完全相同，复制中位约1.88–2.95→0.42–1.35ms。算子小收益，不冒充整线增益。
- 原两力集合切换、PSD、过去输入不倒用、重新出现不重复计力等机制继续通过；300→600Hz无相位确认时仍有阶跃被吞的已知局限。

|实际RAW轮次|处理包|实际有效输出|平均处理FPS|实际有效输出Hz（含续推）|
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

低队列train2轮次：67包只跳过1，37个既定30Hz查询37个预测未过期，但仅35个不同源状态；7查询实际now+0.4均值127.891px、P95 264.489px。没有把实时误差掩盖成源误差99.594px。首次四桨清晰RAW也有模式four，但速度和有效输出未达到30，详见上表；此清晰段缺充分未来人工参考，不编造位置精度。

此前“正在等待相同输入对照”“边缘匹配尚未实现”的条目已完成，以本节为准。下一步先定位清晰段CPU几何/历史复制的真实长尾，保留所有小收益；随后处理参考点、相邻共享图像噪声与局部周期来源连续性。不要再把检测到当前桨叶直接等同于机体中心测量资格。没有本批后台作业时running_jobs应为0；研究目标仍未完成。
'''
    marked(handoff,'native165-final',final)
    log='''## 2026-11-05 阶段165：真实发布时间、相关位移与碎片重定位优化

- 继续用户原生ON/OFF桨叶→五圆/部分/侧视→可测＋未知力→二维主线，用户已撤销10%停止规则；没有子代理、没有用额度券/购买额度，无新训练。未动RAW、权重、两处yolov10或公开ZIP。
- 保留原生计数/GPU空间算子/缓冲复用、远端图解包、稀疏转速分配、EventCD字节拼接和只拥有已使用桨叶像素的历史图像缓冲；前一观测不再依赖可被复用的共享槽。
- 相对位移观测器保留前后位置交叉协方差。同输入冷9查询38.648→35.635px、入画历史40.356→37.844px，P95退化也保留。源时刻开发分数，不是实际发布精度。
- 新限速真实RAW时钟暴露平均FPS可接近31而结果大多过期。有限延迟模式显式记录跳过帧，侧视约19.70Hz有效输出；train2一帧队列/滚动解码约23.59Hz，37预定查询均有未过期预测，但实际7查询127.89px。持续30有效新观测仍未达到。
- 另一录像185.61px大失效来自小桨叶碎片被重锚成机体中心，参考跳变假造速度/力。固定同64包时序逐步185.61→118.05→107.83→101.22px，P95 495.61→189.97px；完整67包121.64→99.59px。显式支撑参考更换不注入v/a，FOV内部匹配消除贴边模板不能负向移动的硬限制。最终仍很差，未称达到30px。
- 记录LK、粗搜、曝光延迟、Matérn、混合CPU直方图、弱像素区域试验等未获最终收益的机制，避免重复试同一失败。弱模板不接CLI，逐像素资格未放宽。k/d依然不可辨识，默认k=200非真实推力。
- 查到CoHelix共享投影多桨相位方案；相对观测借鉴2002 stochastic cloning/2007 SC-KF，仅自写实现，不声称方法首次创新/完整作者复现。
- 报告 output/native_plane165_20261104/index.html；最新交接 event_drone_forecast/docs/HANDOFF_20261105_NATIVE165.md；根latest_handoff与README同步。全部正式推理启动前快照＋结束哈希；标签在推理完成后评分。最后作业数见checkpoint，目标30px/30Hz未完成。
'''
    marked(PROJECT/'WORKLOG.md','native165-results',log)
    print('README, WORKLOG, real report links and checkpoint saved; running_jobs='+str(args.jobs))
