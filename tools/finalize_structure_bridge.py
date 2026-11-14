"""Finalize geometry/position coupling results with all actual revisions kept."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,html,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/five_circle_bridge_20261026'
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from summarize_live_pilot import reference_at,stats
from five_circle_model import diagonal_center


def load(p):return json.loads(p.read_text('utf-8-sig'))
def rows(run):return [json.loads(v) for v in (run/'predictions.jsonl').read_text('utf-8').splitlines()]


def preserve(run):
    protocol=load(run/'protocol.json');expected=dict(protocol['sources'])
    expected.update(protocol['structural_extension']['sources'])
    snapshot=run/'complete_source_snapshot';snapshot.mkdir(exist_ok=True)
    current=(ROOT/'core/run_optimized_forecast.py').read_text('utf-8')
    variants=[current]
    variants.append(current.replace("    paths += [ROOT/'research/stage6/silhouette.py']\n",'').replace(',image=image,prepared=prepared)',',image=image)'))
    variants.append(variants[-1].replace("    paths += [ROOT/'core/whole_body_axis.py']\n",''))
    recovery=[]
    for name,sha in expected.items():
        choices=[ROOT/name,run/'structure_source_snapshot'/name,run/'structure_source_snapshot'/Path(name).name]
        data=None
        for p in choices:
            if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha:data=p.read_bytes();break
        if data is None and name=='run_optimized_forecast.py':
            for text in variants:
                for version in [text,text.replace('\n','\r\n')]:
                    if hashlib.sha256(version.encode()).hexdigest()==sha:
                        data=version.encode();recovery.append(name)
        if data is None:raise AssertionError('Cannot recover exact frozen source: '+str(run)+' '+name)
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    result=dict(files=len(expected),all_match_protocol_hashes=True,
        reconstructed_from_reversible_own_patch_with_exact_hash=recovery)
    (run/'source_recovery_check.json').write_text(json.dumps(result,indent=2),'utf-8')
    return result


def overlay(run,output):
    data=rows(run);cap=cv2.VideoCapture(str(run/'predictions.mp4'));writer=None;count=0;posters={}
    while True:
        ok,frame=cap.read()
        if not ok:break
        row=data[count];m=row['rotor_branch']['five_circle_v3'];g=m['selected'];frame[:,1280:]=246
        hubs=g['model_centers'];centre=g['current_geometric_center']
        if hubs is not None:
            c=np.asarray(centre) if centre is not None else diagonal_center(hubs)
            if c is not None:
                for i,p in enumerate(hubs):
                    col=(210,35,210) if centre is not None and i not in g['inferred_model_ids'] else (150,150,150)
                    cv2.line(frame,tuple(np.rint(c).astype(int)),tuple(np.rint(p).astype(int)),col,1)
                    cv2.circle(frame,tuple(np.rint(p).astype(int)),6,col,2)
                cv2.circle(frame,tuple(np.rint(c).astype(int)),7,(210,35,210),2)
        if g['mode']=='edge_on':
            support=g.get('axis_support',{});c=np.asarray(support.get('center',row['body']['center']))
            tangent=np.asarray(support.get('tangent',[1.,0.]));normal=np.asarray(g['axis_xy']);span=support.get('span_px',120.)
            cv2.line(frame,tuple(np.rint(c-.5*span*tangent).astype(int)),tuple(np.rint(c+.5*span*tangent).astype(int)),(255,220,0),2)
            for sign in [-1,1]:cv2.arrowedLine(frame,tuple(np.rint(c).astype(int)),tuple(np.rint(c+sign*45*normal).astype(int)),(255,220,0),2)
        pair=m['live_position_pair'];new_prediction=pair.get('predictions',{})
        if pair['updated']:
            for name,col in [('v7_same_observations',(30,190,30)),('v7_structure',(190,30,190))]:
                if name in new_prediction:
                    point=new_prediction[name]['0.4'];cv2.circle(frame,tuple(np.rint(point).astype(int)),8,col,2)
        lines=['MECHANISM3: CURRENT EVIDENCE',f'Source t={row["t"]:.2f}s',
            'Mode: '+g['mode'],f'Regions: {g["current_region_count"]}; labelled hubs: {g["observed_count"]}',
            'Purple: current geometric centre','Cyan: unsigned image axis',
            'Green circle: v7 same-input0.4s','Purple circle: structure-CI0.4s',
            'Other old overlay marks retained','Source5Hz, not wall-clock5Hz',
            'ON/OFF only; no reconstructed G','Force/RPM remain uncalibrated']
        for j,text in enumerate(lines):cv2.putText(frame,text,(1290,26+j*26),0,.43,(30,30,30),1,cv2.LINE_AA)
        if writer is None:
            writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),50.,(frame.shape[1],frame.shape[0]));assert writer.isOpened()
        writer.write(frame)
        for label,t in [('B',111.02),('C',112.02),('F',118.02)]:
            if abs(row['t']-t)<1e-7:
                path=OUT/(output.stem+'_'+label+'.png');cv2.imwrite(str(path),frame);posters[label]=path.name
        count+=1
    cap.release()
    if writer:writer.release()
    assert count==len(data)
    return dict(frames=count,posters=posters,video=output.name)


def main():
    runs={n:OUT/n for n in ['raw_side','raw_side_final','raw_edge_final','raw_B_axis','raw_B_density']}
    sources={n:preserve(p) for n,p in runs.items()}
    for p in runs.values():
        completion=load(p/'completion.json');assert completion['sources_unchanged'] and completion['raw_only'] and not completion['labels_read'] and not completion['output_caches_read']
        assert all(r['max_event_t']<round(r['t']*1e6)+1 for r in rows(p))
    side=load(OUT/'actual_side_final_analysis/summary.json');edge=load(OUT/'actual_edge_final_analysis/summary.json')
    B=rows(runs['raw_B_density'])[0];B_old=rows(runs['raw_B_axis'])[0]
    B_prior=next(r for r in rows(runs['raw_side_final']) if abs(r['t']-111.02)<1e-7)
    labels=sorted(load(WORK/'output/dense_labels_side_110_1195/labels.json')['records'].values(),key=lambda r:r['t'])
    current={n:[] for n in ['detector','structure_CI']};deltas={n:[] for n in ['position','velocity','acceleration']}
    physics={}
    for name in ['raw_side_final','raw_edge_final']:
        data=rows(runs[name]);control_delta=[];active=0
        for r in data:
            m=r['rotor_branch']['five_circle_v3'];pair=m['live_position_pair']
            active+=any(w>0 for w in m['delayed_disturbance']['weights'].values())
            if m['delayed_disturbance']['predictions']:
                control_delta.append(float(np.linalg.norm(np.asarray(m['delayed_disturbance']['predictions']['0.4'])-m['delayed_no_input']['predictions']['0.4'])))
            if name!='raw_side_final' or not pair['updated']:continue
            a=pair['coefficients']['v7_same_observations'];b=pair['coefficients']['v7_structure']
            if a is not None and b is not None:
                for k in deltas:deltas[k].append(float(np.linalg.norm(np.asarray(b[k])-a[k])))
            if pair['geometry_used']:
                truth=reference_at(labels,r['t'])
                if truth is not None:
                    current['detector'].append(float(np.linalg.norm(np.asarray(r['body']['center'])-truth)))
                    current['structure_CI'].append(float(np.linalg.norm(np.asarray(pair['structure_body']['center'])-truth)))
        physics[name]=dict(delayed_fusion_active_ticks=int(active),max04s_difference_from_no_input_control_px=max(control_delta,default=0.))
    current={n:stats(v) for n,v in current.items()};deltas={n:dict(mean=float(np.mean(v)),maximum=float(max(v))) for n,v in deltas.items()}
    media={n:overlay(runs[n],OUT/(n+'_overlay.mp4')) for n in ['raw_side_final','raw_edge_final','raw_B_density']}
    old=load(WORK/'output/five_circle_v3_20261026/raw_edge_analysis/summary.json')
    body_axis=B['rotor_branch']['five_circle_v3']['selected'];assert body_axis['mode']=='edge_on'
    core=ROOT/'src/event_drone_forecast/position';oldcore=WORK/'output/flow_acceleration_20261026/actual_side_headless/source_snapshot/src/event_drone_forecast/position'
    assert all(p.read_bytes()==(oldcore/p.name).read_bytes() for p in core.glob('*.py'))
    result=dict(selected_mechanism=3,original_v7_reference_retained=True,
        final_sequence_RAW_ticks=230,all_new_RAW_ticks=382,sources=sources,current_position_scores=current,
        coefficient_differences=deltas,side=side,edge=edge,physics=physics,
        prior_edge_axis_ticks=old['states']['3'].get('edge_on',0),new_edge_axis_ticks=edge['modes']['edge_on'],
        B_before=B_old['rotor_branch']['five_circle_v3']['selected']['mode'],B_after=body_axis,
        B_density_native_frame_matches_prior=B['frame_sha256']==B_prior['frame_sha256'],
        latest_density_fallback_scope='B single-frame RAW test only; complete trajectory scores precede this fallback addition',
        no_real_time_claim=True,no_independent_force_or_rotor_truth=True,media=media,
        position_core7_unchanged=True,current_jobs=[])
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    axes[0].bar(['old axis','updated axis'],[result['prior_edge_axis_ticks'],result['new_edge_axis_ticks']]);axes[0].set(title='Same80-source-tick edge window',ylabel='qualified axis hypotheses')
    axes[1].bar(['detector','structure CI'],[current[n]['mean'] for n in ['detector','structure_CI']]);axes[1].set(title='Current centre:14matched ticks',ylabel='mean error [px]')
    x=np.arange(2)
    for j,(key,label) in enumerate([('v7_same_observations','v7 baseline'),('v7_structure','structure CI')]):
        axes[2].bar(x+(j-.5)*.32,[d['scores04s'][key]['mean'] for d in [side,edge]],width=.32,label=label)
    axes[2].set(xticks=x,xticklabels=['side13','edge7'],ylabel='0.4s endpoint error [px]',title='Source-time forecast; mean did not improve');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'results.png',dpi=140);plt.close(fig)
    table=''.join(f'<tr><td>{label}</td><td>{d["scores04s"]["v7_same_observations"]["n"]}</td><td>{d["scores04s"]["v7_same_observations"]["mean"]:.3f}</td><td>{d["scores04s"]["v7_structure"]["mean"]:.3f}</td></tr>' for label,d in [('110–113s',side),('117.4–119s',edge)])
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>五圆机制3优化与当前卡点</title><style>body{{max-width:1120px;margin:35px auto;padding:0 20px;font:17px system-ui;line-height:1.75;color:#18283a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd3df;padding:9px;text-align:left}}img,video{{width:100%}}.note{{padding:16px;background:#eef3fa;border-left:4px solid #4e70a0}}code{{background:#edf0f5}}</style>
    <h1>机制3继续优化：有可用性和定位收益，尚无预测精度收益</h1><p class="note">保留原v7与全部失败记录。两段最终完整RAW230时刻；首轮150及B两次单帧探针另存，总新增382时刻。研究循环离线，25Hz跟踪/5Hz发布均指源时间采样，不等于墙钟实时。</p>
    <h2>已经实际修改</h2><ol><li>三个以上当前活动区域只要共线、足够扁且方向一致，也可提供无向轴线；不再强求恰好两个桨盘。</li><li>四个当前桨心给出的几何中心与模板身份锁定分开；近期兼容模型可复用，初次/不兼容模型仍确认。</li><li>五圆中心经过同一坐标偏移与固定协方差交集接入v7预测公式，保留原速度光流与加速度链的同观测对照。</li><li>整机周期特征可在电机ID不确定时单独保留，仍标为叶片经过频率代理，不冒充四电机转速和总推力。</li><li>最后复用既有stage6多阈值ON/OFF计数、PCA/LSD，直接从整机ROI测轴线；不使用G重建。这项只做了B帧定向RAW接线。</li></ol>
    <img src="results.png"><h2>真实改变与未改变</h2><p>固定80时刻近侧视窗口，轴线假设16→39。F帧由观测不足变为轴线模式；C帧的当前四点中心已可用，模板身份仍可待确认。当前定位14共同样本均误差5.770→4.852px，参考是人工框中心，非真实重心。</p>
    <table><tr><th>片段</th><th>共同源发布目标</th><th>v7同观测0.4s均误差px</th><th>结构位置耦合px</th></tr>{table}</table><p>两段来自同一录像且是开发窗口；没有显著性或跨录像泛化结论。与旧v7的55.23px时刻口径不同，不能混比。侧视窗口中，加速度差均为0，速度差平均0.314px/s，位置差平均0.369px；结构只带来小位置调整，没有新增可靠机动先兆。</p>
    <h2>B帧直接测线</h2><p>先试原生二值支持，因不够薄而拒绝；失败保留。随后原样复用已有stage6计数密度方法，在111.02s得到约203px长轴线，密度阈值之间角度差0.78°，LSD/PCA角度差1.05°。这只是当前无向图像轴线；单帧不提供有效运动预测或真实推力。计数轴线版本与原连续片段的当前事件画面哈希一致：{result['B_density_native_frame_matches_prior']}。</p>
    <img src="{media['raw_B_density']['posters']['B']}"><video controls preload="metadata" src="{media['raw_B_density']['video']}"></video>
    <h2>现在卡在哪里</h2><ol><li><b>首次获取与桨心证据：</b>A帧人工可见三桨，但本次入口没有整机框、没有合格桨心；已有骨架无法凭空初始化。ROI入口与残弧提取仍限制结构。当前活动核心也不保证是物理转轴。</li><li><b>几何定位尚不能预测变向：</b>同帧位置改善约0.92px，0.4秒预测却略差。当前结构信息不足以提供可靠的速度/加速度先兆，且框中心标注不是真实重心。</li><li><b>周期到有效推力的映射未成立：</b>方向和整机周期的覆盖提高了，但符号、谐波、尺度与默认增益仍不确定；两段延迟融合各只有2个启用时刻，推力/无输入对照最大差分别0和0.127px。总扰动可吸收偏差，不能据此证明推力输入正确。</li><li><b>实时工程仍未接完：</b>完整研究循环约168–181ms/20ms源帧，仍是离线。五圆处理需要低频或异步进入已有v7实时发布器，不能仅修改视频FPS。</li><li><b>新增整机轴线的场景边界：</b>B帧成功不证明所有重叠/正视都不会误判；完整片段的误报与切换持续性尚未确认。</li></ol>
    <h2>下一步优先</h2><p>先把整机计数轴线放到连续侧视窗口，检查误判并低频/异步接入v7发布器；模型负责当前中心/轴线，避免由不稳定几何重复估计整套运动状态。力学部分只在周期身份、投影和激励足够时学习有效增益，与无推力残差对照；不同时自由恢复风速、质量和推力系数。</p>
    <h2>叠加视频</h2><p>紫色当前几何中心/结构预测圈，青色无向轴线，绿色v7同观测预测圈。底图保留旧研究对照标记；50fps为播放轴。</p><video controls preload="metadata" src="{media['raw_side_final']['video']}"></video><video controls preload="metadata" src="{media['raw_edge_final']['video']}"></video>
    <p><a href="summary.json">完整结果与源码快照核验</a> · <a href="actual_side_final_analysis/summary.json">侧视预测</a> · <a href="actual_edge_final_analysis/summary.json">近侧视预测</a> · <a href="controls_final/checks.json">机制控制</a> · <a href="../five_circle_v3_20261026/index.html">上一轮与公开力学数据</a></p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    note='''\n\n### 机制3可观测性修正、v7位置桥接及卡点（2026-10-26）\n\n实际改动：five_circle_model按3+共线当前区域判轴，不再只认2；当前四点中心与模板身份确认分开，保留初次/不兼容双确认。新增live_structure_bridge，用固定对称位置CI及同初始几何/框偏移接入v7尺度/光流/导数公式，源25Hz更新、5Hz预测；不是墙钟独立发布器。新增structure_activity使整机当前周期与电机ID解耦，不伪造隐藏频率或真实推力。最后复用stage6 ON/OFF计数轴线直接从整机ROI取无向轴，B单帧检查成功；二值支持首试失败保留。\n\n实际新增RAW：首侧视150、最终侧视150、近侧视80、B二值1、B密度1，共382时刻；最终完整段230。所有会话44404/65182/93243/87414/7300终止exit0，无后台任务。完整源按protocol哈希保全；旧run_optimized文件按可逆自有补丁恢复并字节哈希精确匹配，source_recovery_check明确记录。原7位置核心字节不变。\n\n同输入80时刻轴线16→39，F118.02s现为edge_on；C112.02s模板仍pending，但当前中心[732.895,620.764]已可用。A仍无NN/桨心，无法起始建模。14个当前中心共同参考：均5.770→4.852px，非COM真值。最终源5Hz同目标0.4s：侧视13个32.062222→32.501038，近侧视7个45.770557→46.012722，未改善预测，不混比旧55.23px成绩。侧视a差0、v差均.314px/s、p差均.369px。位置收益未转成机动先兆。\n\n最终近侧视推力代理可用43/80（旧18/80），但延迟融合各段只2时刻启用，与无输入对照0.4s最大差0/.126575px，不能声称物理增益。B原始二值支持不够薄；直接复用stage6多阈值min(ON,OFF)密度/PCA/LSD后得到约203px轴线、阈值角差.78°，未调原阈值；该最后回退只做单帧接线，未把之前完整轨迹分数冒充此最后版已完整验证。\n\n卡点：首次获取仍依赖NN/当前残弧；活动轮廓不等物理桨心；无可靠方向/周期到推力增益映射；自由扰动会吸收输入/观测误差；研究循环168–181ms/20ms源帧还未接独立实时发布。下一步先整机计数轴线连续段误判核验及低频/异步接v7，再在足够激励条件下估有效增益，保留无输入对照，不恢复不可辨识的独立物理量。\n\n报告output/five_circle_bridge_20261026/index.html。当前research机制3保留，稳定v7未替换。\n'''
    with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:f.write(note)
    p=WORK/'output/checkpoints/latest_handoff.json';d=load(p)
    d.update(utc=datetime.now(timezone.utc).isoformat(),execution_status='Mechanism3 visibility and position-coupling batch completed; no running jobs',
        latest_report=str(OUT/'index.html'),latest_structure_report=str(OUT/'index.html'),
        latest_live_revision='accumulated_v7_flow_acceleration',research_objective_status='incomplete',
        selected_research_mechanism=3,current_jobs=[],experiments_running=False,quota_remaining=dict(short=84,weekly=97),
        next_action='Continue mechanism3. Current centre improves~.92px but0.4s forecasts slightly worsen; preserve all variants. Final whole-body stage6 count-density fallback only verified on B single RAW frame; earlier full230-tick source-clock v7 pair scores precede this last addition. Next check continuous-clip false axes, then low-rate/async geometry in independent v7 wall publisher. A still lacks NN/initial rotor evidence. Do not tune gains to sparse labels; total disturbance has no independent force validation and input proxy ambiguity persists. No jobs running; quota stop10% remains.')
    d['quota_policy'].update(last_observed_primary_used_percent=16,last_observed_primary_remaining_percent=84,last_observed_weekly_used_percent=3)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),source_checks=sources,current=current,media=media),ensure_ascii=False))


if __name__=='__main__':main()
