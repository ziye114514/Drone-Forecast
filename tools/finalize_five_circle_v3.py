"""Render actual structure overlays and a bounded, provenance-aware report."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib,html,json,sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/five_circle_v3_20261026'
sys.path.insert(0,str(ROOT))
from five_circle_model import diagonal_center


def load(p):return json.loads(p.read_text('utf-8-sig'))


def circle(image,point,color,inferred=False):
    p=tuple(np.rint(point).astype(int))
    if inferred:
        for angle in range(0,360,60):cv2.ellipse(image,p,(7,7),0,angle,angle+30,color,2)
    else:cv2.circle(image,p,7,color,2)


def overlay(case):
    run=OUT/('raw_'+case)
    rows=[json.loads(l) for l in (run/'predictions.jsonl').read_text('utf-8').splitlines()]
    cap=cv2.VideoCapture(str(run/'predictions.mp4'));writer=None;count=0;posters={}
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert count<len(rows)
            row=rows[count];g=row['rotor_branch']['five_circle_v3']['selected']
            # Preserve recorded event pixels and existing comparison overlays.
            # Purple adds the new structure; right panel gets an explicit legend.
            h,w=frame.shape[:2];frame[:,1280:]=(245,245,245)
            current=g['current_geometric_center'];hubs=g['model_centers'];color=(210,40,210)
            inferred=set(g['inferred_model_ids'])
            if hubs is not None:
                centre=np.asarray(current) if current is not None else diagonal_center(hubs)
                if centre is not None:
                    for j,hub in enumerate(hubs):
                        predicted=current is None or j in inferred
                        col=(150,150,150) if predicted else color
                        cv2.line(frame,tuple(np.rint(centre).astype(int)),tuple(np.rint(hub).astype(int)),col,1)
                        circle(frame,hub,col,predicted)
                    circle(frame,centre,color if current is not None else (150,150,150),current is None)
            for p in g['observed_centers']:cv2.circle(frame,tuple(np.rint(p).astype(int)),2,(0,220,255),-1)
            if g['mode']=='edge_on' and len(g['observed_centers'])==2:
                p=np.asarray(g['observed_centers']);c=p.mean(0);n=np.asarray(g['axis_xy'])
                cv2.line(frame,tuple(p[0].astype(int)),tuple(p[1].astype(int)),(255,220,0),2)
                for sign in (-1,1):cv2.arrowedLine(frame,tuple(c.astype(int)),tuple((c+sign*45*n).astype(int)),(255,220,0),2)
            lines=['FIVE CIRCLE: MECHANISM3',f'RAW time: {row["t"]:.2f}s',
                'Mode: '+g['mode'],f'Current groups: {g["observed_count"]}',
                'Purple: current structural model','Gray dashed: inferred / predicted',
                'Yellow points: current fitted hubs','Cyan normals: UNSIGNED axis',
                'Existing green/orange: old controls','5ms display;20ms detector input',
                'Playback50fps; OFFLINE research','No Newton force / RPM truth']
            for j,line in enumerate(lines):cv2.putText(frame,line,(1290,26+j*26),0,.47,(35,35,35),1,cv2.LINE_AA)
            if writer is None:
                writer=cv2.VideoWriter(str(OUT/(case+'_structure.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w,h))
                assert writer.isOpened()
            writer.write(frame)
            if g['mode'] in ('four','three','edge_on') and g['mode'] not in posters:
                name=case+'_'+g['mode']+'.png';cv2.imwrite(str(OUT/name),frame);posters[g['mode']]=name
            count+=1
    finally:
        cap.release()
        if writer is not None:writer.release()
    assert count==len(rows)
    return dict(frames=count,posters=posters,video=case+'_structure.mp4',
        input_available=sum(r['rotor_branch']['five_circle_v3']['assumed_thrust_motion']['diagnostic']['input_available'] for r in rows))


def main():
    snap=OUT/'source_snapshot_final';snap.mkdir(exist_ok=False)
    source_names=['five_circle_model.py','lumped_disturbance.py','structure_v3.py','run_five_circle_v3.py','run_optimized_forecast.py','forecast_mainline.py']
    for name in source_names:(snap/name).write_bytes((ROOT/name).read_bytes())
    checks={};raw={};media={}
    for case in ['clear','side','edge']:
        run=OUT/('raw_'+case);protocol=load(run/'protocol.json');completion=load(run/'completion.json')
        assert completion['sources_unchanged'] and completion['raw_only'] and not completion['labels_read'] and not completion['output_caches_read']
        expected=protocol['structural_extension']['sources'];matching=[]
        for name in ['source_snapshot_initial','source_snapshot_final']:
            if all(hashlib.sha256((OUT/name/f).read_bytes()).hexdigest()==sha for f,sha in expected.items()):matching.append(name)
        assert matching,case
        checks[case]=dict(source_snapshots=matching,frames=completion['frames'],
            raw_only=True,labels_entered_inference=False,selected_mechanism=protocol['structural_extension']['selected_mechanism'])
        raw[case]=load(OUT/('raw_'+case+'_analysis')/'summary.json');media[case]=overlay(case)
    controls=load(OUT/'controls/controls.json');public=load(OUT/'neurobem_module/comparison.json')
    side=load(OUT/'cached_side_refined/summary.json');train=load(OUT/'cached_train2_refined/summary.json')
    result=dict(selected_research_mechanism=3,production_position_reference='accumulated_v7_flow_acceleration',
        actual_RAW_ticks=sum(c['frames'] for c in checks.values()),checks=checks,media=media,
        independent_rotor_pose_force_truth=False,real_time_achieved=False,
        raw_results=raw,cached_side=side,cached_train2=train,public_module=public,
        original_failed_full_state_observer_retained=True,project_complete=False,current_jobs=[])
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=['rigid_four_only','visibility_switching']
    vals=[[raw[c]['states']['2'].get('four',0),raw[c]['states']['3'].get('four',0)] for c in ['clear','side','edge']]
    x=np.arange(3)
    for j,name in enumerate(names):axes[0].bar(x+(j-.5)*.3,[v[j] for v in vals],width=.3,label=name)
    axes[0].set(xticks=x,xticklabels=['clear100','side150','edge80'],ylabel='qualified four-model ticks',title='Actual RAW: hypotheses, not accuracy')
    metric=side['scores']['common04s'];keys=['optimized_geometry','structure2_disturbance','structure3_disturbance','structure3_delayed_disturbance']
    axes[1].bar(range(4),[metric[k]['mean'] for k in keys]);axes[1].set(xticks=range(4),xticklabels=['old baseline','rigid+new KF','switch+new KF','switch+delayed'],ylabel='0.4s endpoint error [px]',title='Fixed cached side:302common origins')
    axes[0].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'mechanism_comparison.png',dpi=140);plt.close(fig)
    rows=''.join(f'<tr><td>{label}</td><td>{raw[c]["ticks"]}</td><td>{raw[c]["states"]["3"].get("four",0)}</td><td>{raw[c]["states"]["3"].get("three",0)}</td><td>{raw[c]["states"]["3"].get("edge_on",0)}</td><td>{media[c]["input_available"]}</td></tr>' for c,label in [('clear','26–28s清晰'),('side','110–113s侧视/恢复'),('edge','117.4–119s定向近侧视')])
    videos=''.join(f'<h3>{label}</h3><video controls preload="metadata" src="{media[c]["video"]}"></video>' for c,label in [('clear','清晰与三桨补全'),('side','侧视与复锁'),('edge','两组近成线、恢复与复锁')])
    images=''.join(f'<figure><img src="{v}" alt="实际机制3叠加"><figcaption>{c} / {k}；仅为算法模式，未有姿态真值。</figcaption></figure>' for c in ['clear','edge'] for k,v in media[c]['posters'].items() if k in ['three','edge_on'])
    score_rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["mean"]:.3f}</td><td>{v["p95"]:.3f}</td></tr>' for k,v in metric.items() if k in keys+['structure3_assumed_thrust_disturbance','structure3_delayed_no_input'])
    p_rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["n"]}</td><td>{v["mean_m"]:.4f}</td></tr>' for k,v in public['scores'].items())
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>五圆机制3与总扰动模型小试验</title><style>body{{font:17px system-ui,sans-serif;line-height:1.75;max-width:1120px;margin:32px auto;padding:0 20px;color:#172433}}h1,h2,h3{{line-height:1.4}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #ccd4df;padding:8px;text-align:left}}img,video{{width:100%;height:auto}}.note{{padding:16px;background:#eef3fa;border-left:4px solid #5077aa}}code{{background:#eef1f5;padding:2px 5px}}a{{color:#24599a}}figure{{margin:20px 0}}</style>
    <h1>五圆机制3：结构切换、退化、复锁与总扰动</h1>
    <p class="note">按用户选择，研究机制默认3：五圆结构＋可观测性切换。稳定位置基线v7保留。此处是独立离线结构研究入口，并非完整v7低延迟模式。没有证明真实推力或优于前沿。</p>
    <h2>这次改了什么</h2><p>复用146/147的阈值和残弧、148/149的关联与惯性，新增共享五圆状态。四桨支持建模，三桨仅补近期模型的缺点，两组需细长/方向一致才能提供无向轴线；0/1组或歧义回退。恢复需两次相容观测，补点和惯性点不进入实测计数或频率。</p>
    <p>主入口：<code>forecast_mainline.py structure</code>，默认<code>--strategy 3</code>；1/2保留对照。完整步骤见源码内<code>docs/FIVE_CIRCLE_V3.md</code>。</p>
    <h2>真实RAW小测试</h2><table><tr><th>窗口</th><th>时刻</th><th>机制3四桨锁定</th><th>三桨补全</th><th>近侧视轴线</th><th>条件推力代理输入</th></tr>{rows}</table>
    <p>共330时刻；全部当前RAW推理，无标签或保存检测输入，源未变。表中是合格假设次数，不是桨心正确率。117.4秒窗口按既有观测中的近侧视现象选取，不是盲测。研究处理约190–218ms/时刻，视频50fps仅是播放时间轴。</p>
    <img src="mechanism_comparison.png" alt="模型与误差对照">{images}{videos}
    <h2>未知力可以合并，但不能误解成已经测得风力</h2>
    <p>米制模型：<b>p̈ = (T/m)n + d</b>，<b>d = g + (Fₐₑᵣₒ + Fₒₜₕₑᵣ)/m</b>。阻力方向取决于相对空气速度v−w，并非总与地面运动相反。风和阻力可能属于同一气动作用，合并可避免重复计入。世界坐标下g已知；当前二维未知尺度模型可暂将其投影并入d。</p>
    <p>自采图像中的d单位是px/s²，包含尺度/投影、推力增益误差及模型残差。质量、推力系数和风速不能同时由一个自由扰动唯一辨识。默认200仅是有效图像增益假设，不是已知牛顿推力。总扰动观察器是已有方法，本项目不主张首次提出。</p>
    <h2>实际发现的失败与改进</h2><p>初版扩展状态KF重估p/v/d后，侧视误差明显高于原位置通道；机制3虽优于这里的固定刚体机制2，仍不能据此说预测已改善。已保留失败源码和数据。随后复用149已到期残差更新，保留原位置速度，只对剩余图像加速度做有界修正；推力输入对照仍未显示独立有效增益。</p>
    <table><tr><th>固定缓存侧视，同302起点</th><th>均误差px</th><th>P95px</th></tr>{score_rows}</table>
    <p>这些是源时刻+0.4秒的离线终点误差，不是新低频实际发布成绩；不能与v7的16起点55.23px混比。三号机制是用户选定研究方向，不代表其物理分支已胜过原位置基线。</p>
    <h2>网上数据集与已跑的小模块测试</h2>
    <table><tr><th>数据集</th><th>适合验证</th><th>关键限制</th></tr>
    <tr><td><a href="https://github.com/ZJU-FAST-Lab/VID-Dataset">VID</a></td><td>17/18段有拉绳扰动、力传感器、转速及Vicon</td><td>没有外部桨盘事件画面；此轮未下载rosbag</td></tr>
    <tr><td><a href="https://rpg.ifi.uzh.ch/NeuroBEM.html">NeuroBEM</a></td><td>400Hz轨迹、姿态、逐电机转速及气动残差</td><td>不是独立风力真值，也不是事件检测数据</td></tr>
    <tr><td><a href="https://arxiv.org/abs/1810.01987">Blackbird</a></td><td>机动轨迹与动力学模块</td><td>相机是机载渲染视角；无外部桨盘事件图</td></tr></table>
    <p>实际从NeuroBEM官方ZIP按文件名字典序取首个CSV，HTTP Range只传435901bytes，解压791846bytes；1008行、2.5175秒。前0.8秒用参考IMU和转速校准有效增益，后续用当前姿态/转速和带固定5mm噪声的位置，预测未来0.4秒。没有未来命令或状态输入。</p>
    <table><tr><th>公开日志模块对照</th><th>共同目标</th><th>均误差m</th></tr>{p_rows}</table>
    <img src="neurobem_module/comparison.png" alt="公开日志模块误差"><p class="note">只有7个高度相关目标，且使用参考姿态/转速、初始校准IMU；它验证“输入推力＋总扰动”在子模块中可运行，不证明仅用目标事件RAW已获得相同能力。不把0.0666米与自采55像素比较。</p>
    <h2>接下来应解决</h2><p>优先在相同自采连续片段上核验三桨补全和近侧视复锁是否正确，再把这些资格接入v7的低频位置通道。动力学先用VID17/18的独立外力真值验证总扰动，再替换成事件估计的姿态/周期输入，观察误差如何传递。不要继续让一个自由扰动同时解释所有检测跳变。</p>
    <p><a href="summary.json">完整结果</a> · <a href="raw_edge_analysis/summary.json">最后实际RAW对照</a> · <a href="neurobem_module/comparison.json">公开模块结果/假设</a> · <a href="neurobem_input/provenance.json">下载来源/哈希</a> · <a href="controls/controls.json">几何切换/已知扰动控制</a></p>
    <p>参考：<a href="https://www.roboticsproceedings.org/rss15/p82.pdf">VIMO</a> · <a href="https://github.com/ZJU-FAST-Lab/VID-Fusion">VID-Fusion</a> · <a href="https://arxiv.org/abs/2610.31384">CoHelix</a> · <a href="https://www.scitepress.org/PublishedPapers/2014/49060/">2014四/三/二桨姿态估计</a>。本项目并非这些论文的官方代码或完整复现。</p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    note=f'''\n\n### 用户选定五圆机制3与总扰动（2026-10-26）\n\n本次用户授权恢复结构方向，版本3明确指“结构＋可观测性切换＋退化＋复锁”，不是检测器accumulated_v3_consistency。新增five_circle_model.py、structure_v3.py、lumped_disturbance.py及run_five_circle_v3.py，统一入口forecast_mainline.py structure；稳定v7仍保留，新入口是144/154离线研究链，不冒称完整v7实时。复用146/147/148/149几何、关联、延迟残差及固定FilterPy。\n\n真实RAW clear26–28s100、side110–113s150、edge117.4–119s80，共330；帧核对、源不变、无标签/输出缓存进推理。初版两个RAW源码保存source_snapshot_initial，修正后的edge对应source_snapshot_final。三号机制在edge80时刻有four10、three2、edge_on16；均是假设计数，无桨心/姿态真值。初版整体p/v/d观察器明显更差，失败保留；改用149到期残差、保留原p/v。缓存side302共同起点：旧43.613632、机制2+新KF88.020501、机制3+新KF84.115282、延迟修正43.701542；输入/无输入修正相同，不能称推力增益。最后edge71共同起点旧39.151514→延迟39.150751，仅数值级变化，不能作有效提升证明。\n\n力学定义pddot=(T/m)n+d，d含g、相对空气速度相关气动及未建模项。图像d是px/s²，含投影/增益/观测误差，不能叫真实风力。固定默认增益，不同时自由辨识质量/推力系数和任意扰动。VIMO/VID-Fusion/ESO已有总扰动研究，不称首创。\n\n公共数据：VID17/18有独立拉绳力真值（未下载rosbag）；NeuroBEM官方CSV按字典序首文件事先选择，通过Range仅下载435901bytes，解压791846bytes，1008行2.5175s。0.8s参考IMU/RPM校准后，七个共同0.4s目标：同状态CV0.252746m、无推力扩展状态0.308392m、参考推力+总扰动0.066550m。使用参考姿态/RPM与5mm合成位置噪声，非事件感知/独立力真值/泛化证据，数据留output不打包公开。\n\n报告output/five_circle_v3_20261026/index.html；视频与模型叠加已生成。实际会话85021、76500、92589和下载79805均exit0，无后台任务。下一步优先独立核验三桨/侧视复锁，再将资格接v7低频通道；物理观察器需VID外力真值及事件输入替换对照。\n'''
    with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:f.write(note)
    p=WORK/'output/checkpoints/latest_handoff.json';handoff=load(p)
    handoff.update(utc=datetime.now(timezone.utc).isoformat(),execution_status='User resumed; mechanism3 bounded implementation and pilots completed; no running jobs',
        research_objective_status='incomplete',selected_research_mechanism=3,
        latest_structure_report=str(OUT/'index.html'),latest_report=str(OUT/'index.html'),
        latest_live_revision='accumulated_v7_flow_acceleration',current_jobs=[],experiments_running=False,
        quota_pause_requested=False,quota_remaining=dict(short=85,weekly=98),
        next_action='User chose structure mechanism3. New entry forecast_mainline.py structure; v7 live remains baseline, research entry is not full v7 runtime. Check three-hub inferred geometry and edge relock against independent visible evidence, then connect qualified structure to low-frequency v7. VIMO/VID total-disturbance idea established. VID17/18 independent force data preferred; public NeuroBEM test uses reference inputs, not event-only physical recovery. Preserve failed full-state observer. No background jobs. Keep10% stop policy.')
    handoff['quota_policy'].update(last_observed_primary_used_percent=15,last_observed_primary_remaining_percent=85,last_observed_weekly_used_percent=2,resume_requires_scheduled_check=False)
    p.write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),actual_RAW_ticks=330,checks=checks,media=media)),flush=True)


if __name__=='__main__':main()
