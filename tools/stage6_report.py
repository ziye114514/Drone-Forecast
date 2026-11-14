"""Post-hoc body-axis experiment analysis, plots, actual-future videos and package."""
from pathlib import Path
import collections,hashlib,json,sys,zipfile
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage6'))
from study import OUT,CASES,save,verify

def records(c):return [json.loads(l) for l in (OUT/c/'trace.jsonl').open()]

def analyze():
    labels=json.loads((ROOT.parent/'output/paper_reproduction_v030/evaluation_data/train6.json').read_text())['labels'];data={}
    for case,(start,duration) in CASES.items():
        rows=records(case);times=np.array([r['t'] for r in rows]);names=list(rows[0]['forecasts']);scores=[]
        for horizon in [0.,.1,.2,.4]:
            for l in labels:
                if l['state']!='present' or not start+horizon+.02<=l['t']<=start+duration:continue
                k=np.searchsorted(times,l['t']-horizon+1e-9,side='right')-1
                if k<0:continue
                r=rows[k];box=np.array(l['bbox']);target=(box[:2]+box[2:])/2;errors={}
                for n,s in r['forecasts'].items():
                    if not s['active']:errors[n]=None;continue
                    x=np.array(s['state']);h=l['t']-r['t'];a=np.array(s['acceleration'])
                    errors[n]=float(np.linalg.norm(x[:2]+h*x[2:]+.5*h*h*a-target))
                scores.append(dict(label_t=l['t'],anchor_t=r['t'],horizon_s=horizon,errors_px=errors))
        mean=lambda x:float(np.mean(x)) if x else None
        table=[]
        for horizon,n in [(h,n) for h in [0.,.1,.2,.4] for n in names]:
            rr=[r for r in scores if r['horizon_s']==horizon];e=[r['errors_px'][n] for r in rr if r['errors_px'][n] is not None]
            samebody='dewarped_cv' if n.startswith('dewarped') else 'body_cv'
            common=[r for r in rr if r['errors_px'][n] is not None and r['errors_px']['original_cv'] is not None]
            bodycommon=[r for r in rr if r['errors_px'][n] is not None and r['errors_px'][samebody] is not None]
            table.append(dict(horizon_s=horizon,arm=n,total=len(rr),available=len(e),error_px=mean(e),common_n=len(common),
                common_error_px=mean([r['errors_px'][n] for r in common]),common_original_cv_error_px=mean([r['errors_px']['original_cv'] for r in common]),
                same_body_n=len(bodycommon),same_body_error_px=mean([r['errors_px'][n] for r in bodycommon]),
                same_body_cv_error_px=mean([r['errors_px'][samebody] for r in bodycommon])))
        absent={n:dict(active=0,total=0) for n in names}
        for l in labels:
            if l['state']!='absent' or not start<=l['t']<=start+duration:continue
            k=np.searchsorted(times,l['t'],side='right')-1
            if k<0:continue
            for n,s in rows[k]['forecasts'].items():absent[n]['total']+=1;absent[n]['active']+=int(s['active'])
        body_stats={}
        for n in ['body','dewarped']:
            b=[r['bodies'][n] for r in rows];angles=[]
            for q in b:
                if q.get('axis_measured'):
                    x,y=q['normal_axis_xy'];angles.append(float(np.degrees(np.arctan2(y,x))%180))
            changes=[abs((v-u+90)%180-90) for u,v in zip(angles,angles[1:])]
            body_stats[n]=dict(axis_measurements=sum(q.get('axis_measured',False) for q in b),
                updates_without_four=sum(q.get('axis_measured',False) and not q.get('four_current_observations',False) for q in b),
                median_axis_increment_deg=float(np.median(changes)) if changes else None,axis_increment_is_not_accuracy=True,
                observed_arrow_sign=False,true_force_magnitude=False)
        data[case]=dict(table=table,scores=scores,absent_label_publication=absent,body_stats=body_stats)
    synthetic=json.loads((OUT/'synthetic.json').read_text());agg=[]
    for ratio in [.02,.10,.25,.45]:
        r=[q for q in synthetic['rows'] if q['plane_ratio']==ratio];good=[q['axis_error_deg'] for q in r if q['accepted']]
        agg.append(dict(plane_ratio=ratio,total=len(r),accepted=len(good),median_error_deg=float(np.median(good)) if good else None,
                        max_error_deg=float(np.max(good)) if good else None))
    data['synthetic']=dict(cases=synthetic['cases'],aggregate=agg)
    # Standalone CLI is a native-RAW prefix, not a replay of cached model logs.
    cli=[json.loads(l) for l in (OUT/'cli_prefix/observations_forecasts.jsonl').open()];clear=records('clear')
    assert len(cli)==300
    for a,b in zip(cli,clear):
        assert a['observation']==b['observation'];assert a['four_model']==b['four_model']
        bb=dict(b['bodies']['body']);bb.pop('forecast_diagnostics',None)
        assert a['body']==bb
        for method,s in a['forecasts'].items():
            original=b['forecasts']['body_'+method];np.testing.assert_array_equal(s['position'],original['state'][:2]);np.testing.assert_array_equal(s['velocity'],original['state'][2:]);np.testing.assert_array_equal(s['acceleration'],original['acceleration'])
    data['cli_prefix']=dict(frames=300,native_raw_prefix_exact=True,independent_entrypoint=True)
    save(OUT/'analysis.json',data);return data

def render(case):
    rows=records(case);d=np.load(OUT/case/'event_frames.npz');images=d['images'];times=d['times']
    writer=cv2.VideoWriter(str(OUT/case/'body_axis_future.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,840));assert writer.isOpened();count=0
    for k in range(0,len(rows),5):
        r=rows[k];j=int(np.argmin(abs(times-r['t'])));fj=int(np.argmin(abs(times-r['t']-.4)))
        if abs(times[fj]-r['t']-.4)>.012:continue
        panels=[]
        im=cv2.cvtColor(images[j],cv2.COLOR_GRAY2BGR)
        for q in r['observation']['rotors']:
            cv2.ellipse(im,((q['cx']/2,q['cy']/2),(q['major']/2,q['minor']/2),q['angle']),(80,220,80),1)
        panel=np.zeros((420,640,3),np.uint8);panel[60:]=im
        cv2.putText(panel,f'Current RAW t={r["t"]:.2f}; local rotors={len(r["observation"]["rotors"])}',(8,22),0,.48,(235,235,235),1)
        cv2.putText(panel,'Four-point lock is initialization, not a continued requirement',(8,46),0,.43,(235,235,235),1);panels.append(panel)
        for n in ['body','dewarped']:
            b=r['bodies'][n];im=cv2.cvtColor(images[j],cv2.COLOR_GRAY2BGR)
            m=b.get('body')
            if m:
                box=np.array(m['bbox'])/2;cv2.rectangle(im,tuple(np.int32(box[:2])),tuple(np.int32(box[2:])),(190,130,40),1)
                c=np.array(m['center'])/2;axis=np.array(m['normal_axis_xy']);tangent=np.array([axis[1],-axis[0]])
                length=m['extent_px']/4
                cv2.line(im,tuple(np.int32(c-tangent*length)),tuple(np.int32(c+tangent*length)),(40,220,240),2)
                if b.get('axis_valid'):
                    cv2.line(im,tuple(np.int32(c-axis*35)),tuple(np.int32(c+axis*35)),(220,80,235),2)
            panel=np.zeros((420,640,3),np.uint8);panel[60:]=im
            cv2.putText(panel,f'{n}: {b["mode"]}; current4={b.get("four_current_observations",False)}',(8,22),0,.48,(235,235,235),1)
            cv2.putText(panel,'Yellow:body plane proxy; purple:normal AXIS +/-',(8,46),0,.46,(235,235,235),1);panels.append(panel)
        im=cv2.cvtColor(images[fj],cv2.COLOR_GRAY2BGR)
        for n,col in [('original_cv',(40,155,250)),('body_cv',(220,180,40)),('body_causal_fused',(225,80,235))]:
            s=r['forecasts'][n]
            if s['active']:
                x=np.array(s['state']);p=(x[:2]+.4*x[2:]+.08*np.array(s['acceleration']))/2
                cv2.drawMarker(im,tuple(np.int32(p)),col,cv2.MARKER_CROSS,18,2)
        panel=np.zeros((420,640,3),np.uint8);panel[60:]=im
        cv2.putText(panel,f'Actual future RAW t={times[fj]:.2f}; forecast +0.4s',(8,22),0,.48,(235,235,235),1)
        cv2.putText(panel,'Orange:original CV; cyan:body CV; purple:body fused',(8,46),0,.43,(235,235,235),1);panels.append(panel)
        frame=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]);writer.write(frame);count+=1
        target=24.02 if case=='clear' else 111.02 if case=='side_exit' else 1.02
        if abs(r['t']-target)<.03:
            ok,b=cv2.imencode('.png',frame);assert ok;(OUT/case/'preview.png').write_bytes(b.tobytes())
    writer.release();save(OUT/case/'video_manifest.json',dict(frames=count,fps=10,future_raw_display_only=True,horizon_s=.4,axis_drawn_without_measured_arrow_sign=True))

def plots():
    fig,axs=plt.subplots(2,1,figsize=(11,6))
    for ax,c in zip(axs,['clear','side_exit']):
        rr=records(c)
        for n in ['body','dewarped']:
            times=[];angles=[]
            for r in rr:
                b=r['bodies'][n]
                if b.get('axis_measured'):
                    x,y=b['normal_axis_xy'];times.append(r['t']);angles.append(np.degrees(np.arctan2(y,x))%180)
            ax.plot(times,angles,'.-',ms=2,lw=.7,label=n)
        ax.set_title(c+': observed undirected image normal (not pose ground truth)');ax.set_ylabel('Angle modulo 180 deg');ax.set_xlabel('RAW time (s)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'axis_observations.png',dpi=150);plt.close(fig)

def report(data):
    fmt=lambda x:'—' if x is None else f'{x:.2f}'
    parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第六阶段：整机侧影与推力轴</title><style>body{max-width:1160px;margin:30px auto;padding:0 22px;font:17px/1.75 system-ui;color:#182838;background:#f5f7fa}table{border-collapse:collapse;width:100%;font-size:15px;background:white}td,th{border:1px solid #ccd5de;padding:8px;text-align:left}section{background:white;margin:22px 0;padding:20px}video,img{width:100%;background:#111}a{color:#1766ad}.note{padding:15px;background:#fff0d4;border-left:5px solid #cb8d31}</style>
<h1>第六阶段：四点建模后，用整机侧影观察推力轴</h1><p>2026-10-17。目标：验证“桨叶不可识别时，机体仍可提供二维推力轴观测”。输入仍为你的RAW，不输入人工框、关键点、CAD或飞控日志。</p>
<p class="note">本轮输出的是弱/近似正交投影下的图像方向轴，法线箭头由历史/图像向上先验选择。没有真实相机标定、真实姿态或转速真值。事件支持不等于完整光学轮廓；推力大小来自过去图像运动的代理估计，不能称为测得真实推力。</p>
<h2>论文与开源核查</h2><table><tr><th>研究</th><th>与你的思路的关系</th><th>本轮实现状态</th></tr>
<tr><td><a href="https://arxiv.org/abs/2008.08823">DronePose / ECCVW 2020</a> · <a href="https://github.com/VCL3D/DronePose">GitHub</a></td><td>用整机轮廓约束无人机姿态，已有明确研究基础。</td><td>RGB、已知模型、学习与渲染；本轮未训练或运行原网络，不能直接称为事件RAW方案。</td></tr>
<tr><td><a href="https://arxiv.org/abs/2408.03225">LOPET / TIP 2024</a> · <a href="https://github.com/Zibin6/LOPET">GitHub</a></td><td>已知线框与真实事件线段匹配，部分遮挡时持续跟踪。</td><td>公开简化MATLAB演示；本轮实际调用OpenCV LSD提取机体线段，没有完整移植其6DoF优化。</td></tr>
<tr><td><a href="https://www.cvl.iis.u-tokyo.ac.jp/~kyf/ICRA2024/">Direct 3D model-based object tracking / ICRA 2024</a> · <a href="https://github.com/BluecatLi/direct_event_object_tracker">GitHub</a></td><td>从初始姿态和模型持续用事件修正，通过运动插值处理事件时间。</td><td>需要3D/外观模型及初始化，原环境ROS Noetic/Ubuntu；本轮作为方法参考，未运行原跟踪器。</td></tr>
<tr><td><a href="https://github.com/DLR-RM/3DObjectTracking">SRT3D / ICG / M3T</a></td><td>整机区域、轮廓与多模态模型跟踪，不要求每帧所有关键点都可见。</td><td>需要模型和标定；不是即用事件无人机检测器，本轮未编译。</td></tr>
<tr><td><a href="https://arxiv.org/html/1712.02402v1">四旋翼动力学与旋翼阻力</a></td><td>主推力沿机体轴；运动还受重力和阻力影响。</td><td>指导方向约束；本轮无实际推力幅值测量，未宣称恢复质量或风速。</td></tr></table>
<p>结论：整机轮廓姿态、事件模型跟踪与方向受力约束均有相关研究。未找到可直接完成“未知四旋翼RAW→一次四点建模→侧影切换→准确运动预测”的现成完整项目；这不构成创新性证明。本轮是这些思路的独立简化组合。</p>
<h2>对应项目做了什么</h2><ol><li>清晰四点连续确认后建立无头尾的等效圆盘模型，保存归一化尺度与方向先验。</li><li>用RAW的20ms衰减事件支持生成整机候选；多阈值PCA与实际LSD线段约束侧影主轴。近侧视时法线轴与主轴近似垂直。</li><li>建模后不再要求新四点观测才能更新方向；当整机太厚、线段冲突或没有观测时，记录无效/短时外推。</li><li>额外对照使用过去机体速度平移补偿事件累积；不读未来事件。它没有覆盖旋转、缩放和背景自运动。</li><li>预测对照包括相同位置/速度下CV、历史CA、方向约束CA及过去误差更新的融合。约束方法的推力幅值来自过去0.35秒图像运动，像素重力0/200/500均报告，没有用真值挑选最优值。</li></ol>
<p>实际测试按时间顺序处理0–2秒、20–30秒、108–116秒。中间未测试时段跳过，保留同一RAW的单无人机模型，重置位置/速度。因此这是跨片段模型保留与再捕获试验，不是连续116秒跟踪。箭头正反在长间隔后也没有独立保证。</p>''']
    show=['original_cv','body_cv','body_ca_history','body_axis_g200','body_causal_fused','dewarped_cv','dewarped_causal_fused']
    for c in CASES:
        summary=json.loads((OUT/c/'summary.json').read_text());a=data[c]
        parts.append(f'<section><h2>{c}：{CASES[c][0]:.0f}–{sum(CASES[c]):.0f}秒</h2><p>{summary["frames"]}个发布时刻，解码{summary["events_including_future_display"]:,}事件（含额外0.4秒仅用于未来画面展示）；多组研究实现耗时{summary["wall_s"]:.1f}秒，未达实时。</p>')
        parts.append('<table><tr><th>分支</th><th>有预测的时刻</th><th>机体方向观测</th><th>其中不依赖新四点更新</th></tr>')
        for n in ['body','dewarped']:
            st=summary['stats'][n];bs=a['body_stats'][n]
            parts.append(f'<tr><td>{n}</td><td>{st["active"]}/{summary["frames"]}</td><td>{bs["axis_measurements"]}</td><td>{bs["updates_without_four"]}</td></tr>')
        parts.append('</table><p>未来0.4秒误差到已有人工整机框中心，不是重心或姿态误差。与原位置CV只在相同端点比较；方向约束本身的效果应与同一body CV比较。</p><table><tr><th>方法</th><th>可用/总端点</th><th>各自均值px</th><th>共同端点数</th><th>共同本方法/原CV px</th><th>同body CV px</th></tr>')
        for r in a['table']:
            if r['horizon_s']!=.4 or r['arm'] not in show:continue
            parts.append(f'<tr><td>{r["arm"]}</td><td>{r["available"]}/{r["total"]}</td><td>{fmt(r["error_px"])}</td><td>{r["common_n"]}</td><td>{fmt(r["common_error_px"])} / {fmt(r["common_original_cv_error_px"])}</td><td>{fmt(r["same_body_cv_error_px"])}</td></tr>')
        parts.append('</table><p>无人机被标为未出现的时刻，仍有预测的数量（少量相关标签，只作诊断）：</p><table><tr><th>方法</th><th>仍有输出 / 未出现标签</th></tr>')
        for n in ['original_cv','body_cv','dewarped_cv']:
            r=a['absent_label_publication'][n];parts.append(f'<tr><td>{n}</td><td>{r["active"]}/{r["total"]}</td></tr>')
        parts.append(f'</table><video controls preload="metadata" poster="{c}/preview.png" src="{c}/body_axis_future.mp4"></video><p>黄色为整机平面方向代理，紫色为无独立正反标记的法线轴；右下为实际未来RAW上的预测。未建模、跟丢和离场仍完整保留。</p></section>')
    parts.append('<h2>本轮结果应如何归因</h2><p>侧视片段相同7个端点：原桨叶位置CV为170.55px，整机位置CV为83.09px，整机历史CA为61.24px，方向约束为81.32px，因果融合为67.73px。清晰片段相同10个端点：原CV40.89px，整机CV44.28px，历史CA23.60px，方向约束39.40px，融合29.84px。改善主要来自位置通道和历史运动；尚不能证明推力轴约束优于历史CA。</p><p>侧视/离场共有4个未出现标签，整机分支在1处仍发布预测：114.60秒处于coast，没有新的位置或方向观测；其余3处无输出。这个结果须与发布时限一起理解，不能直接叫作重新识别了背景。有限片段也不能保证之后不会误跟背景。</p>')
    parts.append('<h2>方向测试能证明什么</h2><p>真实RAW没有独立姿态真值，所以本轮只能证明侧影通道确实产生了新的方向观测、检验预测和未出现标签，不能宣称实际姿态准确。角度曲线平稳也可能稳定地跟错机体结构。</p><img src="axis_observations.png"><p>另做192个已知ROI的栅格圆盘侧影样本，含厚度、起落架与噪点。真方向独立于PCA/LSD生成；这是理想支持图而非完整事件相机仿真。</p><table><tr><th>真实平面短长比</th><th>接受/总数</th><th>接受样本方向误差中位数 / 最大值，度</th></tr>')
    for a in data['synthetic']['aggregate']:
        parts.append(f'<tr><td>{a["plane_ratio"]}</td><td>{a["accepted"]}/{a["total"]}</td><td>{fmt(a["median_error_deg"])} / {fmt(a["max_error_deg"])}</td></tr>')
    parts.append('''</table><h2>当前限制与下一步</h2><ol><li>等效圆盘的“整机主轴”可能被起落架、机臂或背景带偏；真实机体不是理想圆盘，近侧视判断需要更多证据。</li><li>当前整机候选仍是几何/事件活动方法，没有语义识别；保存模型后再捕获也可能误跟背景，离场结果必须一起看。</li><li>主推力轴可在桨叶不可辨时提供约束，但推力大小仍未知。运动反推的幅值与固定像素重力只是代理模型；方向更稳定不保证预测更准。</li><li>单目透视、法线符号、相机运动和尺度变化未解决。下一步优先给整机观测独立的轮廓/关键点参考，并检查侧影、桨盘及机臂之间的几何一致性，再完善力学模型。</li></ol>
<p>项目入口：run_body_axis.py，仅需--raw与--output。初次可用四点出现前，整机分支保持未建模。它是实验入口，未覆盖主程序默认预测器。</p><p>证据：<a href="protocol.json">冻结协议</a> · <a href="analysis.json">所有端点、重力假设和负例</a> · <a href="相关研究与实现状态.md">相关研究核查</a> · <a href="validation.json">验证记录</a></p></html>''')
    (OUT/'第六阶段整机侧影实验报告.html').write_text('\n'.join(parts),encoding='utf-8')

def literature():
    (OUT/'相关研究与实现状态.md').write_text('''# 整机侧影与事件推力轴研究核查（2026-10-17）

- DronePose, ECCVW 2020: https://arxiv.org/abs/2008.08823 ; https://github.com/VCL3D/DronePose 。轮廓损失与整机姿态；RGB/已知模型/学习，不是本轮原版复现。
- LOPET, TIP 2024: https://arxiv.org/abs/2408.03225 ; https://github.com/Zibin6/LOPET 。真实事件与物体线段的匹配跟踪，公开简化MATLAB演示。本轮实际调用OpenCV LSD，但未移植其整套6DoF求解。
- Direct 3D model-based object tracking with event camera by motion interpolation, ICRA 2024: https://www.cvl.iis.u-tokyo.ac.jp/~kyf/ICRA2024/ ; https://github.com/BluecatLi/direct_event_object_tracker 。初始姿态+3D模型+事件持续更新；ROS Noetic/Ubuntu，未完整运行原程序。
- SRT3D / ICG / M3T: https://github.com/DLR-RM/3DObjectTracking 。区域/轮廓模型跟踪，需要模型/标定，原代码本轮未编译。
- Faessler/Franchi/Scaramuzza, RA-L 2018: https://arxiv.org/html/1712.02402v1 。主推力沿体轴，平移还包括重力与阻力。二维方向约束不是推力幅值测量。

本轮组合：四点初始化，记住无头尾等效机架；真实RAW多阈值事件支持PCA+LSD机体线观测；近侧视时用正交方向作为图像推力轴；保留未知幅值/符号/透视不确定性。未把虚拟四条机臂当真实图像线。未找到即用的同输入、同目标完整项目不等于创新性证明。

为保持可运行，本轮独立实现二维缩减原型，未宣称以上论文完整复现。全流程不使用RGB/深度/人工初始化或人工框推理。
''',encoding='utf-8')

def validate_and_package():
    verify();old=json.loads((ROOT.parent/'output/stage3_normal_confidence_20261016/raw_v2/train6/run_manifest.json').read_text())['source_hashes']
    for name,sha in old.items():
        if name.startswith('src'):assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    videos=[]
    for c in CASES:
        for r in records(c):
            for s in r['forecasts'].values():
                if s['state'] is not None:assert np.isfinite(s['state']).all();assert np.isfinite(s['acceleration']).all()
        p=OUT/c/'body_axis_future.mp4';cap=cv2.VideoCapture(str(p));assert cap.isOpened();n=0
        while True:
            ok,f=cap.read()
            if not ok:break
            assert f.shape==(840,1280,3);n+=1
        cap.release();assert n>0;videos.append(dict(case=c,frames=n))
    save(OUT/'validation.json',dict(old_runtime_sources_unchanged=True,frozen_protocol_verified=True,native_raw_cli_prefix_frames=300,
          all_forecasts_finite=True,videos_fully_decoded=videos,no_real_pose_truth_claimed=True))
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['src','poster','href'] and '://' not in v:assert (OUT/v).is_file(),v
    Links().feed((OUT/'第六阶段整机侧影实验报告.html').read_text(encoding='utf-8'))
    dest=ROOT.parent/'output/event_drone_forecast_stage6_body_axis_20261017.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p.name!='completion.json':z.write(p,Path('evidence')/p.relative_to(OUT))
        for p in (ROOT/'research/stage6').iterdir():
            if p.is_file():z.write(p,Path('source/research/stage6')/p.name)
        for p in [ROOT/'core/run_body_axis.py',ROOT/'tools/stage6_report.py',ROOT/'research/stage4/tracker.py']:
            z.write(p,Path('source')/p.relative_to(ROOT))
        for p in (ROOT/'src/event_drone_forecast').glob('*.py'):z.write(p,Path('source/src/event_drone_forecast')/p.name)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    save(OUT/'completion.json',dict(package=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
         report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

if __name__=='__main__':
    data=analyze()
    for c in CASES:render(c)
    plots();literature();report(data);validate_and_package()
    for c in CASES:print(c,json.dumps([r for r in data[c]['table'] if r['horizon_s']==.4 and r['arm'] in ['original_cv','body_cv','body_axis_g200','body_causal_fused']]),flush=True)
