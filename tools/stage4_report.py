"""Post-hoc analysis and evidence rendering. Labels never imported by inference."""
from pathlib import Path
import collections, hashlib, html, json, sys, zipfile
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/stage4_skeleton_pilot_20261016'
sys.path.insert(0,str(ROOT/'research/stage4'))
from study import ARMS, CASES, verify, save
from tracker import Skeleton

def records(case):
    return [json.loads(l) for l in (OUT/case/'trace.jsonl').open(encoding='utf-8')]

def analyze():
    labels=json.loads((ROOT.parent/'output/paper_reproduction_v030/evaluation_data/train6.json').read_text(encoding='utf-8'))['labels']
    result={}
    for case,(start,duration) in CASES.items():
        rows=records(case);times=np.array([r['t'] for r in rows]);scored=[]
        for horizon in [0.,.1,.2,.4]:
            for label in labels:
                if label['state']!='present' or not start+horizon+.02<=label['t']<=start+duration:continue
                target=np.array(label['bbox']);target=(target[:2]+target[2:])/2
                anchor=label['t']-horizon;k=np.searchsorted(times,anchor+1e-8,side='right')-1
                if k<0:continue
                row=rows[k];errors={};ages={}
                for n,s in row['forecasts'].items():
                    if not s['active']:errors[n]=None;ages[n]=None;continue
                    x=np.array(s['state']);dt=label['t']-row['t']
                    errors[n]=float(np.linalg.norm(x[:2]+dt*x[2:4]-target));ages[n]=s['support_age_s']
                scored.append(dict(label_t=label['t'],anchor_t=row['t'],horizon_s=horizon,errors_px=errors,ages_s=ages))
        table=[]
        for h in [0.,.1,.2,.4]:
            r=[v for v in scored if v['horizon_s']==h]
            for name in ARMS:
                e=[v['errors_px'][name] for v in r if v['errors_px'][name] is not None]
                common=[v for v in r if v['errors_px'][name] is not None and v['errors_px']['original_cv'] is not None]
                same_age=[v for v in common if v['ages_s'][name]<=.16+1e-9 and v['ages_s']['original_cv']<=.16+1e-9]
                mean=lambda v:float(np.mean(v)) if len(v) else None
                table.append(dict(horizon_s=h,arm=name,total=len(r),available=len(e),mean_error_px=mean(e),
                  common_n=len(common),common_error_px=mean([v['errors_px'][name] for v in common]),
                  common_baseline_error_px=mean([v['errors_px']['original_cv'] for v in common]),
                  fresh_common_n=len(same_age),fresh_common_error_px=mean([v['errors_px'][name] for v in same_age]),
                  fresh_common_baseline_error_px=mean([v['errors_px']['original_cv'] for v in same_age])))
        initialization={n:[] for n in ARMS if n!='original_cv'}
        for row in rows:
            for n in initialization:
                if row['models'][n]['mode']=='initialized':initialization[n].append(row['t'])
        rejects={n:dict(collections.Counter(r['models'][n].get('reason') for r in rows if not r['models'][n]['active'])) for n in initialization}
        model_diagnostics={}
        for n in initialization:
            active=[r['models'][n] for r in rows if r['models'][n]['active']]
            model_diagnostics[n]=dict(
              measured3=sum(v.get('matched_count')==3 and v['mode']=='rotor_affine' for v in active),
              translation2=sum(v['mode']=='rotor_translation' for v in active),
              body_frames=sum(v['mode']=='body_flow' for v in active),
              near_flat_frames=sum(v['flatness']<.12 for v in active),
              median_flatness=float(np.median([v['flatness'] for v in active])) if active else None)
        # Replaying emitted past-only observations must preserve every skeleton output.
        prefix_n=min(len(rows),180);prefix_checks={}
        for n in ['four_only','skeleton_partial']:
            tr=Skeleton(body=False,partial=n=='skeleton_partial')
            for r in rows[:prefix_n]:
                got=tr.step(r['t'],r['observation']['rotors'],None,r['observation'].get('center'))
                assert got==r['models'][n]
            prefix_checks[n]=prefix_n
        result[case]=dict(table=table,scored=scored,initializations=initialization,
                          uninitialized_reasons=rejects,model_diagnostics=model_diagnostics,
                          saved_observation_prefix_exact=prefix_checks)
    seed_file=OUT/'seed_diagnostic/trace.jsonl'
    if seed_file.exists():
        seeded=[json.loads(l) for l in seed_file.open()];original=records('side')
        paired=[]
        for n in ['skeleton_partial','skeleton_body']:
            for horizon in [0.,.4]:
                es=[];bs=[]
                for l in labels:
                    if l['state']!='present' or not 110.42<=l['t']<=114.:continue
                    r=max((v for v in seeded if v['t']<=l['t']-horizon),key=lambda v:v['t'])
                    b=max((v for v in original if v['t']<=l['t']-horizon),key=lambda v:v['t'])
                    s=r['forecasts'][n];base=b['forecasts']['original_cv']
                    if not s['active'] or not base['active']:continue
                    box=np.array(l['bbox']);target=(box[:2]+box[2:])/2
                    x=np.array(s['state']);es.append(float(np.linalg.norm(x[:2]+(l['t']-r['t'])*x[2:]-target)))
                    x=np.array(base['state']);bs.append(float(np.linalg.norm(x[:2]+(l['t']-b['t'])*x[2:]-target)))
                paired.append(dict(arm=n,horizon_s=horizon,n=len(es),error_px=float(np.mean(es)) if es else None,
                                   baseline_error_px=float(np.mean(bs)) if bs else None,
                                   manual_seed=True,independent_test=False))
        result['side']['seeded_diagnostic_paired']=paired
    save(OUT/'analysis.json',result)
    return result

def draw_model(frame,m):
    if not m['active']:return
    pts=np.array(m['points'])/2;center=np.array(m['geometric_center'])/2
    observed=set(m['observed_slots'])
    for i,p in enumerate(pts):
        color=(90,230,100) if i in observed else (60,190,250)
        # Dashed spokes for hidden points: never depicted as observations.
        for k in range(10):
            if i not in observed and k%2:continue
            a=center+(p-center)*k/10;b=center+(p-center)*(k+1)/10
            cv2.line(frame,tuple(np.int32(a)),tuple(np.int32(b)),color,1)
        cv2.circle(frame,tuple(np.int32(p)),4,color,1)
        cv2.putText(frame,str(i),tuple(np.int32(p+[5,-5])),0,.4,color,1)
    cv2.circle(frame,tuple(np.int32(center)),3,(255,170,50),-1)

def render(case):
    folder=OUT/case;rows=records(case);d=np.load(folder/'event_frames.npz');times=d['times'];images=d['images']
    rt=np.array([r['t'] for r in rows]);writer=cv2.VideoWriter(str(folder/'skeleton_future.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,840))
    assert writer.isOpened();count=0
    colors={'original_cv':(40,160,255),'skeleton_body':(235,100,235)}
    for k in range(0,len(rows),5):
        r=rows[k];j=int(np.argmin(abs(times-r['t'])));f=int(np.argmin(abs(times-(r['t']+.4))))
        if abs(times[f]-r['t']-.4)>.012:continue
        panels=[]
        for n in ['four_only','skeleton_partial','skeleton_body']:
            im=cv2.cvtColor(images[j],cv2.COLOR_GRAY2BGR);draw_model(im,r['models'][n])
            panel=np.zeros((420,640,3),np.uint8);panel[60:]=im;m=r['models'][n]
            cv2.putText(panel,f'{n} t={r["t"]:.2f}: {m["mode"]}',(8,22),0,.51,(230,230,230),1)
            text=f'observed={m.get("matched_count",0)} body={m.get("body_inliers",0)}'
            cv2.putText(panel,text+'; dashed=predicted',(8,46),0,.46,(230,230,230),1)
            panels.append(panel)
        im=cv2.cvtColor(images[f],cv2.COLOR_GRAY2BGR)
        for n,color in colors.items():
            s=r['forecasts'][n]
            if s['active']:
                x=np.array(s['state']);p=(x[:2]+.4*x[2:4])/2
                cv2.drawMarker(im,tuple(np.int32(np.round(p))),color,cv2.MARKER_CROSS,18,2)
        panel=np.zeros((420,640,3),np.uint8);panel[60:]=im
        cv2.putText(panel,f'Actual future RAW t={times[f]:.2f}; forecast +0.4s',(8,22),0,.51,(230,230,230),1)
        cv2.putText(panel,'Orange:original CV; purple:skeleton+body CV',(8,46),0,.48,(230,230,230),1)
        panels.append(panel)
        joined=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]);writer.write(joined)
        if abs(r['t']-(24.02 if case=='clear' else 110.02))<.03:
            ok,b=cv2.imencode('.png',joined);assert ok;(folder/'preview.png').write_bytes(b.tobytes())
        count+=1
    writer.release();cap=cv2.VideoCapture(str(folder/'skeleton_future.mp4'));assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==count;ok,_=cap.read();assert ok;cap.release()
    save(folder/'video_manifest.json',dict(frames=count,fps=10,horizon_s=.4,
         full_frame_no_future_crop=True,future_raw_is_display_only=True,hidden_points_dashed=True))

def charts(data):
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=False)
    for ax,(case,_) in zip(axes,CASES.items()):
        rows=records(case)
        for offset,n in enumerate(['four_only','skeleton_partial','skeleton_body']):
            t=[r['t'] for r in rows];v=[offset+(.7 if r['models'][n]['active'] else 0) for r in rows]
            ax.plot(t,v,label=n)
        ax.set_title(case+': skeleton active state (height is categorical)');ax.legend(loc='upper right',fontsize=8)
        ax.set_xlabel('Sensor time (s)');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'tracking_timeline.png',dpi=150);plt.close(fig)

def report(data):
    summaries={c:json.loads((OUT/c/'summary.json').read_text()) for c in CASES}
    fmt=lambda x:'—' if x is None else f'{x:.2f}'
    literature='''<table><tr><th>研究与代码</th><th>可借鉴的内容</th><th>与本项目的距离</th></tr>
<tr><td><a href="https://arxiv.org/html/2609.11920v1">EVPeriscope，2026</a> / <a href="https://github.com/grasp-lyrl/evperiscope">GitHub</a></td><td>已知不对称机架的四桨心，PnP＋EKF整机跟踪。</td><td>论文实际要求四个桨心可见；几何已知。不能直接证明未知机型缺桨或极端侧视有效。</td></tr>
<tr><td><a href="https://arxiv.org/abs/2508.17746">DroneKey，2025</a> / <a href="https://github.com/kkanuseobin/DroneKey">GitHub</a></td><td>整机关键点关系、编号以及适应极端姿态的训练损失。</td><td>RGB学习方法，需要迁移到事件表示；本轮未训练或运行原网络。</td></tr>
<tr><td><a href="https://arxiv.org/abs/2602.06211">DroneKey++，2026</a> / <a href="https://sites.google.com/view/cvl-jnu/publication/dkpp/dkppdataset">作者数据集</a></td><td>关键点与类别联合推断，减少手工尺寸先验。</td><td>依赖学习到的形状先验，不能把不输入尺寸理解为单目绝对尺度必然可辨。</td></tr>
<tr><td><a href="https://arxiv.org/abs/2408.03225">LOPET / TIP 2024</a> / <a href="https://github.com/Zibin6/LOPET">GitHub</a></td><td>已知线模型与真实事件匹配，鲁棒优化及部分遮挡持续跟踪。</td><td>公开的是简化MATLAB演示；并非未知无人机骨架建模，也未在本轮完整复现。</td></tr>
<tr><td><a href="https://arxiv.org/html/2603.09235v1">HelixTrack，2026</a> / <a href="https://github.com/radimspetlik/helixtrack">GitHub</a></td><td>事件相位、桨叶几何和旋转速度联合跟踪。</td><td>可为整机骨架提供局部桨叶观测；本轮没有编译其原始C++跟踪器。</td></tr></table>'''
    parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>整机骨架小试验</title><style>body{max-width:1150px;margin:30px auto;padding:0 24px;font:17px/1.75 system-ui;color:#182531;background:#f6f8fb}h1,h2{line-height:1.3}table{border-collapse:collapse;background:white;width:100%;font-size:15px}td,th{padding:9px;border:1px solid #ccd5df;text-align:left}video,img{width:100%;background:#111}code{background:#e8eef4}section{background:#fff;padding:20px;margin:20px 0}.note{border-left:5px solid #d48b26;padding:12px;background:#fff5e4}a{color:#186ac0}</style>
<h1>整机骨架：16秒 RAW 可行性小试验</h1><p>日期：2026-10-16。先固定协议和阈值，再运行两段已使用过的开发视频。本轮检验骨架测量与短时保持能力，尚未集成新的推力预测器，也没有真实姿态/转速真值。</p>
<p class="note">本轮是参考论文思路的独立小实现，不是五项论文的完整复现。骨架点是电机中心的图像位置，中心是几何中心；虚线点为预测，绝不计入观测数。骨架看起来完整不代表姿态估计正确。</p>
<h2>查到的相关工作</h2>''',literature,
'''<h2>本轮如何整合</h2><p>冻结原始 PaperPerception，以10毫秒处理RAW；每20毫秒对所有方法使用同一组局部候选。原始CV使用原目标中心；另外三组分别为只允许四点更新的骨架、允许三点仿射/两点平移的骨架，以及增加机身事件光流桥接的骨架。所有中心使用相同CV滤波参数与0.30秒发布时限。</p>
<p>骨架为归一化方形四电机的二维仿射投影，能够表达图像平面中的压扁、剪切和转动；未冒充已标定的三维刚体姿态。初始化要求连续5个符合近似平行四边形、点分离、桨尺寸相近且不近共线的四点观测。三点恰好可拟合仿射变换，零拟合残差不等于补全正确；靠历史位置限制关联。</p>
<p>机身辅助使用20毫秒衰减事件图，在当前骨架周围排除已观测桨盘，提取角点并做前后向LK检验、RANSAC相似变换。它仍可能跟踪到背景或旋转桨边缘；没有学会“识别无人机”。该桥接最多离开桨叶观测0.30秒，禁止无限自我续命。</p>
<p>原有代码已经含有短时桨叶星座和相似变换补全。本轮新增的是显式二维骨架、仿射形变、可见槽位区分及机身事件辅助；不能把“首次使用整机结构”作为贡献。</p>
<h2>实际RAW结果</h2>''']
    for case in CASES:
        s=summaries[case];d=data[case]
        parts.append(f'<section><h3>{case}：{CASES[case][0]:.0f}–{sum(CASES[case]):.0f}秒</h3><p>{s["frames"]}个发布时刻；解码{s["events"]:,}个事件（包括额外0.4秒仅供未来画面展示）。运行{s["wall_s"]:.1f}秒；此研究实现未达实时。</p>')
        parts.append('<table><tr><th>方法</th><th>有预测的时刻</th><th>自动建模时刻</th><th>三点仿射 / 两点平移 / 机身辅助帧</th></tr>')
        for n in ARMS:
            md=d['model_diagnostics'].get(n,{})
            parts.append(f'<tr><td>{n}</td><td>{s["stats"][n]["active"]}/{s["frames"]}</td><td>{html.escape(str(d["initializations"].get(n,[])))}</td><td>{md.get("measured3","—")} / {md.get("translation2","—")} / {md.get("body_frames","—")}</td></tr>')
        parts.append('</table><p>下表为未来0.4秒到已有人工整机框中心的距离。相同端点列只比较该方法与原始CV同时有输出的端点；覆盖率不同的均值不可直接作性能排名。不是电机中心或真实重心误差。</p><table><tr><th>方法</th><th>可评估 / 总端点</th><th>各自可用误差px</th><th>共同端点数</th><th>共同端点 本方法 / 原始CV px</th></tr>')
        for r in d['table']:
            if r['horizon_s']!=.4:continue
            parts.append(f'<tr><td>{r["arm"]}</td><td>{r["available"]}/{r["total"]}</td><td>{fmt(r["mean_error_px"])}</td><td>{r["common_n"]}</td><td>{fmt(r["common_error_px"])} / {fmt(r["common_baseline_error_px"])}</td></tr>')
        parts.append(f'</table><p>current、0.1秒、0.2秒、支持年龄≤0.16秒的匹配结果，以及每个评分端点，见<a href="analysis.json">完整分析JSON</a>。标签只在本分析程序读取，没有插值或输入跟踪器。</p><video controls preload="metadata" poster="{case}/preview.png" src="{case}/skeleton_future.mp4"></video><p>固定间隔完整展示该片段，包含未初始化和丢失。右下为实际未来RAW；橙色原始CV，紫色骨架＋机身CV。</p></section>')
    if (OUT/'seed_diagnostic/summary.json').exists():
        seed=json.loads((OUT/'seed_diagnostic/summary.json').read_text())
        parts.append('<h2>把“建模失败”和“建模后跟踪失败”分开</h2><p class="note">以下单独给一次人工四点初始化，属于辅助诊断，不能计入RAW自动部署成绩。四点从110.00秒当时事件图的电机轴附近人工估计，约8像素不确定度；此后没有人工点或未来帧参与跟踪。这不是独立的姿态真值。</p>')
        parts.append('<p>仅四点更新的版本110.32秒失去骨架；允许三点/两点更新的两个版本均保持到114秒测试结束，即至少4秒。它们仍可能保持错误形状，保持时长不等于姿态准确。机身辅助版本额外使用27帧光流桥接。</p>')
        parts.append('<table><tr><th>人工初始化诊断</th><th>预测间隔</th><th>共同端点</th><th>本方法 / 原始CV误差px</th></tr>')
        for r in data['side']['seeded_diagnostic_paired']:
            parts.append(f'<tr><td>{r["arm"]}</td><td>{r["horizon_s"]:.1f}秒</td><td>{r["n"]}</td><td>{fmt(r["error_px"])} / {fmt(r["baseline_error_px"])}</td></tr>')
        parts.append('</table><p>未来0.4秒在6个相同端点上有改善，但当前框中心误差略差。一次人工初始化、很少且相关的开发端点、重新建立的速度状态，均限制结论；不能称为自动系统的预测提升，也不能推出真实姿态更准确。</p><video controls preload="metadata" poster="seed_diagnostic/preview.png" src="seed_diagnostic/manual_seed_future.mp4"></video>')
    probe=OUT/'withheld_candidates.json'
    if probe.exists():
        held=json.loads(probe.read_text())
        parts.append('<h2>三点拟合为什么会“看起来很好但补错”</h2><p>在清晰片段保留的四个椭圆候选中，每次隐藏一个，用另三个拟合骨架，再与被隐藏的候选比较。三点拟合的可见残差接近零，但第四点仍有明显偏差：</p><table><tr><th>样本</th><th>帧数</th><th>隐藏点偏差中位数 / P95，px</th></tr>')
        for r in held['summary']:
            if not r['samples']:continue
            parts.append(f'<tr><td>{r["case"]}，'+('通过初始化几何条件' if r['qualified_only'] else '所有四椭圆帧')+f'</td><td>{r["frames"]}</td><td>{fmt(r["median_error_px"])} / {fmt(r["p95_error_px"])}</td></tr>')
        parts.append('</table><p>这是与原椭圆候选的一致性检查，不是电机中心真值误差。它混合了局部检测误差、椭圆中心与真实转轴投影的差异，以及仿射近似不足。不能据此断言全部是骨架算法误差；但它明确说明三点零残差无法验证第四点。</p>')
    parts.append('''<h2>当前卡点与下一步</h2><ol>
<li><b>自动建模的输入仍不可靠。</b>旧前端输出的是“像桨叶的事件块”，可能把同一个桨盘分成多个候选，也会把机臂当桨。骨架门限只能排除部分错误，不能赋予这些点正确的电机身份。优先建立整机ROI与四电机关键点观测，保留合并/遮挡标志。</li>
<li><b>三点补全依赖正确对应与模型。</b>未知编号的三点可能产生多种解释；仿射三点拟合没有多余残差来验证第四点。后续保留多个编号假设，等待新到事件验证，联合机身轮廓、桨盘相位和历史几何选择。</li>
<li><b>二维仿射骨架无法保证真实三维姿态。</b>强透视、非对称机架和电机高度差都会产生系统误差。先学习归一化机架比例，再在有真实相机内参时切换到刚体重投影优化；缺内参时明确输出图像骨架与不确定性。</li>
<li><b>真正的擦边视角会丢信息。</b>点接近共线，电机编号互换或重叠。引入机身厚度、机臂/起落架等非共面特征可增加约束；长期没有独立信息时，只能扩大不确定性，不能继续输出高置信推力方向。</li>
<li><b>机身事件也有条件。</b>静止或局部对比度不足时机身可能没有事件；光流点也可能落在背景。下一轮应检查真实机臂线段的事件支持，借鉴LOPET事件—线匹配，而不是把图中虚拟四线直接当成可见边缘。</li>
<li><b>预测还未被骨架自动解决。</b>本轮只有共享CV预测，用于检验测量改进。需等骨架身份和姿态可信后，再接回RPM/推力与在线有效参数学习；绝不使用补出的桨心作为额外真值更新推力模型。</li></ol>
<h2>几何隔离试验与证据边界</h2><p>另运行400个点级几何样本：20个随机种子、5个倾角、仿射/透视两种投影、缺1点/缺2点。电机身份已知；缺2点还给了上一时刻正确形状，因此属于有利条件下的模块诊断，不是自动检测准确率。详见<a href="synthetic.json">原始样本</a>。例如65像素半宽的机架，在89度倾角时短轴全厚度仅约2.27像素，像素级抖动就会影响分离与编号。</p>
<img src="tracking_timeline.png"><p>所有视频为真实RAW的因果事件累积。旧源代码未修改；本轮协议、参数与源码哈希见<a href="protocol.json">protocol.json</a>。既有框标签稀疏，视频已用于开发，结论不能推广为独立测试集或SOTA性能。</p></html>''')
    (OUT/'第四阶段骨架小试验.html').write_text('\n'.join(parts),encoding='utf-8')
    (OUT/'相关研究与方案.md').write_text('''# 整机骨架相关研究核查（2026-10-16）

- EVPeriscope: https://arxiv.org/html/2609.11920v1 ; https://github.com/grasp-lyrl/evperiscope 。已知不对称四旋翼几何，四桨心PnP/EKF；实际实验要求四桨可见，三点可行是理论说明。
- DroneKey: https://arxiv.org/abs/2508.17746 ; https://github.com/kkanuseobin/DroneKey 。RGB关键点关系、编号与姿态适应损失，不是事件网络直接可用。
- DroneKey++: https://arxiv.org/abs/2602.06211 ; https://sites.google.com/view/cvl-jnu/publication/dkpp/dkppdataset 。类别与形状先验、无手工尺寸输入；不等于消除了单目尺度可辨识性限制。
- LOPET / TIP 2024: https://arxiv.org/abs/2408.03225 ; https://github.com/Zibin6/LOPET 。已知线模型的事件线匹配和遮挡跟踪；公开简化MATLAB演示。
- HelixTrack: https://arxiv.org/html/2603.09235v1 ; https://github.com/radimspetlik/helixtrack 。局部桨几何/相位/RPM，可成为骨架观测因子。

整合建议：整机检测ROI→可见电机/机身关键点与线段→带多种身份假设的刚体骨架跟踪→可靠性门控→RPM/推力预测。可见点、遮挡点预测、真实机臂线观测三者必须区分。本轮只独立实现了二维仿射骨架和机身光流桥接，未运行这些项目的完整原算法；不能据此宣称创新或原论文复现完成。
''',encoding='utf-8')

def package():
    verify();p=ROOT.parent/'output/event_drone_forecast_stage4_skeleton_pilot_20261016.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in OUT.rglob('*'):
            # Completion describes the finished archive; including its previous
            # revision on a rebuild would leave a stale archive hash inside it.
            if f.is_file() and f.name!='completion.json':z.write(f,Path('evidence')/f.relative_to(OUT))
        for f in (ROOT/'research/stage4').iterdir():
            if f.is_file():z.write(f,Path('source/research/stage4')/f.name)
        for f in (ROOT/'tools').glob('stage4_*.py'):z.write(f,Path('source/tools')/f.name)
        for f in (ROOT/'src/event_drone_forecast').glob('*.py'):z.write(f,Path('source/src/event_drone_forecast')/f.name)
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
    save(OUT/'completion.json',dict(package=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        frozen_sources_unchanged=True,report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

if __name__=='__main__':
    data=analyze()
    for c in CASES:render(c)
    charts(data);report(data);package()
    for c in data:
        print(c,json.dumps([r for r in data[c]['table'] if r['horizon_s']==.4]),flush=True)
