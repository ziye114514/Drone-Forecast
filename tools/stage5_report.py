"""Post-hoc analysis, report and package for the strict four-point study."""
from pathlib import Path
import collections,hashlib,html,json,sys,zipfile
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage5'))
from study import OUT,CASES,ARMS,save,verify,make_world
from projective import project,ellipse_points,ProjectiveFour

def records(case):return [json.loads(l) for l in (OUT/case/'trace.jsonl').open()]

def analyze():
    labels=json.loads((ROOT.parent/'output/paper_reproduction_v030/evaluation_data/train6.json').read_text())['labels'];data={}
    for case,(start,duration) in CASES.items():
        rows=records(case);times=np.array([r['t'] for r in rows]);scored=[]
        for h in [0.,.1,.2,.4]:
            for l in labels:
                if l['state']!='present' or not start+h+.02<=l['t']<=start+duration:continue
                k=np.searchsorted(times,l['t']-h+1e-9,side='right')-1
                if k<0:continue
                b=np.array(l['bbox']);target=(b[:2]+b[2:])/2;r=rows[k];errors={};ages={}
                for n,s in r['forecasts'].items():
                    if not s['active']:errors[n]=None;ages[n]=None;continue
                    x=np.array(s['state']);errors[n]=float(np.linalg.norm(x[:2]+(l['t']-r['t'])*x[2:]-target));ages[n]=s['support_age_s']
                scored.append(dict(label_t=l['t'],anchor_t=r['t'],horizon_s=h,errors_px=errors,ages_s=ages))
        table=[]
        mean=lambda x:float(np.mean(x)) if x else None
        for h,n in [(h,n) for h in [0.,.1,.2,.4] for n in ARMS]:
            r=[v for v in scored if v['horizon_s']==h];e=[v['errors_px'][n] for v in r if v['errors_px'][n] is not None]
            common=[v for v in r if v['errors_px'][n] is not None and v['errors_px']['original_cv'] is not None]
            table.append(dict(horizon_s=h,arm=n,total=len(r),available=len(e),mean_error_px=mean(e),common_n=len(common),
                common_error_px=mean([v['errors_px'][n] for v in common]),common_original_cv_error_px=mean([v['errors_px']['original_cv'] for v in common])))
        init={n:[r['t'] for r in rows if r['models'][n]['mode']=='initialized'] for n in ARMS[1:]}
        rejects={n:dict(collections.Counter(reason for r in rows for reason in r['models'][n].get('rejections',[]))) for n in ARMS[2:]}
        shifts={};updating={}
        for n in ARMS[2:]:
            fits=[r['models'][n]['fit'] for r in rows if r['models'][n]['mode'] in ['initialized','four_update']]
            v=[x for f in fits for x in f['center_correction_px']]
            shifts[n]=dict(qualified_updates=len(fits),median_correction_px=float(np.median(v)) if v else None,p95_correction_px=float(np.percentile(v,95)) if v else None)
            updating[n]=collections.Counter(r['models'][n].get('source') for r in rows if r['models'][n]['mode'] in ['initialized','four_update'])
        # Old frontend and old four-only arm match previous frozen pilot exactly.
        old=[json.loads(l) for l in (ROOT.parent/f'output/stage4_skeleton_pilot_20261016/{case}/trace.jsonl').open()]
        assert len(old)==len(rows)
        assert all(a['observation']==b['observation'] for a,b in zip(old,rows))
        assert all(a['models']['four_only']==b['models']['legacy_four'] for a,b in zip(old,rows))
        prefix={}
        for n in ARMS[2:]:
            tr=ProjectiveFour(joint=n!='homography_four')
            for r in rows[:180]:
                groups={'original':r['observation']['rotors']}
                if n=='joint_density_four':groups['density']=r['density_rotors']
                got=tr.step(r['t'],groups,r['observation'].get('center'))
                assert got==r['models'][n]
            prefix[n]=180
        data[case]=dict(table=table,scored=scored,initializations=init,rejections=rejects,corrections=shifts,
                       update_sources={n:dict(c) for n,c in updating.items()},
                       frontend_and_legacy_exact_stage4=True,saved_observation_prefix_exact=prefix)
    synthetic=json.loads((OUT/'synthetic.json').read_text());agg=[]
    for noise in [0.,.6]:
        rs=[r for r in synthetic['rows'] if r['noise_px']==noise];a=dict(noise_px=noise,n=len(rs),accepted=sum(r['joint_accepted'] for r in rs))
        for key in ['ellipse_center_bias_mean_px','raw_mean_center_error_px','homography_center_error_px','joint_motor_error_px','joint_center_error_px']:
            v=[r[key] for r in rs if key in r];a[key]=dict(median=float(np.median(v)),max=float(np.max(v)))
        for key in ['ippe_raw','ippe_corrected','ippe_true_centers']:
            v=[r[key]['lowest_reprojection_axis_error_deg'] for r in rs if key in r]
            a[key]=dict(median_axis_error_deg=float(np.median(v)),max_axis_error_deg=float(np.max(v)))
        agg.append(a)
    data['synthetic']=dict(cases=synthetic['cases'],aggregate=agg)
    temporal=OUT/'temporal_shape/summary.json'
    if temporal.exists():
        data['temporal_shape']=dict(summary=json.loads(temporal.read_text()),cases={})
        from stage5_temporal_shape import TemporalShape
        for c,(start,duration) in CASES.items():
            rs=[json.loads(l) for l in (temporal.parent/c/'trace.jsonl').open()];tt=np.array([r['t'] for r in rs]);orig=records(c)
            table=[]
            for h in [0.,.1,.2,.4]:
                es=[];bs=[];total=0
                for l in labels:
                    if l['state']!='present' or not start+h+.02<=l['t']<=start+duration:continue
                    total+=1;k=np.searchsorted(tt,l['t']-h+1e-9,side='right')-1
                    if k<0:continue
                    r=rs[k];base=orig[k];s=r['forecast'];b=base['forecasts']['original_cv']
                    if not s['active'] or not b['active']:continue
                    box=np.array(l['bbox']);target=(box[:2]+box[2:])/2;x=np.array(s['state'])
                    es.append(float(np.linalg.norm(x[:2]+(l['t']-r['t'])*x[2:]-target)))
                    x=np.array(b['state']);bs.append(float(np.linalg.norm(x[:2]+(l['t']-base['t'])*x[2:]-target)))
                table.append(dict(horizon_s=h,available=len(es),total=total,error_px=float(np.mean(es)) if es else None,
                                  common_original_cv_error_px=float(np.mean(bs)) if bs else None))
            tr=TemporalShape()
            for r,b in zip(rs[:180],orig[:180]):
                o=b['observation'];got=tr.step(r['t'],{'original':o['rotors']},o.get('center'))
                assert got==r['model']
            data['temporal_shape']['cases'][c]=dict(table=table,saved_observation_prefix_exact=180)
    official=OUT/'official_evperiscope/summary.json'
    if official.exists():
        o=json.loads(official.read_text());author_eval={}
        for case in ['clear','side','absent']:
            rr=[json.loads(l) for l in (official.parent/case/'trace.jsonl').open()];tt=np.array([v['t'] for v in rr]);arm_eval={}
            for name in ['full','causal_roi']:
                checks=[]
                for l in labels:
                    if not tt[0]<=l['t']<=tt[-1]:continue
                    k=np.searchsorted(tt,l['t'],side='right')-1;a=rr[k]['arms'][name]
                    c=dict(t=l['t'],present=l['state']=='present',stable=a['stable'])
                    if c['present'] and a['points_xy'] is not None:
                        pts=np.array(a['points_xy']);b=l['bbox'];c['points_inside_box']=int(np.sum((pts[:,0]>=b[0])&(pts[:,0]<=b[2])&(pts[:,1]>=b[1])&(pts[:,1]<=b[3])))
                    checks.append(c)
                arm_eval[name]=dict(label_checks=checks,warning='Inside a whole-object box is only a necessary condition, not independent four-motor truth.')
            author_eval[case]=arm_eval
        data['official']=dict(summary=o,posthoc_checks=author_eval)
    save(OUT/'analysis.json',data);return data

def render(case):
    folder=OUT/case;rows=records(case);d=np.load(folder/'event_frames.npz');it=d['times'];imgs=d['images'];writer=cv2.VideoWriter(str(folder/'four_point_future.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,840));assert writer.isOpened()
    count=0
    temporal=OUT/'temporal_shape'/case/'trace.jsonl'
    if temporal.exists():
        extra=[json.loads(l) for l in temporal.open()]
        for r,x in zip(rows,extra):r['models']['temporal_joint_four']=x['model'];r['forecasts']['temporal_joint_four']=x['forecast']
    third='temporal_joint_four' if temporal.exists() else 'joint_density_four'
    for k in range(0,len(rows),5):
        r=rows[k];j=int(np.argmin(abs(it-r['t'])));f=int(np.argmin(abs(it-r['t']-.4)))
        if abs(it[f]-r['t']-.4)>.012:continue
        panels=[]
        for n in ['legacy_four','joint_four',third]:
            m=r['models'][n];im=cv2.cvtColor(imgs[j],cv2.COLOR_GRAY2BGR)
            if m['active']:
                p=np.array(m['points'])/2;c=np.array(m['geometric_center'])/2
                observed=m.get('observed_count',4 if m['mode']!='coast' else 0)>0
                color=(80,230,90) if observed else (70,170,250)
                for v in p:
                    for z in range(10):
                        if not observed and z%2:continue
                        cv2.line(im,tuple(np.int32(c+(v-c)*z/10)),tuple(np.int32(c+(v-c)*(z+1)/10)),color,1)
                    cv2.circle(im,tuple(np.int32(v)),4,color,1)
            panel=np.zeros((420,640,3),np.uint8);panel[60:]=im
            cv2.putText(panel,f'{n} t={r["t"]:.2f}: {m["mode"]}',(8,22),0,.48,(235,235,235),1)
            cv2.putText(panel,'4 current observations required; dashed=coasting',(8,46),0,.46,(235,235,235),1);panels.append(panel)
        im=cv2.cvtColor(imgs[f],cv2.COLOR_GRAY2BGR)
        for n,col in [('original_cv',(40,155,250)),(third,(225,100,230))]:
            s=r['forecasts'][n]
            if s['active']:
                x=np.array(s['state']);cv2.drawMarker(im,tuple(np.int32((x[:2]+.4*x[2:])/2)),col,cv2.MARKER_CROSS,18,2)
        panel=np.zeros((420,640,3),np.uint8);panel[60:]=im
        cv2.putText(panel,f'Actual RAW t={it[f]:.2f}; +0.4s forecast',(8,22),0,.48,(235,235,235),1)
        cv2.putText(panel,'Orange:original CV; purple:'+third,(8,46),0,.43,(235,235,235),1);panels.append(panel)
        joined=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]);writer.write(joined);count+=1
        if abs(r['t']-(24.02 if case=='clear' else 110.02))<.03:
            ok,b=cv2.imencode('.png',joined);assert ok;(folder/'preview.png').write_bytes(b.tobytes())
    writer.release();save(folder/'video_manifest.json',dict(frames=count,fps=10,future_s=.4,future_used_for_display_only=True,third_arm=third))

def diagram():
    H,K,R,q,rs=make_world(60,5,.7,.7);true=project(H,q);centers=np.array([[r['cx'],r['cy']] for r in rs]);c=project(H,np.array([[0.,0.]]))[0]
    fig,ax=plt.subplots(figsize=(9,5))
    for i,r in enumerate(rs):
        p=ellipse_points(r,200);ax.plot(p[:,0],p[:,1],color='#6378a9',lw=1)
        ax.plot([c[0],true[i,0]],[c[1],true[i,1]],color='#98a5ba',lw=1)
        ax.annotate('',xy=true[i],xytext=centers[i],arrowprops=dict(arrowstyle='->',color='#b95c18'))
    ax.scatter(*true.T,s=50,marker='+',c='#068650',label='True projected motor shafts')
    ax.scatter(*centers.T,s=25,facecolors='none',edgecolors='#b95c18',label='Ellipse centers')
    ax.scatter(*c,s=60,c='#7644a3',marker='x',label='Projected frame center')
    ax.scatter(*centers.mean(0),s=50,c='#cf3644',marker='x',label='Mean ellipse center')
    ax.set_aspect('equal');ax.invert_yaxis();ax.legend(fontsize=9);ax.set_title('Analytic example: planar circle centers are not ellipse centers')
    ax.set_xlabel('Image x (px)');ax.set_ylabel('Image y (px)');fig.tight_layout();fig.savefig(OUT/'perspective_explanation.png',dpi=160);plt.close(fig)

def report(data):
    fmt=lambda x:'—' if x is None else f'{x:.2f}'
    parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第五阶段：四点透视建模</title><style>body{max-width:1160px;margin:30px auto;padding:0 22px;font:17px/1.75 system-ui;color:#182838;background:#f5f7fa}table{border-collapse:collapse;width:100%;font-size:15px;background:white}td,th{border:1px solid #ccd5de;padding:8px;text-align:left}section{background:white;margin:22px 0;padding:20px}video,img{width:100%;background:#111}a{color:#1766ad}.note{padding:15px;background:#fff0d4;border-left:5px solid #cb8d31}</style>
<h1>第五阶段：四个观测点、透视建模与圆心修正</h1><p>2026-10-16。沿用四点初始化路线，四点观测才更新；缺点时仅短时外推。未修改原始src或之前阶段协议。</p>
<p class="note">本轮测试的是四点检测与图像平面模型。真实RAW没有相机内参或姿态真值；不填作者另一套相机的标定来制造三维结果。预测仍使用相同CV，尚未把新观测接回推力模型。</p>
<h2>第三处问题到底是什么</h2><p>上一阶段用了二维仿射模型：平行线投影后仍平行，矩形机架会变成平行四边形。真实相机有透视，近侧与远侧缩放不同，所以强行按平行四边形拟合可能移动电机点。现在改用单应性H，允许透视四边形。</p>
<p>另一个误差来自桨盘：空间圆心的投影一般不是投影椭圆的中心。四个椭圆中心也不能简单平均成机架中心。已知正确电机投影时，中心对称矩形的机架中心应由对角线交点或H投影得到。这个效应并不能解释上一阶段的全部误差：检测错误、事件运动拖尾、非理想机架与模型失配都还在。</p>
<img src="perspective_explanation.png"><p>上图为独立解析投影示例，不是实际RAW的姿态真值。中心点投影、椭圆中心和四中心均值分别显示。</p>
<h2>找到的论文与GitHub：哪些实际跑了</h2><table><tr><th>对应困难</th><th>研究/开源</th><th>本轮使用情况</th></tr>
<tr><td>平面机架透视与双解</td><td><a href="https://github.com/tobycollins/IPPE">IPPE / IJCV 2014</a>，<a href="https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/ippe.cpp">OpenCV源码</a></td><td>实际调用OpenCV solvePnPGeneric(IPPE)，在有真实已知K的模拟中保留两个候选。真实RAW未运行标定姿态模块。</td></tr>
<tr><td>平面透视变换</td><td><a href="https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html">OpenCV单应性教程与实现</a></td><td>实际用getPerspectiveTransform建立四点图像映射；四点零重投影误差不作为正确检测证明。</td></tr>
<tr><td>椭圆中心不等于投影圆心</td><td><a href="https://arxiv.org/abs/1902.04541">Center of circle after perspective transformation，2019</a>；<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5676954/">2017圆心极点/极线标定研究</a></td><td>前者研究同心圆；本轮没有把不同桨盘当同心圆。独立实现四个共面等半径圆的联合投影拟合。</td></tr>
<tr><td>圆模型联合优化</td><td><a href="https://github.com/walkfish8/Riemannian_Circle">Riemannian_Circle</a></td><td>借鉴投影圆与椭圆联合优化方向；原仓库是已标定双目，未完整运行其优化器。</td></tr>
<tr><td>事件四桨心检测</td><td><a href="https://arxiv.org/html/2609.11920v1">EVPeriscope论文</a> / <a href="https://github.com/grasp-lyrl/evperiscope">仓库</a></td><td>主对照加入简化ROI密度候选；另外直接调用未修改官方Python检测器跑RAW，单独报告。</td></tr>
<tr><td>高效桨叶检测</td><td><a href="https://github.com/radimspetlik/effqpdet">SCIA 2025 effqpdet</a></td><td>Poisson事件突发检测值得下一轮对照；原C++依赖Metavision/Boost/yaml-cpp/HDF5/Protobuf，本轮未编译，不能算已复现。</td></tr>
<tr><td>四点身份与整机关系</td><td><a href="https://arxiv.org/abs/2508.17746">DroneKey论文</a> / <a href="https://github.com/kkanuseobin/DroneKey">仓库</a></td><td>核对了训练入口；未找到可直接调用的预训练推理说明，也尚未训练事件模型。不能把下载等同复现。</td></tr></table>
<h2>本轮实现</h2><p>先要求四个当前椭圆观测、凸四边形、足够的点间距与投影厚度。进一步要求四个椭圆能由同一个平面单应性、一个归一化矩形机架和同一桨半径解释。利用多于四点位置的椭圆形状信息，优化H、初始机架长宽比和相对桨半径；连续5次合格后冻结归一化形状，之后仅四点更新H。所有尺寸均为比例，不代表米制机架或真实质量。</p>
<p>对照包含：原中心CV、上一阶段仅四点仿射骨架、四点单应性、四圆联合修正、联合修正＋事件密度四桨候选。四点单应性与四圆联合模型共享同类几何准入条件。后两者的差异仍可能改变关联，不是完整严格的单因素实验。</p>
<p>新增密度分支借鉴EVPeriscope的20ms衰减、阈值和聚类，但只在当前/过去原检测或已确认骨架的ROI中提出四个候选，并由四圆模型验证。k=4只说明算法被要求分四类，不证明四个电机存在。这个分支不是官方算法原版。</p>''']
    for case in CASES:
        s=json.loads((OUT/case/'summary.json').read_text());a=data[case]
        parts.append(f'<section><h2>{case}：{CASES[case][0]:.0f}–{sum(CASES[case]):.0f}秒RAW</h2><p>{s["frames"]}个20ms时刻，解码{s["events_including_future_display"]:,}事件（含仅展示未来画面的0.4秒）；多组并行研究实现耗时{s["wall_s"]:.1f}秒，未达实时。</p><table><tr><th>方法</th><th>预测可用时刻</th><th>自动初始化时刻</th><th>当前四点更新 / 外推帧</th></tr>')
        for n in ARMS:
            st=s['stats'][n];updated=st.get('four_update',st.get('rotor_affine',st.get('measurement',0)))+st.get('initialized',0)
            parts.append(f'<tr><td>{n}</td><td>{st["active_forecasts"]}/{s["frames"]}</td><td>{html.escape(str(a["initializations"].get(n,[])))}</td><td>{updated} / {st.get("coast",0)}</td></tr>')
        parts.append('</table><p>未来0.4秒误差以人工整机框中心为参考，均为开发数据。不同可用端点不能直接排名；共同端点与原始中心CV配对：</p><table><tr><th>方法</th><th>端点可用/总数</th><th>各自误差px</th><th>共同端点数</th><th>共同端点本方法 / 原中心CV px</th></tr>')
        for r in a['table']:
            if r['horizon_s']!=.4:continue
            parts.append(f'<tr><td>{r["arm"]}</td><td>{r["available"]}/{r["total"]}</td><td>{fmt(r["mean_error_px"])}</td><td>{r["common_n"]}</td><td>{fmt(r["common_error_px"])} / {fmt(r["common_original_cv_error_px"])}</td></tr>')
        parts.append(f'</table><p>所有0/0.1/0.2/0.4秒端点、初始化失败原因、通过拟合后的圆心修正量在<a href="analysis.json">分析JSON</a>。缺点时只外推，虚线不代表新观测。</p><video controls preload="metadata" poster="{case}/preview.png" src="{case}/four_point_future.mp4"></video></section>')
    if 'temporal_shape' in data:
        parts.append('<h2>后续诊断：前5帧就冻结形状是否过早</h2><p>第一轮清晰片段的归一化长宽比在22.94秒估为1.22；单独拟合23/24/25/26秒四桨观测，则约为1.32/1.06/1.12/1.02。这不是真实机架变化的证据，而是观测或模型偏差尚未稳定。固定1.22会拒绝部分后续观测。诊断见<a href="shape_diagnostic.json">shape_diagnostic.json</a>。</p><p>因此另外冻结一份后续协议：只用最近1秒的合格四点观测，至少10次，缓慢更新归一化长宽比和相对半径。改变模型坐标时保持投影点和目标锚点连续。原始第一次试验保留，未修改阈值来覆盖失败。此对照重放已解码的RAW观测，无人工输入，也没有新一次原生RAW解码。</p><table><tr><th>片段</th><th>缓慢学习后可用时刻</th><th>未来0.4秒共同端点</th><th>本方法 / 原中心CV误差px</th></tr>')
        for c in CASES:
            s=data['temporal_shape']['summary'][c];r=next(v for v in data['temporal_shape']['cases'][c]['table'] if v['horizon_s']==.4)
            parts.append(f'<tr><td>{c}</td><td>{s["active_forecasts"]}/{s["frames"]}</td><td>{r["available"]}/{r["total"]}</td><td>{fmt(r["error_px"])} / {fmt(r["common_original_cv_error_px"])}</td></tr>')
        parts.append('</table><p>上方视频第三格及紫色未来位置为此后续版本。清晰片段相同7个端点的0.4秒误差为26.72px，原中心CV为18.08px，未改善预测；当前框中心误差也更大。因此该版本仅保留为诊断，不作为默认升级。覆盖率改善不能等价为预测提升；比例学习也无法创造缺失的电机观测，不声称恢复了真实物理尺寸。</p>')
    parts.append('<h2>几何测试与已知K的IPPE对照</h2><p>192个解析投影样本：4种倾角、2种距离、2种桨相对半径、2种机架比例、2种噪声与3个种子。电机身份已知、完整桨盘；这是模型隔离测试。</p><table><tr><th>轮廓噪声</th><th>合格数</th><th>椭圆圆心误差中位数px</th><th>四椭圆中心均值误差px</th><th>四点单应性中心误差px</th><th>联合模型中心误差px</th></tr>')
    for a in data['synthetic']['aggregate']:
        parts.append(f'<tr><td>{a["noise_px"]}</td><td>{a["accepted"]}/{a["n"]}</td><td>{fmt(a["ellipse_center_bias_mean_px"]["median"])}</td><td>{fmt(a["raw_mean_center_error_px"]["median"])}</td><td>{fmt(a["homography_center_error_px"]["median"])}</td><td>{fmt(a["joint_center_error_px"]["median"])}</td></tr>')
    parts.append('</table><p>注意：即使无噪声，联合优化也有局部解，最大中心误差约2.89px；0.6px轮廓噪声时最大中心误差约3.61px。不能只报中位数接近零。IPPE使用正确已知K与已知模型，保留两个姿态；真实圆心的姿态解误差接近数值精度，不能迁移为实际RAW姿态精度。</p>')
    if 'official' in data:
        parts.append('<h2>直接运行官方EVPeriscope检测器</h2><p>已下载MIT开源仓库，固定提交 <code>a2e6f2208bf12ffbe8663d2ea01d3f32f93c2bbc</code>。实际导入未修改的scripts/detect.py，5ms输入、tau=20ms，RAW时间戳转换为秒，行列坐标转为x/y。官方全图模式与“同一官方核心＋过去检测ROI限制输入”的适配模式分开；没有运行其飞控或借用其他无人机的尺寸/相机标定。</p><table><tr><th>小片段</th><th>原版全图 stable / 帧</th><th>ROI输入适配 stable / 帧</th></tr>')
        for c,s in data['official']['summary'].items():
            a=s['stats'];parts.append(f'<tr><td>{c}</td><td>{a["full"]["stable"]}/{a["full"]["frames"]}</td><td>{a["causal_roi"]["stable"]}/{a["causal_roi"]["frames"]}</td></tr>')
        parts.append('</table><p class="note">stable是作者检测器自身的标志，不是四个电机都准确识别的真值。其固定50px跟踪半径和等距稳定性判据依赖图像尺度/视角；空簇可能保留旧中心。是否落入整机框的必要条件检查见analysis.json；仍不足以验证点身份。</p>')
        for c in ['clear','side','absent']:
            parts.append(f'<p>官方检测模块：{c}</p><video controls preload="metadata" poster="official_evperiscope/{c}/preview.png" src="official_evperiscope/{c}/official_points.mp4"></video>')
    parts.append('''<h2>当前判断与下一步</h2><ol><li>第三处的几何近似已替换为透视模型，并明确修正椭圆中心与转轴投影的区别；这不等于检测前端已经给出了四个真实电机。</li>
<li>严格四圆一致性会降低覆盖率。真实候选可能是残缺桨盘、运动累积拖尾或机身边缘；过强的等圆、共面、矩形假设也会拒绝真实目标。不能把所有拒绝都归为背景误检。初始形状过早冻结的问题已做因果缓慢更新对照，但还缺独立形状真值。</li>
<li>四点单应性恰好有足够自由度拟合四个点，因此零残差不证明模型正确。下一步应以真实轮廓/事件相位验证四个桨心，不再靠“四个椭圆中心拟合得好”自我确认。</li>
<li>原版四点检测器现已接通，可作为后续真正的前端基线；固定像素半径、视角稳定性条件需要按尺度分析。更有针对性的下一步是局部运动补偿＋四桨观测可靠性，而不是继续放宽四点门限。</li>
<li>若要报告真实姿态与推力方向，应提供实际相机内参，或明确研究内参不确定性；当前单应性继续服务二维预测，不输出伪精确的三维角度。未知米制大小与质量不会阻塞二维建模。</li></ol>
<p>检查：独立解析几何、IPPE两候选、严格四点更新、合并点拒绝与超时失效；三个新模型180帧已保存观测前缀重放一致。原前端与上一阶段四点基线逐帧一致。预测有限值、视频及旧源码哈希在validation.json中记录。</p>
<p>证据：<a href="protocol.json">冻结协议</a> · <a href="analysis.json">全部分析</a> · <a href="synthetic.json">模拟原始结果</a> · <a href="相关研究与实现状态.md">相关研究与实现状态</a></p></html>''')
    (OUT/'第五阶段四点透视建模报告.html').write_text('\n'.join(parts),encoding='utf-8')

def literature():
    text='''# 四点与透视误差相关研究（2026-10-16）

用户路线：四个实际电机观测建立模型，缺点时不更新形状。所有工作服务事件RAW二维预测，不要求提前输入无人机米制尺寸/质量。

| 困难 | 论文或项目 | 本轮状态 |
|---|---|---|
| 平面单应性 | https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html | 实际调用OpenCV getPerspectiveTransform，安装版5.0.0 |
| 平面位姿两解 | Collins & Bartoli, IJCV 2014, IPPE, https://github.com/tobycollins/IPPE | 通过OpenCV solvePnPGeneric(IPPE)在已知K的模拟运行；真实RAW无K不运行姿态 |
| 投影圆心≠椭圆中心 | Wang/Chern/Alexa 2019 https://arxiv.org/abs/1902.04541 | 理论核查；原方法针对同心圆，不直接当四桨算法套用 |
| 圆心极点/极线关系 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5676954/ | 支持数学解释；本轮四共面圆联合拟合为独立实现 |
| 多圆投影优化 | https://github.com/walkfish8/Riemannian_Circle | 已标定双目项目，仅研究参考，未运行原优化器 |
| 事件四点检测+整机EKF | https://arxiv.org/html/2609.11920v1 ; https://github.com/grasp-lyrl/evperiscope | 已下载固定a2e6f2208bf12ffbe8663d2ea01d3f32f93c2bbc，原Python detect.py直接运行；飞控未运行，真实RAW姿态未运行 |
| 事件突发桨叶检测 | https://github.com/radimspetlik/effqpdet | 原C++依赖Metavision/Boost/yaml-cpp/HDF5/Protobuf，本轮未编译 |
| 电机关键点身份 | https://arxiv.org/abs/2508.17746 ; https://github.com/kkanuseobin/DroneKey | 有训练代码，未确认可即用预训练推理方案，未训练事件模型 |

本轮不是完整复现EVPeriscope：仅官方检测类被直接执行，RAW输入桥接不改变核心源码；另行报告ROI输入适配。四圆联合H优化不是IPPE、同心圆论文或Riemannian_Circle原代码的移植，不称为新颖算法。
'''
    (OUT/'相关研究与实现状态.md').write_text(text,encoding='utf-8')

def validate_and_package():
    verify();old=json.loads((ROOT.parent/'output/stage3_normal_confidence_20261016/raw_v2/train6/run_manifest.json').read_text())['source_hashes']
    for p,h in old.items():
        if p.startswith('src'):assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    for c in CASES:
        for r in records(c):
            for n,m in r['models'].items():
                if n in ARMS[2:] and m['mode'] in ['initialized','four_update']:assert m['observed_count']==4
            for s in r['forecasts'].values():
                if s['state'] is not None:assert np.isfinite(s['state']).all()
    temporal=OUT/'temporal_shape/protocol.json'
    if temporal.exists():
        protocol=json.loads(temporal.read_text())
        assert hashlib.sha256((ROOT/'tools/stage5_temporal_shape.py').read_bytes()).hexdigest()==protocol['source_sha256']
        for c in CASES:
            assert hashlib.sha256((OUT/c/'trace.jsonl').read_bytes()).hexdigest()==protocol['observations_sha256'][c]
            for line in (temporal.parent/c/'trace.jsonl').open():
                r=json.loads(line);m=r['model']
                if m['mode'] in ['initialized','four_update']:assert m['observed_count']==4
                if r['forecast']['state'] is not None:assert np.isfinite(r['forecast']['state']).all()
    author=OUT/'official_evperiscope/protocol.json'
    if author.exists():
        protocol=json.loads(author.read_text())
        assert hashlib.sha256((ROOT/'external/evperiscope/scripts/detect.py').read_bytes()).hexdigest()==protocol['source_sha256']
    videos=[]
    for p in OUT.rglob('*.mp4'):
        cap=cv2.VideoCapture(str(p));assert cap.isOpened();n=0
        while True:
            ok,f=cap.read()
            if not ok:break
            n+=1
        cap.release();assert n>0;videos.append(dict(path=str(p.relative_to(OUT)),decoded_frames=n))
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src','poster'] and '://' not in v:assert (OUT/v).is_file(),v
    Links().feed((OUT/'第五阶段四点透视建模报告.html').read_text(encoding='utf-8'))
    save(OUT/'validation.json',dict(old_runtime_sources_unchanged=True,frozen_protocol_verified=True,strict_four_updates_verified=True,
        all_forecasts_finite=True,html_local_links_exist=True,videos=videos))
    dest=ROOT.parent/'output/event_drone_forecast_stage5_four_point_20261016.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p.name!='completion.json':z.write(p,Path('evidence')/p.relative_to(OUT))
        for p in (ROOT/'research/stage5').iterdir():
            if p.is_file():z.write(p,Path('source/research/stage5')/p.name)
        z.write(ROOT/'research/stage4/tracker.py','source/research/stage4/tracker.py')
        for p in (ROOT/'tools').glob('stage5_*.py'):z.write(p,Path('source/tools')/p.name)
        for p in (ROOT/'src/event_drone_forecast').glob('*.py'):z.write(p,Path('source/src/event_drone_forecast')/p.name)
        for p in ['scripts/detect.py','scripts/epa.py','scripts/epa_filter.py','LICENSE','README.md']:
            z.write(ROOT/'external/evperiscope'/p,Path('source/external/evperiscope')/p)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    save(OUT/'completion.json',dict(package=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
         report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

if __name__=='__main__':
    data=analyze()
    for c in CASES:render(c)
    diagram();literature();report(data);validate_and_package()
    for c in CASES:print(c,json.dumps([r for r in data[c]['table'] if r['horizon_s']==.4]),flush=True)
