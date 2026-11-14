"""Measured RAW event-force chain, honest temporal extension and handoff figures."""
from pathlib import Path
import datetime,hashlib,html,json,re,sys
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage73'),str(ROOT/'research/stage71')]
import study73 as latest
import study71 as actual
OUT=latest.OUT
load=latest.load;save=latest.save;sha=latest.sha;rows=latest.rows
NAMES=dict(clear6='Clear6',side6='Side6',train3='Train3')

def table(records):
    if not records:return '<p>无可评分目标。</p>'
    cols=list(records[0]);return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[k],3) if isinstance(r[k],float) else r[k]))+'</td>' for k in cols)+'</tr>' for r in records)+'</tbody></table></div>'

def run():
    latest.verify();actual.verify()
    original=load(BASE/'stage69_event_force_forecast_20261021/analysis.json')
    increment=load(BASE/'stage72_force_innovation_20261021/analysis.json')
    extended=load(OUT/'temporal_scores73.json');full=load(OUT/'scoring/analysis.json')
    raw71=load(actual.OUT/'checks71.json');control=load(BASE/'stage68_event_force_input_20261021/controls68.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    models=['trusted_acc','force_density','force_periodic','increment_periodic_constant','increment_periodic']
    short=['History','Density force','Periodic force','Direction increment','Event increment']
    old_table=[]
    for ax,case in zip(axes,NAMES):
        rr=[]
        for m in models:
            source=increment if m.startswith('increment_') else original
            r=next(x for x in source['table'] if x['case']==case and x['model']==m and x['nominal_h']==.4)
            rr.append(r);old_table.append({k:r[k] for k in ['case','model','present_targets','valid_forecasts','mean_fde_px','joint_success']})
        xx=np.arange(len(models));values=[r['mean_fde_px'] for r in rr]
        bars=ax.bar(xx,values,color=['#52677e','#c66d50','#c5a55b','#8894a6','#6e53a2'])
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
        ax.set_xticks(xx,short,rotation=45,ha='right',fontsize=8);ax.set_title(f"{NAMES[case]} / n={rr[0]['valid_forecasts']}")
        ax.set_ylabel('0.4 s endpoint error (px)');ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Known development clips: connected force input does not imply better prediction')
    fig.savefig(OUT/'known_prefix_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained');new_table=[]
    for ax,case in zip(axes,NAMES):
        rr=[r for r in extended['table'] if r['case']==case and r['region']=='later_extension' and r['nominal_h']==.4 and r['model'] in ['trusted_acc','increment_periodic_constant','increment_periodic','increment_guarded_events']]
        order=['trusted_acc','increment_periodic_constant','increment_periodic','increment_guarded_events'];rr=sorted(rr,key=lambda r:order.index(r['model']))
        values=[r['mean_fde_px'] for r in rr];new_table+=rr
        if all(v is not None for v in values):
            bars=ax.bar(np.arange(len(rr)),values,color=['#52677e','#8894a6','#6e53a2','#438b72']);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
            ax.set_ylim(0,max(values+[1])*1.25)
        ax.set_xticks(np.arange(len(rr)),['History','Direction inc.','Event inc.','Guarded events'],rotation=35,ha='right',fontsize=8)
        ax.set_title(f"{NAMES[case]} / n={rr[0]['valid'] if rr else 0}");ax.set_ylabel('0.4 s endpoint error (px)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Later temporal extension: same frozen parameters, sparse boxes, same recordings')
    fig.savefig(OUT/'temporal_extension_comparison.png',dpi=160);plt.close(fig)
    fixed=[];raw_table=[];timing=[];usage=[]
    for case in latest.CASES:
        cfg=latest.CASES[case];rr=rows(OUT/case/'observations_forecasts.jsonl');completion=load(OUT/case/'completion.json');check=load(OUT/case/'regression73.json')
        assert completion['source_hashes_unchanged'] and check['original30_prefix_exact'] and not completion['labels_read']
        raw_table.append(dict(case=case,parent=len(rr),native10=2*len(rr),old_prefix_exact=check['old_prefix_frames'],video_frames=check['video_frames_decoded']))
        timing.append(dict(case=case,median_ms=completion['processing_median_ms'],p95_ms=completion['processing_p95_ms'],fraction_over20ms=completion['fraction_over20ms']))
        for method in ['guarded_events','guarded_constants','guarded_all']:
            choices={}
            for r in rr:
                name=r['frontends']['delayed_feedback']['increment_guard_diagnostic']['choices'][method]['chosen'];choices[name]=choices.get(name,0)+1
            usage.append(dict(case=case,branch='increment_'+method,choices=choices))
        if case=='negative5':continue
        # Fixed by temporal midpoint of the new3s region, never by prediction success.
        desired=latest.NEW_START[case]+1.7
        candidates=[d for d in full['details'] if d['case']==case and d['model']=='increment_periodic' and d['nominal_h']==.4 and d['forecast_valid'] and d['target_state']=='present' and d['target_t']>latest.NEW_START[case]+.4]
        if not candidates:continue
        d=min(candidates,key=lambda d:abs(d['target_t']-desired));index=round((d['anchor_t']-cfg['start'])*50)-1;r=rr[index];q=r['frontends']['delayed_feedback']
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();cap.release();assert ok
        target=np.rint(d['target_center']).astype(int);cv2.drawMarker(im,tuple(target),(255,220,30),cv2.MARKER_CROSS,26,2)
        cv2.putText(im,f"Future box center at {d['target_t']:.2f}s (evaluation only)",tuple(np.maximum(target+[10,-14],[10,25])),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,220,30),1)
        cv2.putText(im,'GREEN history | PURPLE event increment | CYAN future sparse truth',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        name=f'{case}_fixed_future.png';cv2.imencode('.png',im)[1].tofile(str(OUT/name))
        fixed.append(dict(case=case,anchor_t=d['anchor_t'],target_t=d['target_t'],event_error_px=d['error_px'],file=name))
    save(OUT/'delivery_data73.json',dict(known_table=old_table,extension_table=new_table,raw_table=raw_table,timing=timing,guard_usage=usage,fixed=fixed,
        old71=raw71,controls68=control['table']))
    videos=''.join('<article><h3>'+html.escape(c)+'</h3><video controls preload="metadata" src="'+c+'/forecasts.mp4"></video></article>' for c in NAMES)
    pictures=''.join('<figure><img src="'+r['file']+'"><figcaption>'+html.escape(str({k:v for k,v in r.items() if k!='file'}))+'</figcaption></figure>' for r in fixed)
    body_html='''<h1>事件活动已经接入推力与位置预测：68–73阶段</h1>
<div class="lead"><b>完整链路已运行；稳定优于纯历史预测的目标仍未证明。</b><p>直接RAW → 当前无人机框 → 局部ON/OFF活动/周期 → 2D有效推力代理 → 延迟位置反馈更新 → 未来位置。侧视不要求四桨或四个桨心。默认入口没有升级，原始失败与恒定输入对照保留。</p></div>
<h2>这次实际改了什么</h2><ol><li>68：当前20ms双极事件率按ROI面积归一化并减背景；160ms分格极性频谱增加时间相位、解释能量与背景共同周期排除。没有G灰度重建。</li><li>69：活动平方或局部频率平方进入2D有效推力模型；使用已经成熟的过去预测与当前检测位置更新RLS。各有相同方向、准入和回退的恒定输入版本。另做历史预测残差校正。</li><li>70：只在过去0.2/0.4秒预测都优于历史时采用物理分支；事件分支还须优于匹配的恒定输入。该门槛在原片段没有批准任何事件分支，完全退回历史；不能称为事件收益。</li><li>71：建立真正只读RAW和固定检测权重的独立入口，所有26路预测、当前观测和学习状态与缓存消融精确一致；两段实际RAW截断前缀通过。</li><li>72：用连续相对频率和推力变化量修正历史预测，避免稳态加速度重复计入。倍频/半频连续性只是先验，仍不知道机械RPM。</li><li>73：冻结参数后延长清晰到27秒、侧视到116秒、3号到34秒；保留原30路预测，新增三路增量保护分支。省掉未使用的三张支持图、缓存相位载波，原区域逐项回归通过。</li></ol>
<h2>旧开发片段：0.4秒稀疏未来框中心误差</h2><p>清晰只有2个、侧视5个、3号11个有效目标。清晰增量频率14.31→12.74px是小样本信号；侧视恒定方向增量还比事件增量好，3号事件增量略差。不能概括为物理方法已胜出。</p><img src="known_prefix_comparison.png">'''+table(old_table)+'''
<h2>后续时间段验证</h2><p>参数在读取这些后续未来标签前冻结；仍是相同开发录像，不是新无人机或盲测。只统计锚点在新增区域的目标（目标时间大于24.4/113.4/31.4秒）。边界目标完整保存在scoring/analysis.json，但不挤入下表。误差是可见框中心FDE，不是密集轨迹ADE或真实重心误差。</p><img src="temporal_extension_comparison.png">'''+table(new_table)+'''
<h2>实际预测视频</h2><p>绿色：纯历史；紫色：本轮事件推力增量预测，可能退化或与绿色重合。视频按源时间50fps播放，实际处理速度见下表，不能据此称实时。默认入口仍旧54。</p><div class="grid">'''+videos+'''</div>
<h2>固定画面与未来位置</h2><p>从后续时间段中点附近按时间选图，没有按误差挑成功帧。青色十字只在评分后标出未来稀疏框中心，推理时不可见。</p>'''+pictures+'''
<h2>实际处理与保护分支用量</h2>'''+table(raw_table)+table(timing)+table(usage)+'''
<p>71另有985父/1970原生窗口（700主段+285重复前缀）及650主视频帧；另40父/80窗口清晰pilot，均已结束。73与71旧区间重叠，不能相加成独立样本。</p>
<h2>周期控制与尚未解决的困难</h2>'''+table([dict(kind=k,**v) for k,v in control['table'].items()])+'''
<p>新门槛在这些计数过程控制中拒绝了有色变化、单次突发、共同闪烁，但40组频率切换仍有7组通过；混合频率保留30/40。不是硬件桨叶RPM准确率。实采空场没有NN身份，零发布不能作为完整周期检测误报率。</p>
<ul><li>事件强度受对比度、可见面积、姿态、平移、阈值影响；事件多不能直接等同推力大。局部频率仍可能是谐波或相邻桨混合。</li><li>整机长轴不总等于桨盘平面；侧视法线符号依靠直立/时间连续先验，2D方向没有完整3D姿态。</li><li>位置反馈含检测偏差；在线学习可能把检测跳变学成外力。有效像素系数不等于真实质量、风速或推力系数，单目数据存在不可辨识性。</li><li>160ms过去窗口及目前处理耗时都限制响应速度。本轮完成工程接线与可审计对照，尚无持续、准确、实时预测或超过前沿论文的证据。</li></ul>
<p>下一步优先在固定测试方案下验证局部周期与真实转速的关联、对象身份稳定性和投影尺度变化，再扩展独立录像/密集轨迹标注。不要继续用这几个稀疏目标反复选参数并称泛化。</p>
<h2>来源与复用范围</h2><p><a href="https://arxiv.org/abs/2604.18289">原桨叶状态估计论文</a>、<a href="https://arxiv.org/abs/2511.13100">Count Every Rotation</a>、<a href="https://arxiv.org/abs/2603.01997">Event-Only Drone Trajectory Forecasting</a>作为思路来源；实际代码复用本项目原6/21/29/52/62/64/66模块及既有SDK，新增代理/胶合/消融。没有声称运行这些论文未验证的作者代码或完整复现其指标，没有新依赖、没有访问yolov10-main。</p>
<p><a href="protocol.json">73冻结协议</a> · <a href="temporal_scores73.json">时间分段评分</a> · <a href="scoring/analysis.json">全部目标与所有33模型评分</a> · <a href="delivery_data73.json">本页数据</a> · <a href="../stage71_RAW_event_force_20261021/checks71.json">71完整RAW/前缀检查</a> · <a href="../stage69_event_force_forecast_20261021/analysis.json">69全部失败/对照</a></p>'''
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>事件推力与未来位置实验报告</title><style>body{font:16px/1.7 system-ui;background:#f2f5f8;color:#203044;max-width:1260px;margin:32px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:36px}.lead,article,figure{background:white;padding:20px;border-radius:12px;margin:18px 0}.lead{border-left:6px solid #7460aa}img{max-width:100%;height:auto;background:white}video{width:100%;background:#111}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.scroll{overflow:auto;margin:20px 0}table{border-collapse:collapse;background:white;font-size:13px;width:100%}th,td{border:1px solid #d4dde6;text-align:left;padding:7px;vertical-align:top}th{background:#e6edf5}a{color:#245e99}figcaption{font-size:13px;overflow-wrap:anywhere}</style>'+body_html+'</html>'
    path=OUT/'第六十八至七十三阶段事件推力预测报告.html';path.write_text(page,encoding='utf-8')
    save(OUT/'delivery73.json',dict(helper_sha256=sha(__file__),report=str(path),raw_cases=len(raw_table),fixed_frames=fixed,
        objective_achieved=False,default_promoted=False,actual_RAW_force_chain_completed=True,
        no_real_RPM_or_wind_identification=True))
    print('68–73 report/plots/videos delivered:',path,flush=True)

if __name__=='__main__':run()
