"""Report current-axis availability without treating it as real pose accuracy."""
import html
import cv2
import numpy as np
import study63 as current
OUT=current.OUT;rows=current.rows;load=current.load;save=current.save;sha=current.sha


def table(rr):
    cols=list(rr[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[k],3) if isinstance(r[k],float) else r[k]))+'</td>' for k in cols)+'</tr>' for r in rr)+'</table></div>'


def run():
    current.verify();summary=[];angles=[];prefix=[]
    for case in current.CASES:
        d=load(OUT/case/'completion.json');rr=rows(OUT/case/'observations.jsonl')
        assert d['native10']==round(current.CASES[case]['duration']*100) and d['parent']==len(rr)
        assert not d['labels_read'] and not d['future_forecasts_executed']
        for name in current.METHODS:
            s=d['methods'][name]
            summary.append(dict(case=case,method=name,parent=d['parent'],NN_supported=s.get('body_supported',0),
                axis_candidate=s.get('valid',0),without_current_rotor_axis=s.get('without_current_rotor_axis',0),
                while62_axis_inactive=s.get('while62_axis_inactive',0),measurement_median_ms=s['measurement_median_ms'],rejections=s['reasons']))
            values=[r['methods'][name]['undirected_disagreement_with_stored_rotor_axis_deg'] for r in rr if r['methods'][name].get('valid') and 'undirected_disagreement_with_stored_rotor_axis_deg' in r['methods'][name]]
            jumps=[]
            for a,b in zip(rr,rr[1:]):
                a=a['methods'][name];b=b['methods'][name]
                if a.get('valid') and b.get('valid'):
                    jumps.append(float(np.degrees(abs((a['angle_rad']-b['angle_rad']+np.pi/2)%np.pi-np.pi/2))))
            angles.append(dict(case=case,method=name,paired_with_rotor_axis=len(values),
                median_disagreement_deg=float(np.median(values)) if values else None,
                p90_disagreement_deg=float(np.percentile(values,90)) if values else None,
                consecutive_candidate_pairs=len(jumps),median_axis_change_deg=float(np.median(jumps)) if jumps else None,
                warning='Stored rotor axis and temporal smoothness are diagnostics, not independent attitude truth.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.1),layout='constrained')
    for ax,case in zip(axes,['clear6','side6','train3']):
        selected=[r for r in summary if r['case']==case];x=np.arange(2)
        ax.bar(x-.18,[r['axis_candidate'] for r in selected],.36,label='Direction candidates',color='#277da1')
        ax.bar(x+.18,[r['while62_axis_inactive'] for r in selected],.36,label='During62 axis inactive',color='#e76f51')
        for i,r in enumerate(selected):
            ax.text(i-.18,r['axis_candidate']+4,str(r['axis_candidate']),ha='center')
            ax.text(i+.18,r['while62_axis_inactive']+4,str(r['while62_axis_inactive']),ha='center')
        ax.set(title=case,xticks=x,xticklabels=['All event counts','Balanced ON/OFF'],ylim=(0,selected[0]['parent']*1.12),ylabel='Current20ms steps');ax.legend(fontsize=7)
    fig.suptitle('Whole-body PCA/LSD directions: candidate counts, not verified plane normals')
    fig.savefig(OUT/'body_axis_availability63.png',dpi=160);plt.close(fig)
    result=dict(helper_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),table=summary,angle_diagnostics=angles,
        native_RAW_ticks=1400,current_parent_ticks=700,new_future_scores=False,new_RPM=False,
        deployable_joint_entry=False,default_promoted=False,
        conclusion='No-four-lock whole-body axis can be extracted from current NN boxes. '
                   'Side139/250 candidates add0 intervals beyond62 active axis. '
                   'Train3 raw-count support gives217 candidates and135 during62 inactivity, '
                   'balanced ON/OFF only81 and4. Many extra count-support axes may be body '
                   'edges or motion trails; no independently verified plane normal. '
                   'Current NN confidence gate still limits low-visibility side intervals.',
        next='Do not insert unvalidated body major axis as a full 3D normal. '
             'First reuse6/21 causal motion compensation with past body CV velocity, '
             'compare current support versus accumulated trails and threshold/temporal '
             'axis stability on known-projection controls. A line constrains direction '
             'but not tilt-to-camera or thrust magnitude. Then isolated63 axis-only '
             'prediction ablation if evidence passes; local periodicity extraction '
             'remains a separate work item.')
    save(OUT/'delivery63.json',result)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>63：不依赖四桨的整机轴观测</title><style>body{max-width:1400px;margin:28px auto;padding:0 22px;font:16px/1.8 system-ui;color:#243c49;background:#f7fafc}.lead{padding:16px;background:#eaf1fa;border-left:5px solid #277da1}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:13px}td,th{border:1px solid #ccd8df;padding:6px;vertical-align:top}img{max-width:100%;height:auto}a{color:#2457a3}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>63：不要求四桨锁定，先提取整机侧影方向</h1><div class="lead"><b>能提到线方向，但没有验证它就是桨盘平面方向。</b><p>实际复用原6 PCA/多阈值/LSD模块，将调用条件改为当前NN整机身份，不再要求旧四点模型。侧视250时刻有139个候选，但都落在62方向已可用的区间；3号全事件支持在62失效时补出135个候选，双极ON/OFF只补4个。不能把更多候选直接当成姿态精度或预测增益。</p></div><h2>本轮范围</h2><p>14秒真实RAW，1400原生10ms窗口、700当前20ms时刻；直接读取RAW生成指数衰减事件支持，使用61相同截止时刻NN框。total_decay复用原6全事件计数；bipolar_decay复用原8邻域ON/OFF最小值，再用相同20ms衰减。PCA/LSD模块及所有阈值不改，无G通道/亮度重建，无四点初始化，也不生成桨心、机械RPM或推力。当前使用已保存同截止时刻NN框，属于观测诊断，尚不是新的自动RAW联合入口。</p><h2>完整可用性结果</h2><img src="body_axis_availability63.png">'''+table(summary)+'''<h2>方向一致性诊断</h2><p>与已有转子轴比较及相邻时刻变化都不是姿态真值。滑动事件记忆也会让错误方向看起来平滑，因此不能只按时间稳定就认定是正确桨盘平面。</p>'''+table(angles)+'''<h2>固定画面：左全事件，右双极ON/OFF</h2><p>展示各段25%/50%/75%中恰好落在20ms发布网格且实际保存的时刻；侧视109.25/111.75秒和空场0.25/0.75秒不在网格上，因此不补造画面。没有按成功挑选。下图为同一个当前NN框的局部放大，原图链接也保留。绿线表示通过旧观测器内部门槛，红线表示有方向但门槛不通过；颜色不表示真实姿态准确。</p>'''
    for case in current.CASES:
        cfg=current.CASES[case]
        input_rows={round(r['t'],8):r for r in rows(current.REF/f'off_{case}/observations_forecasts.jsonl')}
        for f in [.25,.5,.75]:
            t=cfg['start']+cfg['duration']*f
            paths=[OUT/case/(method+f'_{t:.2f}.png') for method in current.METHODS]
            if not all(p.exists() for p in paths):continue
            page+='<h3>'+case+' '+f'{t:.2f}s'+'</h3><div class="pair">'
            for method,path in zip(current.METHODS,paths):
                image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
                body=input_rows[round(t,8)]['body_observation']
                if body.get('detected'):
                    b=np.rint(body['bbox']).astype(int);lo=np.maximum(b[:2]-30,[0,0]);hi=np.minimum(b[2:]+30,[1280,720]);image=image[lo[1]:hi[1],lo[0]:hi[0]]
                zoom=path.with_name('zoom_'+path.name);cv2.imencode('.png',image)[1].tofile(str(zoom))
                page+='<a href="'+case+'/'+path.name+'"><img src="'+case+'/'+zoom.name+'"></a>'
            page+='</div>'
    page+='''<h2>当前真正卡住的位置</h2><p>1. 侧视漏检时，NN身份也经常缺失，单靠换方向提取器不能跨过这个限制；本轮139个侧视候选没有填补62方向失效区间。2. 全事件保留机臂、机身边缘与平移拖尾，3号虽然候选大增，其物理含义仍未验证。3. 双极活动可以抑制单次运动边缘，也会删除侧视变稀的有效支持，3号217→81。4. 整机主轴不严格等于桨盘平面的投影，清晰四盘可见时旧观测器也经常判为side_body_axis；它不能独自给出相机深度方向的倾角、法线正负或推力幅值。</p><p>下一步应先复用原6/21已有补偿，用过去整机CV速度区分机体平移拖尾与刚体方向，配合已知投影控制验证角度误差，再对可信线只更新无向轴。不能直接把PCA轴补成完整3D法线，也不凭事件总数补四桨转速。由于尚无这组验证，本轮未把观测候选强行接入预测，更没有新未来误差成绩。</p><p>复查了<a href="https://github.com/event-driven-robotics/RT-EvLDT">RT-EvLDT作者库</a>和<a href="https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html">LSD官方API</a>；实际执行既有本地6/8模块与安装的OpenCV5，未执行RT-EvLDT，无新颖算法主张。</p><p><a href="delivery63.json">交付与下一步</a> · <a href="protocol.json">冻结观测协议</a> · <a href="../stage62_independent_axis_20261021/第六十二阶段方向与转速独立使用报告.html">62实际预测结果</a></p></html>'''
    (OUT/'第六十三阶段不依赖四桨的整机轴观测报告.html').write_text(page,encoding='utf-8')
    print('63 observation report saved; no new future prediction claim',flush=True)


if __name__=='__main__':run()
