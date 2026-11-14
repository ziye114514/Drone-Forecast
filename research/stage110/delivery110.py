"""Post-forecast scoring and reviewable delivery for the fixed morphology pilot."""
from pathlib import Path
import collections, html, json, sys
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage110'),str(ROOT/'research/stage109'),str(ROOT/'research/stage107'),str(ROOT/'research/stage74')]
from study110 import OUT, CASES, PARAMS, verify, rows, save, summarize
from center107 import NAMES
from hybrid74 import prior
AUDIT=ROOT.parent/'output/stage109_rotor_extraction_audit_20261022'
OLD=ROOT.parent/'output/stage107_center_bias_filter_20261022'
LABELS={'closed':'原闭运算（先判线）','open_only':'全局取消闭运算','merge_safe':'只保护被合并的合格分量'}

def score():
    verify()
    # Deliberately read targets here, after all forecasts have been saved.
    targets=json.loads((ROOT.parent/'output/stage78_RAW_separated_forecast_20261021/scoring/analysis.json').read_text('utf-8'))['details']
    details=[]
    for case in ['clear6','side6']:
        pred={round(r['t'],8):r for r in rows(OUT/case/'records.jsonl')}
        old={round(r['t'],8):r['frontends']['delayed_feedback'] for r in rows(ROOT.parent/CASES[case]['reference'])}
        for target in targets:
            if target['case']!=case or target['frontend']!='event_force' or target['model']!='hold' or target['target_state']!='present':continue
            t=round(target['anchor_t'],8) if target.get('anchor_available') else None
            r=pred.get(t) if t is not None else None
            coefficients={}
            if r is not None and r['valid']:
                for mode in PARAMS['modes']:
                    coefficients.update({mode+'/'+k:v for k,v in r['outputs'][mode]['forecast']['coefficients'].items()})
                coefficients['original_fusion']=old[t]['coefficients'].get('complement_fallback')
            for name in [mode+'/'+n for mode in PARAMS['modes'] for n in NAMES]+['original_fusion']:
                c=coefficients.get(name);err=None
                if c is not None:
                    h=target['actual_h'];position=np.asarray(prior.evaluate(c,h))
                    err=float(np.linalg.norm(position-target['target_center']))
                details.append(dict(case=case,method=name,h=target['nominal_h'],error_px=err,valid=c is not None,
                    anchor_t=target.get('anchor_t'),target_t=target['target_t']))
    table=[]
    for case in ['clear6','side6']:
        for h in [.1,.2,.4]:
            for method in sorted({r['method'] for r in details}):
                group=[r for r in details if r['case']==case and r['h']==h and r['method']==method]
                errors=[r['error_px'] for r in group if r['valid']]
                table.append(dict(case=case,h=h,method=method,n=len(errors),targets=len(group),mean_px=float(np.mean(errors)) if errors else None))
    save(OUT/'scores110.json',dict(table=table,details=details,
        scope='Previously examined sparse future BOX centers, not actual COM/pose/RPM. Geometry was newly decoded fromRAW; matched body/axis were replayed. No score for continuous6 without independent labels.'))
    print(json.dumps([r for r in table if r['h']==.4 and ('fixed' in r['method'] or r['method']=='original_fusion')]),flush=True)

def images():
    allrows=rows(OUT/'continuous6/records.jsonl');selected=next(r for r in allrows if abs(r['t']-112.02)<1e-8)
    a=np.load(AUDIT/'continuous6/112.02_counts.npz');on=a['on'];off=a['off'];origin=a['origin']
    base=np.full((*on.shape,3),255,np.uint8);base[(on>0)&(off==0)]=(40,80,220);base[(off>0)&(on==0)]=(220,170,20);base[(on>0)&(off>0)]=(15,15,15)
    grid=np.full((325,1260,3),245,np.uint8)
    for i,mode in enumerate(PARAMS['modes']):
        panel=base.copy();o=selected['outputs'][mode]
        for j,r in enumerate(o['geometry'].get('rotors',[])):
            c=np.array([r['cx'],r['cy']])-origin
            cv2.ellipse(panel,(tuple(c),(r['major'],r['minor']),r['angle']),(20,170,20),1)
            cv2.circle(panel,tuple(np.rint(c).astype(int)),2,(0,100,0),-1)
        scale=min(410/panel.shape[1],245/panel.shape[0]);im=cv2.resize(panel,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        grid[:im.shape[0],i*420:i*420+im.shape[1]]=im
        cv2.putText(grid,mode+': '+str(len(o['geometry'].get('rotors',[])))+' selected',(i*420+5,267),0,.56,(20,35,50),1)
        cv2.putText(grid,o['center']['measurement']['reason'],(i*420+5,296),0,.39,(20,35,50),1)
    cv2.imwrite(str(OUT/'C112_02_comparison110.png'),grid)
    # Scientific plots, independently named from native ON/OFF evidence.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    ts=np.array([r['t'] for r in allrows]);colors=['#777777','#ea8a23','#1a9290']
    for mode,color in zip(PARAMS['modes'],colors):
        counts=[len(r['outputs'][mode]['geometry'].get('rotors',[])) for r in allrows]
        current=[int(r['outputs'][mode]['center']['current']) for r in allrows]
        axes[0].plot(ts,counts,label=mode,color=color,alpha=.8)
        axes[1].plot(ts,np.cumsum(current),label=mode,color=color)
        values=[np.linalg.norm(r['outputs'][mode]['forecast']['coefficients'].get('center_bias_fixed107',dict(velocity=[np.nan,np.nan]))['velocity']) for r in allrows]
        axes[2].plot(ts,values,label=mode,color=color,alpha=.8)
    axes[0].set_ylabel('Selected candidates');axes[1].set_ylabel('Cumulative accepted centers');axes[2].set_ylabel('Estimated speed (px/s)');axes[2].set_xlabel('Source time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Development continuous6: measurements and estimated speed, no COM truth');fig.tight_layout();fig.savefig(OUT/'continuous_measurements110.png',dpi=150);plt.close(fig)
    return selected

def table_html(records,fields):
    return '<table><tr>'+''.join('<th>'+html.escape(label)+'</th>' for key,label in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(key,'')))+'</td>' for key,label in fields)+'</tr>' for r in records)+'</table>'

def report():
    verify();summarize();score();selected=images()
    availability=json.loads((OUT/'availability110.json').read_text('utf-8'));scores=json.loads((OUT/'scores110.json').read_text('utf-8'))
    scoretable=[r for r in scores['table'] if r['h']==.4 and (r['method'].endswith('center_body_kf107') or r['method'].endswith('center_smooth_kf107') or r['method'].endswith('center_bias_fixed107') or r['method']=='original_fusion')]
    table=table_html(availability,[('case','片段'),('mode','提取方法'),('frames','时刻'),('four_candidates','四候选'),('qualified_conic','合格共面拟合'),('current_center','当前中心更新'),('initializations','初始化'),('max_qualified_streak','最长连续合格')])
    scores_html=table_html(scoretable,[('case','片段'),('method','方法'),('n','有效端点'),('targets','全部端点'),('mean_px','0.4秒框中心误差/px')])
    c={mode:dict(count=len(selected['outputs'][mode]['geometry'].get('rotors',[])),reason=selected['outputs'][mode]['center']['measurement']['reason']) for mode in PARAMS['modes']}
    checks={case:json.loads((OUT/case/'checks110.json').read_text('utf-8')) for case in CASES}
    for case in ['clear6','continuous6']:checks[case+'_prefix']=json.loads((OUT/(case+'_prefix')/'prefix110.json').read_text('utf-8'))
    save(OUT/'delivery_checks110.json',dict(checks=checks,C112_02=c,default_promoted=False,
        report_scope='Completed fixed RAW-geometry/downstream-replay pilot. No new full deployment, no COM/RPM truth, no proven originality or real-time claim.'))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>109—110：闭运算合并与姿态中心分离</title><style>body{max-width:1360px;margin:28px auto;padding:0 20px;font:17px/1.8 system-ui;color:#243b4a}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #cbd5df;padding:6px}th{background:#edf5f7}code{background:#edf2f5}a{color:#087c87}.notice{padding:12px;background:#fff3db}</style><h1>109—110：先修复观测，再判断姿态与真实位移的分离</h1>
<p>本轮找到 C112.02 秒漏桨的一个明确来源：原生双极事件经过阈值化、开运算后，左侧仍有两个独立分量；3×3闭运算把它们连接，随后被当成一个大椭圆。不能通过直接补第四个点解决。</p>
<p>109重新读取三段 RAW，共555个20ms时刻；仅记录中间层，原20ms几何逐帧严格一致。C帧的阈值为2.0016408，左侧开运算分量面积287/584像素，闭运算合并后面积913像素。</p>
<p><a href="../stage109_rotor_extraction_audit_20261022/continuous6/112.02_pipeline.png">完整提取链路图</a></p><img src="../stage109_rotor_extraction_audit_20261022/continuous6/112.02_pipeline.png">
<h2>固定三路对照</h2><p>原闭运算（统一先判线控制）；全局取消闭运算；有条件保护原分量。第三路只在闭运算连接至少两个分别满足椭圆条件、面积覆盖及尺度条件的分量时保留子轮廓，其余位置保留原闭运算。无固定四个峰、无隐藏桨心、无修改五帧初始化/共面圆拟合门槛。</p>
<p>复用 <a href="https://docs.opencv.org/5.0/tutorials/imgproc/imgtrans/distance_transformation/distance_transform.html">OpenCV连通域/距离变换相关工具</a>及本项目原椭圆/分组/滤波代码。另核查 <a href="https://github.com/Li-Zhaoxi/AAMED">AAMED弧段椭圆检测</a>，本轮未编译运行。C已有分离分量，因此先检验形态学保护，不增加水岭拆分和编译依赖。距离变换仅作诊断图，不作为本轮推理。</p>
<img src="C112_02_comparison110.png"><p>C帧实际结果：'''+html.escape(json.dumps(c,ensure_ascii=False))+'''。</p>
<h2>完整固定窗口的观测结果</h2>'''+table+'''<p>四候选数不是四个真实电机的准确率；合格共面拟合仍是模型一致性检验。连续段与侧视段重叠，均是已查看过的开发数据，不能累加当独立录像，也不能称盲测。</p><img src="continuous_measurements110.png">
<h2>预测对照与边界</h2>'''+scores_html+'''<p class="notice">本轮新读 RAW 提取几何；机体框及侧影轴来自同一时刻的旧完整RAW输出，用于匹配下游输入。此处是模块组合小试验，尚未新增完整RAW-only部署入口。评分目标是既有稀疏未来框中心，不是实际重心、姿态或RPM真值；连续段没有独立标签，不报告其预测精度。保留全部无效输出与退步，默认54/API93不变。</p>
<h2>创新应该落实在哪里</h2><p>修补闭运算属于已有工具的针对性改进，本轮不称原创算法。真正的后续贡献候选仍是：利用事件桨盘的结构证据，把姿态造成的观测中心变化与真实平移分开，并在部分重叠/成线时分别管理姿态与位置观测的可靠性。应先证明恢复出来的轮廓确实对应桨盘，再检验连续初始化、退化和恢复；不能因候选变多就放宽门槛。</p>
<p>下一步依据本表的首要失败层：若拆分恢复但共面拟合仍不稳，应改为真实支持弧段的联合拟合/支持量诊断，避免把阈值内不完整区域拟合成的椭圆当真实桨盘；随后才比较真实转向与原地姿态波动，加入普通平滑和相同发布覆盖的控制。</p>
<p><a href="protocol110.json">冻结方案</a> · <a href="availability110.json">全部可用性/拒绝原因</a> · <a href="scores110.json">全部时距/逐端点评分</a> · <a href="delivery_checks110.json">校验与实际前缀</a></p></html>'''
    dest=OUT/'第一百零九至一百一十阶段桨盘合并与中心预测报告.html';dest.write_text(page,encoding='utf-8');print(str(dest),flush=True)

if __name__=='__main__':report()
