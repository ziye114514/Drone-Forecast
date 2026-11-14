"""Correct128's apparent early-lock gain; report guard pilot and failures."""
from pathlib import Path
import collections,html,json,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage129'),str(ROOT/'research/stage128')]
from study129 import OUT,CASES,verify,save,NAMES
from study128 import OUT as OLD
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def main():
    verify()
    for case in ['prefix_natural6','negative5']:assert (OUT/'raw'/case/'checks129.json').exists()
    scores=load(OLD/'scores128.json');table=[];points=[];records={c:{round(r['t'],8):r['output'] for r in load(OUT/(c+'.json'))} for c in CASES}
    for pair in scores['pairs']:
        c=records[pair['case']][round(pair['origin'],8)]['forecast']['coefficients'].get(NAMES[2]);assert c is not None
        prediction=np.array(c['position'])+pair['actual_h']*np.array(c['velocity']);error=float(np.linalg.norm(prediction-pair['truth_bbox_center']))
        points.append(dict(case=pair['case'],h=pair['h'],origin=pair['origin'],target=pair['target'],error_px=error,prediction=prediction.tolist()))
    for row in scores['table']:
        if row['method']!='count_model':continue
        values=[p['error_px'] for p in points if p['case']==row['case'] and p['h']==row['h']];assert len(values)==row['paired']
        table.append(dict(row,method='FOV_guard',mean_px=float(np.mean(values)) if values else None,median_px=float(np.median(values)) if values else None,rmse_px=float(np.sqrt(np.mean(np.square(values)))) if values else None))
    save(OUT/'scores129.json',dict(table=table,pairs=points,scope='Cached-current129 model replay at identical128 futureBOX target/origin pairs. Full129 RAW deployment tested only60 positive+30 negative, not400. Rule fixed after128 outcome, developmental audit not independent efficacy test.'))
    availability=load(OUT/'availability129.json');clear=next(r for r in availability if r['case']=='natural6');before=next(r for r in scores['table'] if r['case']=='natural6' and r['h']==.4 and r['method']=='count_model');uncorrected=next(r for r in scores['table'] if r['case']=='natural6' and r['h']==.4 and r['method']=='independent_models');now=next(r for r in table if r['case']=='natural6' and r['h']==.4)
    old=load(ROOT.parent/'output/stage127_independent_support_20261023/natural6.json');audit=[]
    for r in old:
        if r['output']['chosen']!='occupancy':continue
        a=records['natural6'][round(r['t'],8)]['boundary_audit']['occupancy'];audit.append(dict(t=r['t'],old_chosen='occupancy',any_fitted_ellipse_outside_FOV=any(v['ellipse_outside_sensor'] for v in a),ellipse_bounds=a))
    assert len(audit)==8 and sum(r['any_fitted_ellipse_outside_FOV'] for r in audit)==6
    save(OUT/'apparent_early_gain_audit129.json',dict(occupancy_only_updates=8,updates_with_outside_FOV_fitted_ellipse=6,records=audit,source='127 observations/fusion previously proven exactly actual128400; current129 boundary audit on same inputs.',not_physical_rotor_truth=True))
    def tab(rr,keys):
        return '<table><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.3f}' if isinstance(r.get(k),float) else str(r.get(k,'')))+'</td>' for k,_ in keys)+'</tr>' for r in rr)+'</table>'
    outcome=f'原计数0.4秒平均{before["mean_px"]:.3f}px；未加边界的独立模型{uncorrected["mean_px"]:.3f}px；修正完整性后{now["mean_px"]:.3f}px，反而变差（同11个开发端点，缓存当前模型重放，非COM）。不推荐替换主预测路径。'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>桨盘边界完整性修复与最终结论</title><style>body{max-width:1140px;margin:30px auto;padding:0 24px;font:17px/1.85 system-ui;color:#243746}h2{margin-top:2em}.note{background:#fff0d3;padding:17px}img,video{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #cdd7dd}th{background:#edf4f5}a{color:#087d89}</style><h1>桨盘提取与边界完整性：修了什么，还没解决什么</h1><p>2026-10-24。124—128完成原生支持与独立模型的完整实验；129在视频复核中发现并修复“出画椭圆被当成完整观测”的可靠性问题。</p>
<p class="note">当前不能说侧飞预测卡点已解决。二值ON/OFF支路恢复了部分被计数门槛压掉的轮廓，但独用会退步。独立模型原先表面上236→244次有效几何，且提前0.12秒；视频复核发现其中6次新观测的拟合边界超出相机视野，这个更早初始化不能作为可靠收益。修正后当前几何235/400，首次22.86秒，未来框预测反而变差；实验支路不替换主路径。</p>
<h2>视频复核发现的真实错误</h2><img src="../stage128_RAW_independent_support_20261023/natural6_poster128.png"><p>22.68秒右上桨盘的拟合椭圆延伸至x≈1288，而相机最后一个像素中心是1279。原程序仍把它标为完整ellipse、visible_arc_fraction=1，并据此参与完整四桨初始化。新增8次占据补位中，22.68—22.78秒的6次都有拟合边界越界。其余两次为22.88和22.94秒。</p>
<h2>已经加入的修复</h2><p>按旋转椭圆的解析包围边界检查完整性：超出[0,1279]×[0,719]的拟合不再声明为完整桨盘，而保留为当前部分观测，标记arc/inferred_shape，完整可见弧比例设为未知。原四桨/三桨资格因此拒绝用它完成初始化。没有删除真实事件或补造隐藏点，也没有降低原五帧门槛。</p><p>计数和占据各自学习结构，位置/速度仍在同一个KF里。边界规则只保证不把视野外拟合当成完整观测；它不是物理桨盘真值、真实遮挡分割或误差置信保证。占据提取、双模型、滤波和该几何守门本身都不称创新。</p>
<h2>修正后的完整分母</h2>'''+tab(availability,[('case','片段'),('frames','时刻'),('clipped_count_frames','计数含越界时刻'),('clipped_occupancy_frames','占据含越界时刻'),('current','当前模型'),('first_current','首次当前/s'),('first_occupancy_model','占据首次学好/s')])+'''
<p>清晰段：原计数236/400；未守门独立模型244/400；修正后235/400，其中计数233、占据补2。覆盖下降体现此前完整性条件不足，不能保留无依据的“提前建模收益”。1号、2号和连续侧视仍没有首次模型，空场没有有效预测。</p><h2>未来位置误差</h2><p>'''+html.escape(outcome)+'''</p>'''+tab([dict(before,method='原计数模型'),dict(uncorrected,method='边界修正前独立模型'),dict(now,method='边界守门后')],[('method','方法'),('paired','共同端点'),('mean_px','平均误差/px'),('median_px','中位误差/px')])+'''
<p>端点与起点严格沿用128全部共同评分对。129规则在128评分和视频核验后提出，因此这是开发诊断，不是独立测试；没有按误差调节边界或阈值。边界守门改变了初始化时刻和所学形状/图像偏移，当前滤波未能保证更好的未来框估计；这需要后续拆分形状学习、坐标参考与动力学误差，不能靠保留越界观测“恢复成绩”。框中心不是真COM，位置误差不能当姿态/转速精度。历史完整实验见<a href="../stage128_RAW_independent_support_20261023/桨盘提取与独立模型补位实验报告.html">124—128报告、完整预测视频与失败结果</a>。</p>
<h2>实际RAW验证范围</h2><p>128完成790时刻实际RAW、原69/70路等价、活动前缀和负例；129追加60时刻正例和30时刻空场的实际RAW小试。新增守门结果逐项等于独立129重放，原71路逐项保持128，事件cutoff正确，worker不读旧检测或标注。129没有重新做完整8秒端到端性能实验；400时刻覆盖/未来评分属于同期缓存当前观测重放，不能混称完整新部署。</p><p>129使用同一worker内插入边界守门，减少进程额外占用。前一次会话中断与一次内存资源失败均保留日志，随后轻量串行重跑完成；没有修改系统设置或其他应用。默认稳定入口保留，实验入口为run_FOV_guard_forecast.py，输入仍只需RAW和已有权重。</p>
<h2>现在真正剩下的卡点</h2><p>需要区分“桨盘高速旋转的双极事件”与“机体运动产生的双极边缘”，并处理相互重叠和画面截断。仅增大支持或候选数量不足以解决它。侧视不足四盘时，模板可以约束历史推断，但无法把缺失边界变成当前测量。下一步应验证ROI内逐像素时间周期对桨盘/机体边缘的区分，再用连续片段检验建模、退化和恢复；事件数量不直接等于转速或推力。</p><p>目前实现也远超20ms预算；128清晰完整研究对照路径中位约884ms。此轮先修观测完整性，没有证明实时性、独立RPM收益或创新成立。</p>
<p><a href="scores129.json">同组未来评分</a> · <a href="availability129.json">完整失败分母</a> · <a href="protocol129.json">边界规则冻结与事后诊断范围</a> · <a href="raw/prefix_natural6/checks129.json">60时刻实际RAW核验</a> · <a href="raw/negative5/checks129.json">空场实际核验</a> · <a href="../../event_drone_forecast/research/stage129/README.md">复现入口</a></p></html>'''
    if (OUT/'guarded_startup129.mp4').exists():page=page.replace('<h2>现在真正剩下的卡点</h2>','<h2>边界修正的实际RAW小试</h2><video controls preload="metadata" src="guarded_startup129.mp4" poster="guarded_startup129.png"></video><p>60时刻实际RAW重绘：灰为原计数0.4秒预测，绿为守门支路；红色表示拟合边界越出视野，仍保留为部分输入。未读未来框，50fps只代表回放速度。</p><h2>现在真正剩下的卡点</h2>')
    path=OUT/'桨盘边界完整性修复与最终结论.html';path.write_text(page,'utf-8')
    save(OUT/'delivery129.json',dict(report=str(path),outcome=outcome,guarded_current=clear['current'],guarded_first=clear['first_current'],apparent_early_gain_retracted=True,
        old_score=before['mean_px'],uncorrected_score=uncorrected['mean_px'],guarded_score=now['mean_px'],real_COM_or_generalization_proven=False,actual129_RAW_pilot90=True,full8s129_end_to_end=False,default_promoted=False))
    oldreport=OLD/'桨盘提取与独立模型补位实验报告.html';text=oldreport.read_text('utf-8');banner='<p class="note">后续视频复核已在129发现完整性错误：本页236→244与22.68秒更早初始化包含视野外拟合，不再视为可靠收益。<a href="../stage129_FOV_guard_20261024/桨盘边界完整性修复与最终结论.html">请先看边界修正后的最终结论</a>。</p>'
    if banner not in text:text=text.replace('<h2>实际修了哪一层</h2>',banner+'<h2>实际修了哪一层</h2>');oldreport.write_text(text,'utf-8')
    print(json.dumps(dict(report=str(path),outcome=outcome)),flush=True)
if __name__=='__main__':main()
