"""Report frozen natural extension, including failures and sparse target coverage."""
from pathlib import Path
import html,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage123')]
from study123 import OUT,CASES,METHODS,verify,rows,save
from evaluate123 import state

def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def table(rr,fields):
    return '<table><tr>'+''.join('<th>'+html.escape(label)+'</th>' for key,label in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r.get(k):.3f}' if isinstance(r.get(k),float) else str(r.get(k,'')))+'</td>' for k,_ in fields)+'</tr>' for r in rr)+'</table>'

def regression():
    # New natural control still preserves the original64 in the overlapping
    # old118 window, even though the new structural branches see no dropout.
    current=rows(OUT/'raw/natural6/observations_forecasts.jsonl');prior=rows(ROOT.parent/'output/stage118_RAW_learned_structure_20261022/raw/clear6/observations_forecasts.jsonl')
    names=set()
    for a,b in zip(current,prior):
        qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback'];assert a['body_observation']==b['body_observation']
        for key in ['coefficients','predictions']:
            for name,value in qb[key].items():
                if name in ['model_free20_kf118','model_fixed_kf118','model_fixed_guarded_kf118']:continue
                assert qa[key][name]==value,(a['t'],key,name);names.add(name)
    assert len(names)==64
    save(OUT/'original64_regression123.json',dict(overlap_frames=len(prior),all_original64_exact118=True,current_body_exact=True,new_models_naturally_unmasked=True))

def main():
    verify();regression();availability=load(OUT/'availability123.json');scored=load(OUT/'scores123.json');summary=[]
    for a in availability:
        row=dict(case=a['case'],frames=a['frames'],body_current=a['body_current'],valid_forecast=a['valid_forecast'],candidates4=a['rotor_count_histogram'].get('4',0),candidates3=a['rotor_count_histogram'].get('3',0),validation_updates=a['validation_updates'],transitions=a['n_complete_natural_transitions'])
        m=next(m for m in a['methods'] if m['mode']=='validated122');row.update(first_structure=m['first_structure'],geometry_current=m['current'],partial3_current=m['partial3_current']);summary.append(row)
    scores=[r for r in scored['table'] if r['h']==.4];positive=[r for r in summary if r['case']!='negative5'];complete=sum(r['transitions'] for r in positive);extension_models=sum(r['first_structure'] is not None for r in positive if r['case'].startswith('extension'))
    videos=load(OUT/'video_checks123.json');checks=[load(OUT/'raw'/c/'checks123.json') for c in CASES]+[load(OUT/'raw/prefix_natural6/checks123.json')]
    # Keep literal findings derived from the complete denominator.
    outcome=f'两段扩展录像中有{extension_models}/2段完成首次结构学习；三段正例共记录{complete}次满足预先定义的自然三桨持续至少100ms、随后四桨恢复过程。'
    if complete==0:outcome+='本轮没有获得可证明自然缺桨恢复收益的完整过程，不能把先前人工删特征的成功当成自然遮挡成功。'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>123：自然退化扩展与未来框预测</title><style>body{max-width:1350px;margin:30px auto;padding:0 22px;font:17px/1.8 system-ui;color:#243947}img,video{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #cbd5df}th{background:#eaf3f4}.note{padding:15px;background:#fff0cf}a{color:#087c87}code{background:#edf2f5}h2{margin-top:2em}</style><h1>123：冻结方法，检查自然缺桨是否真的能用</h1>
<p>本轮算法仍是122，没有增加新的拟合阈值、初始化捷径或预测参数。三段输入：1号40—46秒、2号40—46秒、6号22—30秒；关闭所有人为特征删除，使用完整RAW和既有检测权重。另做1号独立2秒截断、6号1.2秒活动模型截断和0.6秒已知无目标负例，共1190个实际20ms父时刻。</p><p class="note">'''+html.escape(outcome)+'''本报告分别统计观测覆盖、模型建立、分支实际使用和未来框误差；选择器标记为conic3不等于当时存在有效几何观测。</p>
<h2>当前进度与首次建模瓶颈</h2>'''+table(summary,[('case','片段'),('frames','时刻'),('body_current','当前NN'),('valid_forecast','有效预测'),('candidates4','四候选'),('candidates3','三候选'),('first_structure','首次结构/s'),('geometry_current','当前几何'),('partial3_current','当前三桨更新'),('validation_updates','自检次数'),('transitions','完整自然恢复')])+'''
<img src="natural_coverage123.png"><p>分母包含所有固定时刻，不因看不清/未建模/未发布而删除。当前候选数是算法输出，不是人工可见桨盘数；“四候选”不证明四个都是真桨盘。“完整自然恢复”要求已锁定、连续至少五个20ms时刻有三个真实当前观测及有效选中几何，并在同一机体身份下恢复四桨。</p>
<h2>已有人工事件框：先核对当时能不能看见</h2>'''+table(scored['label_checks'],[('case','片段'),('present_checked','有目标稀疏时刻'),('NN_detected_at_present','当时NN检出'),('geometry_current_at_present','当时几何有效'),('median_bbox_iou_including_misses','含漏检的中位IoU'),('median_target_width_height','目标中位宽高/px'),('absent_checked','无目标稀疏时刻'),('valid_at_absent','无目标时仍有预测')])+'''
<p>标签来自项目既有portable_annotations人工事件框，由prepare_evaluation.py按1280×720事件坐标导出；没有插值制造连续真值。它们评估框和框中心，不是桨心、实际重心、姿态或转速。IoU仅作事后诊断，没有用来调整或筛选推理。</p>
<h2>未来框中心预测：同一组发布时刻公平比较</h2>'''+table(scores,[('case','片段'),('method','方法'),('eligible_present_targets','可用有目标端点'),('paired_targets','共同发布端点'),('mean_px','平均误差/px'),('median_px','中位误差/px'),('actual_horizon_min','实际最短h/s'),('actual_horizon_max','实际最长h/s')])+'''
<img src="sparse_forecast123.png"><p>只在三段预测、实际截断、负例全部完成后才读取标签坐标。对每个目标，选target−h之前最近的有效发布时刻，时间差不超过20ms；五方法用同一时刻，按实际target−origin外推。名义0.4秒实际可能为0.400—0.420秒，范围见表。未发布单独计入覆盖，没有当成零误差或从分母隐藏。</p>
<p>上一轮“11.52→2.26px”是删特征后与完整模型的一致性；本表是未来人工框中心误差，两个指标不能直接比较。若没有学到结构，几条结构分支可能数值相同，因为都在使用机体位置历史，而非证明几种几何方法一样准确。清晰正例属于已开发录像，少量稀疏端点不足以证明泛化或创新优势。</p>
<h2>原生ON/OFF检查</h2><img src="native_visibility_contact123.png"><p>红为ON、青为OFF、黑为两种事件均出现，每张仅20ms，无灰度重建G、椭圆叠加或补出的桨心。为显示NN漏掉的目标，部分裁剪事后使用最近的人工框；这只影响报告展示，未回流到推理。</p>
<h2>2号片段提取链路：候选在模型之前已碎化</h2><img src="extraction_funnel123.png"><p>对相同300个RAW时刻重新记录阈值、形态学、候选和分组，最终桨盘输出逐项精确匹配原122运行。43.5秒只有两个通过候选筛选的小椭圆，尺寸约12.17×6.63和11.39×4.41像素，分组未发布；45秒三个候选约20.38×11.35、18.39×7.11、14.24×6.06像素，最后只保留两个。报告中的彩色强度是事件计数密度，不是G重建。</p><p>原生ON/OFF的“黑”只表示两极性都出现，不表示每个像素的重复事件次数足够高。当前20ms计数归一到10ms后，双极最小值、固定最低阈值2.0和开闭运算可能只保留高计数内核。该控制支持“事件支持碎化、候选/分组先于姿态失败”的判断；没有独立桨盘边界标签，不能声称已确定每个真实桨盘的损失率，也没有按这两帧降低阈值重新宣称成功。</p>
<h2>实际RAW预测视频</h2>'''
    for case in ['extension1','extension2','natural6']:
        page+=f'<h3>{case}</h3><video controls preload="metadata" poster="{case}_poster123.png" src="{case}_validated123.mp4"></video>'
    page+='''<p>绿：自检分支；橙：三椭圆；灰：纯机体中心/速度KF；青：当前候选。没有当前结构时曲线可能重合。视频源50fps是离线回放，计算仍不满足20ms实时预算。</p>
<h2>检验范围与保留的失败</h2><p>历史元数据清点显示：1号已知使用30—35秒但有未知历史范围；2/3号0—60秒、4/6号0—120秒过去已被早期系统扫描。因此本轮称冻结参数的扩展验证，不称盲测、未见目标或检测器训练独立测试。区间按可读40秒和固定开发正例确定，没有按未来误差选取。</p><p>首次并行启动三段均在Torch导入shm.dll时触发Windows1455内存提交/分页限制，预测文件均0字节；原目录和日志保留后改为逐段运行，没有改系统内存设置、算法参数或权重。这是执行资源失败，与本报告实际跑完后保留的检测/几何失败分开记录。</p>
<p>原122源文件指纹保持；自然正例前250时刻的原64路预测/系数精确118，当前机体观测一致。1号独立100时刻截断没有检测/有效预测，因此又补充6号60时刻截断，核验已初始化69路的全部已发布预测、活动模型/自检内部状态与完整运行前缀相同。该补充在主评分后进行，只核验因果性，不新增调参或评分目标。两前缀事件字段SHA相同，负例无有效预测，事件cutoff、零人为删桨及视频全帧解码通过。并发失败不用于任何速度比较。</p>
<p>补充前缀最初直接调用旧运行入口，60帧已生成后，旧进程级审计回调引用已清理的全局变量，导致读核验文件时报错。保留原辅助脚本并改成子进程CLI，现有60帧在全新检查进程通过，无需重跑或移除审计。活动前缀包含19次当前几何、17次自检和69路初始化。<a href="positive_prefix_wrapper_repair123.json">辅助工具修正记录</a>。</p>
<h2>下一步优先级</h2><p>现在应先改善“首次可靠建模的覆盖”，再扩大选择器的创新主张。先对照原生图确认漏检、桨盘合并、轮廓失真、四圆拟合不一致分别占多少；对真实看得见四桨的帧，优先恢复正确事件支持而不是降低资格门槛。之后在同一连续录像学到结构，再验证自然三桨和恢复。若当前数据没有足够可分辨的桨盘，优先选自采近距离片段，增加小目标FRED不能自动补足几何信息。</p><p>创新候选仍是122的逐桨隐藏模型适配性自检，但本轮不会把“代码存在”当成“自然实飞收益已证明”。既有研究与代码边界见<a href="../stage122_RAW_validated_structure_20261023/第一百一十九至一百二十二阶段缺桨与模型自检报告.html">119—122报告</a>。下一次改动应保持本轮区间和评分作回归，另留独立片段验证。</p>
<p><a href="scores123.json">全部0.1/0.2/0.4秒逐点评分</a> · <a href="availability123.json">所有失败原因和自然恢复统计</a> · <a href="historical_exposure123.json">历史数据使用清点</a> · <a href="startup_failure_audit123.json">启动资源失败记录</a> · <a href="protocol123.json">运行前冻结方案</a></p></html>'''
    report=OUT/'第一百二十三阶段自然退化扩展与未来框预测报告.html';report.write_text(page,encoding='utf-8')
    save(OUT/'delivery_checks123.json',dict(report=str(report),actual_RAW_parent=sum(r['frames'] for r in checks),no_parameter_changes=True,no_feature_dropout=True,
        actual_prefix_extension100_and_active60=True,original64_overlap250_exact=True,source_preview_decoded=sum(r.get('decoded_frames',0) for r in checks),overlay_decoded=sum(r['frames'] for r in videos),
        extension_recordings_with_model=extension_models,complete_natural_transitions=complete,summary=summary,real_COM_accuracy_proven=False,default_promoted=False,
        outcome=outcome))
    print(json.dumps(dict(report=str(report),outcome=outcome)),flush=True)

if __name__=='__main__':main()
