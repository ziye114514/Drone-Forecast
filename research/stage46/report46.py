import html,json
import numpy as np
from backend46 import *
def main():
    verify();b=load(OUT/'benchmark_scoped_path.json');seq=OUT/'sequence';a=load(seq/'analysis.json');old=load(common44.INPUT/'analysis.json');ref={(r['case'],r['frontend'],r['model'],r['nominal_h']):r for r in old['table']};differences=[];table=[]
    for r in a['table']:
        q=ref[r['case'],r['frontend'],r['model'],r['nominal_h']]
        for key in ['valid_forecasts','joint_success','absent_forecasts_inside','known_negative_published_anchors']:
            if r[key]!=q[key]:differences.append(dict(case=r['case'],method=r['frontend'],model=r['model'],h=r['nominal_h'],field=key,new=r[key],old=q[key]))
        if r['frontend']=='body_first' and r['nominal_h']==.4 and r['model'] in ['cv','trusted_acc'] and r['case']!='negative5':table.append(dict(case=r['case'],model=r['model'],valid=r['valid_forecasts'],targets=r['present_targets'],original_mean=q['mean_fde_px'],fp16_mean=r['mean_fde_px'],success=r['joint_success']))
    assert not differences;assert not b['failures'];original={(r['case'],r['t']):r for r in map(json.loads,(common44.INPUT/'detections.jsonl').read_text(encoding='utf-8').splitlines()) if r['method']=='finetuned'};n=same=0;maximum=0.
    for r in map(json.loads,(seq/'detections.jsonl').read_text(encoding='utf-8').splitlines()):
        if r['method']!='finetuned':continue
        q=original[r['case'],r['t']];n+=1;same+=len(r['boxes'])==len(q['boxes'])
        if len(r['boxes'])==len(q['boxes']):
            for x,y in zip(r['boxes'],q['boxes']):maximum=max(maximum,float(abs(np.array(x['bbox'])-y['bbox']).max()))
    assert n==1550 and same==1550
    save(OUT/'quality_regression.json',dict(frames=n,per_frame_box_count_same=same,max_retained_bbox_coordinate_difference_px=maximum,all_horizons_all_models_coverage_joint_success_absent_equal=True,finite_raw_output_assertion=True,table=table,limitations='Same previously seen data; precision conversion quality preservation, not new generalization or predictive improvement. cuDNN frontend emitted unsupported-plan messages during setup, execution completed; keep compatibility note.'))
    timing=[dict(scope=scope,method=method,n=v['n'],median_ms=v['median_ms'],p95_ms=v['p95_ms']) for scope,methods in b['times'].items() for method,v in methods.items()]
    def tab(rr):
        cols=list(rr[0]);return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[x],5) if isinstance(r[x],float) else r[x]))+'</td>' for x in cols)+'</tr>' for r in rr)+'</table>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第四十六阶段FP16图执行验证</title><style>body{font:16px/1.8 Arial;max-width:1300px;margin:30px auto;color:#233e4c}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border:1px solid #ccd8df}</style><h1>第四十六阶段：标准16位候选保留1550时刻质量，速度收益需分清范围</h1><p>40训练权重不变，44 ONNX经已安装ORT转换器做固定FP16内部转换，IO保持FP32；无重训练/标签校准/混精搜索。相同batch1/384×640/conf.25，CUDAgraph固定地址。配对12图5次交替，FP32graph27.67ms、FP16graph21.96ms，约1.26倍；PyTorch51.59ms。这里包含预处理/模型/NMS，排除RAW/几何/IPC。<b>不是整帧20ms实时证明。</b></p>'''+tab(timing)+'''<p>单独持续1550图的模型链中位10.16ms，与多后端交替时钟/同步组织不同，不能拿10.16除以配对PyTorch51.59冒充同条件5倍。后续47直接读RAW测父子通信/完整或单位置模式。</p><h2>完整当前与未来质量</h2><p>先全部1550新神经输出，全部raw有限值检查，再原41按时间回放及稀疏框评分。逐帧框数1550一致；全部时距六模型的有效数、联合成功、无机输出统计与41保持。无机400时刻全零。未来均值有小浮点变化，不能称预测算法改善。</p>'''+tab(table)+'''<p>3号当前仍12/12，0.4秒11/12可预报且收缩约110px、成功2/12。此处不验证姿态、机械RPM或真实推力，更不证明跨未知机型泛化。质量比对数据此前开发见过。</p><h2>保留的修复与兼容性</h2><p>原转换器把input Cast放在首Conv后，checker先拒绝，保留未排序副本。补充协议后用ORT官方topological_sort，仅调node顺序，节点多重集/权重/边/IO逐项不变。原FP16工厂宽路径锚点两处触发初始化前断言，benchmark.json保留；构造器限定的scoped helper和benchmark_scoped_path.json才是真候选。未改阈值/数据以通过。</p><p>初始化出现cuDNN frontend“no execution plans”提示；最终图执行配置会话、换图/回放、有限值与1550测试完成。这仍是兼容性限制，不宣称所有设备无警告可部署，旧默认不改。</p><p>复用<a href="https://onnxruntime.ai/docs/performance/model-optimizations/float16.html">ORT官方16位方案</a>和现成图执行，不称创新算法。<a href="quality_regression.json">质量回归</a> · <a href="sequence/analysis.json">完整未来评分</a> · <a href="benchmark_scoped_path.json">真正配对计时</a> · <a href="protocol.json">冻结协议</a></p></html>'''
    (OUT/'第四十六阶段半精度图执行验证报告.html').write_text(text,encoding='utf-8');(OUT/'阶段结论.md').write_text('同进程FP32graph27.67→FP16graph21.96ms约1.26x，1550框数和全部时距六模型覆盖/联合成功保持。未来误差微小浮点差，非精度改善。初始化兼容提示和修复保留，未替换旧默认。',encoding='utf-8');print('Stage46 quality/report complete',flush=True)
if __name__=='__main__':main()
