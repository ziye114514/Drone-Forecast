"""Report completed phase/native-motion batch; labels opened after inference."""
from pathlib import Path
import ast,hashlib,json,sys
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from compare_final_pipeline import reference_at,stats
from physical_publication import evaluate


def read(p):return json.loads(p.read_text('utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines()]


def main():
    out=ROOT.parent/'output/phase_speed163_20261101';base=ROOT.parent/'output/observed_force_partition_20261101'
    data={c:rows(out/f'{name}/predictions.jsonl') for c,name in [('clear','clear_candidate_final'),('side','side_final')]}
    result=dict(target=dict(horizon_s=.4,error_mean_px=30.,new_observation_FPS=30.),runs={},observations={},phase={},score_rows=[])
    for c,name in [('clear','clear_candidate_final'),('side','side_final')]:
        protocol=read(out/name/'protocol.json')
        assert protocol['source_snapshot_before_worker_start'] and not protocol['source_changed_during_run']
        for f,h in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
        result['runs'][c]=read(out/name/'summary.json')
        old=rows(base/f'final_replay/{c}_observed.jsonl');errors=[]
        for a,b in zip(data[c],old):
            assert a['t']==b['t']
            if a['body'].get('detected') and b['body'].get('detected'):
                errors.append(float(np.linalg.norm(np.asarray(a['body']['center'])-b['body']['center'])))
        result['observations'][c]=dict(max_center_change_px=max(errors,default=0.),
            detection_flags_equal=all(a['body'].get('detected')==b['body'].get('detected') for a,b in zip(data[c],old)))
        result['phase'][c]=dict(qualified_slot_results=sum(r['spatial_phase'].get('qualified_count',0) for r in data[c]),
            selected_slot_rates=sum(len(r['phase_rate_bridge']['selected']) for r in data[c]),
            reasons=dict(Counter(r['spatial_phase']['reason'] for r in data[c])),
            current_input_count=sum(r['force']['active'] for r in data[c]))
    truth=sorted(read(ROOT.parent/'output/dense_labels_side_110_1195/labels.json')['records'].values(),key=lambda r:r['t'])
    queries=read(base/'final_replay/summary.json')['scored_rows']
    for query in queries:
        q=query['query_t'];r=[r for r in data['side'] if r['t']<=q+1e-9][-1]
        assert r['valid'] and q-r['t']<=.16
        h=float(q-r['t']+.4);target=reference_at(truth,q+.4);assert target is not None
        result['score_rows'].append(dict(t=q,stage162_error_px=query['observed'],
            stage163_error_px=float(np.linalg.norm(np.asarray(evaluate(r,[h])[str(h)])-target))))
    result['side_scores']={a:stats([r[a+'_error_px'] for r in result['score_rows']]) for a in ['stage162','stage163']}
    result['integral_matching']=read(out/'integral_matching_warm.json')
    result['unique_harmonics']=read(out/'unique_harmonics.json')
    result.update(labels_used_only_for_scoring=True,cached_native_RAW_windows=True,
        fresh_E_RAW_read_this_batch=False,metric_RPM_or_thrust_measured=False,
        targets_achieved=False,clear_forecast_error_scored=False,
        source_time_scores_exclude_processing_delay=True,source_hashes_match=True,
        phase_step_test_is_conditional_bridge_not_end_to_end_latency=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    parts=['<html lang="zh-CN"><meta charset="utf-8"><title>阶段163：相位接线与匹配优化</title><style>body{font:17px/1.8 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#24374a}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #bcc8d4}pre{background:#eef3f8;white-space:pre-wrap;padding:16px}</style>',
        '<h1>朝0.4秒30px、30FPS推进：相位已接线，目标尚未实现</h1>',
        '<p>2026-11-01。仍为原生ON/OFF桨叶→五圆/部分桨/侧视→可测贡献＋不可测合力→二维预测。隐藏桨力归未知项，无G、无整机NN前置。</p>',
        '<h2>实际改了什么</h2><p>相位模块支持两槽逐步扩到四槽；单槽安全拒绝；过滤未来和重复过去区间。保留旋转方向性判据：真实旋转和共同平面摆动能通过，静止闪烁应被拒绝，上一条日志的“判据结构不可达”结论已纠正。</p>',
        '<p>当前空间相位在与当前时间谱候选相容时可选择频率，并确认较大同槽变化；不再仅因变化幅度大便撤销输入。隐藏槽不补频率。合成条件确认桥能保留600Hz阶跃引起的-600px/s²变化，但不等于整条相位模块能零延迟发现真实阶跃。</p>',
        '<p>匹配能量用积分图替代第二次FFT；六组保存输入位移相同。GPU共享谐波系数从228减到147项，九窗当前掩膜/所选频率一致，但总体耗时更慢，保留unique_harmonics=True实验选项，默认关闭。</p>',
        '<h2>本批数据</h2><table><tr><th>片段</th><th>核心FPS</th><th>中位ms</th><th>P95 ms</th><th>相位合格槽结果</th><th>实际选用次数</th></tr>']
    for c in ['clear','side']:
        s=result['runs'][c];p=result['phase'][c]
        parts.append(f'<tr><td>{c}</td><td>{s["compute_capacity_fps"]:.2f}</td><td>{s["compute_ms_median"]:.1f}</td><td>{s["compute_ms_p95"]:.1f}</td><td>{p["qualified_slot_results"]}</td><td>{p["selected_slot_rates"]}</td></tr>')
    score=result['side_scores'];m=result['integral_matching']['means_ms']
    parts+=['</table>',f'<p>固定9个侧视0.4秒查询均值：{score["stage162"]["mean"]:.3f}→{score["stage163"]["mean"]:.3f}px。清晰段没有足够未来位置参考，不编造其预测精度。</p>',
        f'<p>预热交替微测：匹配算子平均{m["False"]:.2f}→{m["True"]:.2f}ms。六组输入相隔多帧，仅证明固定输入的位移与算子成本，不是30FPS整线证明。首轮未预热的记录受首次SciPy导入影响，不用其均值。</p>',
        '<p>本批使用原始RAW已保存的完整原生事件窗口，未重新读E盘；核心FPS不含缓存解压和初始化，不是实际持续30Hz发布。新旧机器负载不同，不用不同轮次的FPS直接归因加速。图像中心与旧控制的差值详见summary.json。</p>',
        '<h2>下一步真正卡点</h2><ol><li>侧视没有可用共享平面，相位桥在该段仍不能提供逐桨转速。需要局部运动补偿、时间相位和混合源候选，不能强行恢复四桨。</li><li>相位FFT用160ms滚动窗，真实阶跃仍存在延迟；需递推相位/角速度及候选创新，不能只调方向性阈值。</li><li>GPU窗口处理和CPU原生几何/匹配仍远超33ms；下一批应对锁定频带调度、原生几何成本和有界流水线动手，而不是新增更多频谱候选。</li></ol>',
        '<p>机制参考：<a href="https://arxiv.org/abs/2603.09235">HelixTrack</a>；性能路径参考<a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html">NVIDIA官方最佳实践</a>。本地独立实现，没有复制作者代码。</p>',
        '<p><a href="summary.json">全部数值</a> · <a href="phase_integration_checks.json">相位机制检查</a></p></html>']
    (out/'index.html').write_text('\n'.join(parts),'utf-8')
    names=['shared_phase_fft.py','phase_rate_bridge.py','rotor_thrust_bank.py','rotor_support_front.py','rotor_cuda_device.py','run_rotor_only.py','run_fast_structure.py','tools/check_phase_integration163.py','tools/report_phase_speed163.py','tools/probe_integral_matching.py','tools/prepare_github_release.py']
    for name in names:ast.parse((ROOT/name).read_text('utf-8-sig'))
    print(json.dumps({k:result[k] for k in ['runs','observations','phase','side_scores']},ensure_ascii=False))


if __name__=='__main__':main()
