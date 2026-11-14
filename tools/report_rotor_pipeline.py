"""Open existing references only after predictions are saved; report small pilot."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from responsive_force import DampedResponsiveForce
from physical_publication import evaluate
from compare_final_pipeline import score,reference_at,stats
from rotor_only_front import filtered_rgb


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
    runs={name:out/name.replace('_final','_bank') for name in ['clear_final','side_final','empty_final']}
    data={};checks={};ablation={}
    for name,folder in runs.items():
        if not (folder/'summary.json').exists():continue
        data[name]=read(folder/'summary.json');r=rows(folder/'predictions.jsonl')
        checked=0
        for f in (folder/'images').glob('*_evidence.npz'):
            with np.load(f) as z:
                for suffix,key in [('density','density'),('period','period'),('native_arcs','native_arcs')]:
                    image=cv2.imread(str(f.with_name(f.name.replace('_evidence.npz',f'_{suffix}.png'))))
                    assert image is not None and not np.any(image[~z[key]])
                assert not np.any(z['native_arcs']&((z['on']+z['off'])==0))
                checked+=1
        checks[name]=dict(saved_masks_zero_outside=True,native_pixels_have_actual_events=True,
            evidence_frames_checked=checked,no_NN_used=all(not z['whole_body_NN_used'] for z in r))
        control=DampedResponsiveForce();without=[];replayed=DampedResponsiveForce();delta=0.
        for row in r:
            body=row['body'];c=row['coefficient'];position=body.get('center') if body['detected'] else None
            _,f=control.step(row['t'],c,position,None,0.)
            muted=dict(row,force=f);muted['physical']=evaluate(muted,[.1,.2,.4]) if c else {};without.append(muted)
            replayed.input_learning_current=row.get('period_selection',{}).get('learning_input_current',True)
            u=row['force'].get('input_u');_,original=replayed.step(row['t'],c,position,u,.5 if u is not None else 0.)
            values=evaluate(dict(row,force=original),[.1,.2,.4]) if c else {}
            for h in values:delta=max(delta,float(np.max(abs(np.array(values[h])-row['physical'][h]))))
        checks[name]['force_replay_max_delta_px']=delta
        assert delta<1e-7
        (folder/'without_period_input.jsonl').write_text('\n'.join(json.dumps(z) for z in without)+'\n','utf-8')
        if name=='side_final':
            labels=read(ROOT.parent/'output/dense_labels_side_110_1195/labels.json')
            truth=sorted(labels['records'].values(),key=lambda z:z['t'])
            ablation[name]=dict(on=score(r,truth),off=score(without,truth),
                qualification='Existing manual whole-body centers are a proxy; rotor-center definition differs. Source-time offline pilot, not actual-publication accuracy or independent test.')
            baseline_path=ROOT.parent/'output/motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl'
            baseline=rows(baseline_path)
            matched=[];missing=[]
            for query in np.arange(110.2,112.0001,.2):
                selected=[]
                for arm in [baseline,r,without]:
                    earlier=[z for z in arm if z['t']<=query+1e-9]
                    z=earlier[-1] if earlier else None
                    selected.append(z if z and z['valid'] and query-z['t']<=.16 else None)
                target=reference_at(truth,float(query+.4))
                if target is None or any(z is None for z in selected):
                    missing.append(float(query));continue
                errors=[]
                for z in selected:
                    # This existing evaluator reproduces stored baseline0.4s.
                    assert np.max(abs(np.array(evaluate(z,[.4])['0.4'])-z['physical']['0.4']))<1e-7
                    horizon=float(query-z['t']+.4)
                    errors.append(float(np.linalg.norm(np.array(evaluate(z,[horizon])[str(horizon)])-target)))
                matched.append(dict(query_t=float(query),target_t=float(query+.4),
                    old_body_baseline_px=errors[0],rotor_period_on_px=errors[1],rotor_period_off_px=errors[2]))
            ablation[name]['fixed_source_queries']=dict(rows=matched,missing_query_times=missing,
                scores={key:stats([x[key] for x in matched]) for key in
                    ['old_body_baseline_px','rotor_period_on_px','rotor_period_off_px']},
                actual_compute_or_publication_time_not_evaluated=True)
    clear=runs['clear_final']/'images/0008_evidence.npz'
    if clear.exists():
        with np.load(clear) as z:
            panels=[]
            for mask,title in [(np.ones_like(z['period']),'Raw ON/OFF'),(z['period'],'Strict periodic pixels'),(z['native_arcs'],'Native rotor contours')]:
                image=filtered_rgb(z['on'],z['off'],mask)[0:190,780:1130,::-1]
                image=cv2.resize(image,(525,285),interpolation=cv2.INTER_NEAREST)
                canvas=np.zeros((320,525,3),np.uint8);canvas[35:]=image
                cv2.putText(canvas,title,(10,24),0,.6,(255,255,255),1);panels.append(canvas)
            cv2.imwrite(str(out/'filter_comparison.png'),np.hstack(panels))
    summary=dict(runs=data,checks=checks,source_force_ablation=ablation,
        independent_RPM_or_Newton_thrust_proven=False,all_scene_rotor_precision_proven=False,
        no_actual_publication_30Hz_claim=True,stable_prediction_gain_proven=False,
        integration_status='optional_RAW_route_integrated_original_baseline_preserved')
    (out/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    lines=[]
    for name,s in data.items():
        lines.append(f'<tr><td>{name}</td><td>{s["current_rotor_detections"]}/{s["frames"]}</td><td>{s["valid_forecasts"]}</td><td>{s["current_force_inputs"]}</td><td>{s["compute_capacity_fps"]:.2f}</td><td>{s["frames"]/s["end_to_end_elapsed_s"]:.2f}</td><td>{s["modes"]}</td></tr>')
    scores=''
    if 'side_final' in ablation:
        a=ablation['side_final'];on=a['on']['physical_future_error_px']['0.4'];off=a['off']['physical_future_error_px']['0.4']
        scores=f'<p>同位置输入、同源起点、同未来参考，侧视0.4秒：关闭周期输入 {off["mean"]:.2f}px → 开启 {on["mean"]:.2f}px（各{on["n"]}个起点）。这是已有人工整机中心代理的离线小试，不能当桨叶中心真值或实际输出精度。</p>'
        fixed=a['fixed_source_queries']['scores']
        scores=f'<p class="note"><b>固定查询主对照：当前没有稳健预测收益，原基线保留。</b>固定0.2秒查询网格并精确重算源年龄，共同{fixed["rotor_period_on_px"]["n"]}点：旧整机基线{fixed["old_body_baseline_px"]["mean"]:.2f}px，新桨叶入口开启/关闭周期{fixed["rotor_period_on_px"]["mean"]:.2f}/{fixed["rotor_period_off_px"]["mean"]:.2f}px。不计实际计算延迟。下面11个按源起点抽样的诊断均值改善未在固定查询复现，收益对取样敏感，不能只报较好的结果。</p>'+scores
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶入口接入预测</title><style>body{font:17px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#23374b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #bbb;padding:8px}img,video{max-width:100%}.note{padding:16px;background:#fff0d2}pre{white-space:pre-wrap;background:#eef3f7;padding:14px}</style>
    <h1>原生ON/OFF桨叶入口 → 原弧 → 几何 → 预测</h1>
    <p>2026-11-01。新的可选完整入口，保留原整机检测基线。不读NN权重、不读G重建、不用真值裁剪。视频和以下精度按源时间展示，尚未实时。</p>
    <h2>实际过滤图</h2><img src="filter_comparison.png">
    <p>左：原始ON/OFF；中：严格周期确认像素；右：由周期支持引导恢复的原生ON/OFF连通轮廓。背景真正清零。右图增加的像素来自当前原始事件，并非全都通过周期检验，也不是人工证明的纯桨叶真值。</p>
    <h2>小试结果</h2><table><tr><th>片段</th><th>当前桨叶区域</th><th>有效源预测</th><th>当前代理力输入</th><th>核心计算FPS</th><th>缓存回放含导出FPS</th><th>状态计数</th></tr>'''+''.join(lines)+'''</table>
    <p class="note">计算能力不等于持续有效30Hz。此原型仍然过慢，源时间0.4秒预测不能当实际输出后0.4秒的精度。候选区域检出数量不是有独立真值的召回率；数据包括不足一个完整窗口的末包。</p>
    <h2>预测对照</h2>'''+scores+'''<p>频率平方只作可见区域的相对推力代理，法线正负使用图像向上先验；没有测到四桨真实总推力、隐藏桨RPM或风力。沿用共享位置状态与因果到期残差修正，不增加传统并行预测器。</p>
    <h2>清晰段视频</h2><video controls src="clear_bank/filtering.mp4"></video>
    <h2>侧视段视频</h2><video controls src="side_bank/filtering.mp4"></video>
    <p>四格依次为原始、密度、严格周期、原生轮廓。右下黄色线为预测轨迹，十字为源时刻+0.4秒预测；不显示未来真值。</p>
    <h2>具体实现</h2><ul><li>增加--rotor-only主入口分支，在读取NN模型/启动整机检测前转到run_rotor_only.py。</li>
    <li>周期种子选择原生ON/OFF连通成分，回到原事件像素做几何，不拿周期掩膜的人工边界当桨盘边缘。</li>
    <li>位移匹配仅用筛选后的ON/OFF图。四桨/三桨/部分可见/线状侧视走已有五圆与轴分支，证据不足降级，连通块数不等于桨数。</li>
    <li>CUDA复用predictive_harmonics_cuda，经常驻进程共享内存与SDK Python连接；超容量回CPU，不丢事件。空间运算复用OpenCV。</li></ul>
    <h2>运行</h2><pre>python run_fast_structure.py --rotor-only --raw "E:\\dataset\\train\\6\\media\\event.raw" --output "新的目录" --start 26.2 --duration 1.2
    # 可选 --gpu-python "已安装CUDA Torch的Python路径"；省略则CPU参考。</pre>
    <h2>尚未完成</h2><p>排除其他周期物体、稳定真实桨数识别、可靠RPM、持续有效30Hz和25px均未证实。原型接通不等于所有预测更准；不覆盖旧默认模型，也未更新GitHub ZIP。</p>
    <p>复用来源：<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_propagation.html">SciPy形态重建</a>、<a href="https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html">OpenCV连通域和形状</a>；没有复制新第三方代码。</p>
    <p><a href="final_summary.json">全部结果和检查</a> · <a href="../rotor_support_20261101/index.html">先前独立前端试点</a></p></html>'''
    (out/'index.html').write_text(html,'utf-8')
    print(json.dumps({k:{m:v for m,v in s.items() if m in ['frames','modes','compute_capacity_fps']} for k,s in data.items()}))


if __name__=='__main__':main()
