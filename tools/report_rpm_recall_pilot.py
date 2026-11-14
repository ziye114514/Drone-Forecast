"""Write the bounded EV-Tach/positive-recall experiment, including failures."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from score_fred_transfer import reference
from prepare_fred_transfer import parse
from compare_final_pipeline import stats,score as manual_score


def read(path):return json.loads(Path(path).read_text('utf-8'))
def records(path):return [json.loads(x) for x in Path(path).read_text('utf-8').splitlines()]


def main(output):
    out=Path(output);base=ROOT.parent/'output/stability_audit_20261031';summary={}
    configs={'FRED7':('FRED7_final_scores.json','fred_transfer_20261029/unseen7/input/7/coordinates.txt'),
             'FRED9':('FRED9_budget_scores.json','stability_audit_20261031/holdout9/input/coordinates.txt')}
    table=[];wall=[]
    for name,(oldfile,lab) in configs.items():
        old=read(base/oldfile)['candidate'];new=read(out/(name+'_final_scores.json'))
        folder=out/f'RAW_{name}_recall_final4s';rows=records(folder/'predictions.jsonl');by_time={round(r['t'],6):r for r in rows}
        labels=parse((ROOT.parent/'output'/lab).read_text('utf-8-sig'));pairs=[]
        for a in old['targets']:
            b=by_time.get(round(a['t'],6))
            if not a['available'] or b is None or not b['valid'] or '0.4' not in b['physical']:continue
            gt=reference(labels,a['t']+.4)
            if gt is not None:pairs.append((a['error_px'],float(np.linalg.norm(np.asarray(b['physical']['0.4'])-(gt[:2]+gt[2:])/2))))
        matched=dict(n=len(pairs),old=stats([a for a,b in pairs]),new=stats([b for a,b in pairs]))
        rt=read(folder/'summary.json');pub=read(out/(name+'_final_common_publication.json'))
        summary[name]=dict(control={k:v for k,v in old.items() if k not in ['anchors','targets']},candidate={k:v for k,v in new.items() if k not in ['anchors','targets']},same_old_source_times=matched,wall_clock=pub,runtime=rt)
        table.append(f"<tr><td>{name}</td><td>{old['current_outputs']}/{old['annotation_source_ticks']} → {new['current_outputs']}/{new['annotation_source_ticks']}</td><td>{100*old['IoU50_fraction_including_misses']:.1f}% → {100*new['IoU50_fraction_including_misses']:.1f}%</td><td>{old['error_04s_px']['mean']:.2f} → {new['error_04s_px']['mean']:.2f}</td><td>{old['predicted_target_slots']}/{old['target_slots']} → {new['predicted_target_slots']}/{new['target_slots']}</td></tr>")
        m=pub['matched'];wall.append(f"<tr><td>{name}</td><td>{m['n']}</td><td>{m['control_error_px']['mean']:.2f} → {m['candidate_error_px']['mean']:.2f}</td><td>{rt['timing_ms']['completion_lag_ms']['p95']:.2f}ms</td><td>{rt['publication']['valid_outputs']}/{rt['publication']['outputs']}</td></tr>")
    side=out/'RAW_side_recall_final8s'
    if (side/'summary.json').exists():
        truth=sorted(read(ROOT.parent/'output/dense_labels_side_110_1195/labels.json')['records'].values(),key=lambda r:r['t'])
        s=manual_score(records(side/'predictions.jsonl'),truth)
        summary['side']=dict(score=s,runtime=read(side/'summary.json'))
    clear=out/'RAW_clear_recall_final2s'
    if (clear/'summary.json').exists():summary['clear']=read(clear/'summary.json')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True);manifest={}
    files=['run_fast_structure.py','live_worker.py','rotor_icp_speed.py','tools/prepare_rpm_recall_pilot.py','tools/prepare_recall_windows.py','tools/probe_recall_zoom.py','tools/probe_recall_native_window.py','tools/score_recall_windows.py','tools/probe_rotor_icp_speed.py','tools/score_common_publication.py','tools/report_rpm_recall_pilot.py']
    for name in files:
        src=ROOT/name;data=src.read_bytes();ast.parse(data.decode('utf-8-sig'));dst=snapshot/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data);manifest[name]=hashlib.sha256(data).hexdigest()
    (snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2),'utf-8')
    summary['limitations']=['Development pilot, not a blind official benchmark','Offline frame comparison is not a live FPS test','No independent mechanical RPM or Newton thrust truth','Actual-wall-time FRED9 accuracy regresses on common targets','Original realtime30 control and publication ZIPs preserved']
    (out/'final_summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    # Visible source-time forecasts, using the same current native20ms image.
    name='FRED9';oldrows=records(base/'RAW_FRED9_budget4s/predictions.jsonl');newrows=records(out/'RAW_FRED9_recall_final4s/predictions.jsonl')
    labels=parse((ROOT.parent/'output'/configs[name][1]).read_text('utf-8-sig'))
    video=cv2.VideoWriter(str(out/'FRED9_prediction_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30.,(1200,420))
    for k,(a,b) in enumerate(zip(oldrows,newrows)):
        im=np.load(out/'frames'/name/f'{k:04d}.npz')['image'];gt=reference(labels,a['t']);future=reference(labels,a['t']+.4)
        center=(gt[:2]+gt[2:])/2 if gt is not None else np.array([640.,360.]);lo=np.clip(center-[250,175],[0,0],[780,370]).astype(int)
        panels=[]
        for row,title in [(a,'Original realtime30'),(b,'Recall30 candidate')]:
            image=im.copy()
            if gt is not None:cv2.rectangle(image,tuple(np.rint(gt[:2]).astype(int)),tuple(np.rint(gt[2:]).astype(int)),(0,255,0),1)
            if row['body'].get('detected'):
                box=np.rint(row['body']['bbox']).astype(int);cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(255,0,255),1)
            if row['valid'] and row['physical']:
                points=[row['coefficient']['position']]+[row['physical'][str(h)] for h in (.1,.2,.4)]
                cv2.polylines(image,[np.rint(points).astype(np.int32)],False,(0,220,255),2)
                cv2.drawMarker(image,tuple(np.rint(points[-1]).astype(int)),(0,220,255),cv2.MARKER_TILTED_CROSS,9,1)
            if future is not None:cv2.drawMarker(image,tuple(np.rint((future[:2]+future[2:])/2).astype(int)),(0,255,0),cv2.MARKER_CROSS,11,1)
            crop=image[lo[1]:lo[1]+350,lo[0]:lo[0]+500];crop=cv2.resize(crop,(600,420),interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop,f"{title} t={row['t']:.3f} current={row['body'].get('detected',False)}",(8,24),0,.5,(0,220,255),1);panels.append(crop)
        frame=np.hstack(panels);video.write(frame)
        if k==70:cv2.imwrite(str(out/'prediction_preview.png'),frame)
    video.release()
    rpm=read(out/'rpm/icp_results.json');errors=', '.join(f"{100*r['relative_error']:.1f}%" for r in rpm['synthetic'])
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>转速配准与FRED检出改善</title><style>body{{font:17px/1.7 system-ui;max-width:1120px;margin:32px auto;color:#203044;padding:0 24px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:10px;border:1px solid #d5dce5;text-align:left}}h1,h2{{line-height:1.3}}img,video{{max-width:100%}}.note{{padding:16px;background:#fff2d9}}code,pre{{background:#eef3f7}}a{{color:#156f91}}</style>
    <h1>转速配准与FRED检出改善</h1><p>2026-10-31 · 阶段158 · 仍未全面达到25px／持续有效30Hz。</p>
    <p class="note">FRED原比例只统计有真值目标的时刻，漏失确实存在。新候选显著改善小目标整机框；预测源时刻均值有改善，实际输出时间的精度尚未稳定同步改善。自采侧视32.85→39.01px，滞后P95约963ms，未作全场景默认。原算法保留。</p>
    <h2>已落实的优化</h2><p><code>--recall30</code>：已找到的小目标（近期框最长边≤192px），且没有已确认五圆模型／最近9次源更新没有几何证据时，NN使用完整当前33.333ms事件包，每源时刻一次当前NN。冷搜索、大目标、已有几何继续20ms与原异步节奏。光流和桨盘几何仍20ms；原生时间戳、周期和二力模型不变。不增加等待窗口、不更换或训练权重。</p>
    <pre>python run_fast_structure.py --raw "events.raw" --output "新的目录" --start 20 --duration 4 --sdk-avx2 --structure-guided-nn --recall30</pre><p>原工作配置仍为<code>--realtime30</code>。净力补充<code>--position-net-force</code>继续可选，不是机械推力测量。</p>
    <h2>完整RAW结果：不把检测和预测混为一项</h2><table><tr><th>片段</th><th>当前图像输出／有标注时刻</th><th>IoU≥0.5比例，含漏失</th><th>0.4s源时隙误差均值／px</th><th>预测源时隙覆盖</th></tr>{''.join(table)}</table><p>源评测每绝对5Hz时隙取第一个有效输出，跨版本可能选择不同起点；不能将各自均值差当纯算法因果收益。<a href="final_summary.json">逐项结果</a>另存严格相同旧源时刻对照；人工框只用于事后评分。</p>
    <h2>实际输出＋0.4s：保留退化</h2><table><tr><th>片段</th><th>共同查询数</th><th>相同查询误差／px</th><th>新运行滞后P95</th><th>有效发布／全部心跳</th></tr>{''.join(wall)}</table><p>FRED9同查询仍退化，不能宣传预测全面提高。30Hz发布不等于每秒30个有效新观测。FRED7/9均无可用桨盘结构，改进来自定位输入，不能称作五圆或推力优势。最后的近期几何保护在这两段所有源时刻均不触发，保护补丁没有改变其分支决策。</p>
    <h2>预测对照</h2><p>同源时刻：绿色框／十字为当前位置／0.4s未来真值，紫框为当前定位，黄色为未来轨迹。视口用真值居中仅方便观看；真值未进入推理。此视频不是实际发布延迟演示。</p><video controls src="FRED9_prediction_comparison.mp4" poster="prediction_preview.png"></video>
    <h2>为何增加事件窗</h2><p>同模型同图像时刻，完整33ms包的合格NN框从FRED0 20/94→34/94、FRED7 58/120→84/120、FRED9 43/118→68/118。它是独立当前帧对照，不是上面包含光流的运行系统比例。也有丢失原本成功的帧，下图一并保留。</p><details><summary>查看成功与失败的原生ON/OFF对照</summary><img src="window_changes.png"></details>
    <h2>EV-Tach小试：本轮不替换测速器</h2><p>独立编写的有限点数、椭圆矫正、双向时空ICP粗到细适配，不是作者实现或完整复现。3000/6000/9000RPM合成事件的相对误差约{errors}；低离散度仍可能有偏差。一个清晰真实33ms窗口的4个桨均未通过局部一致性门槛，每桨177–219ms；部分／侧视没有可矫正椭圆。本轮没有更准确RPM的证据，没有把该估计送入推力，因而没有预测提升。该失败只说明本地轻量适配未成功，不否定作者论文。<a href="rpm/icp_results.json">原始结果</a></p><p>既有净力候选继续保留：在本轮首版固定小目标同步刷新下，FRED7实际0.4s源均值74.60→71.64px，120帧当前框、几何和运动状态逐项相同，等于固定回放结果；只改变无合格桨力／光流时的未知净加速度。不是转速精度改善，亦非最终收窄冷启动后的配置成绩，不自动启用。</p>
    <h2>复用与未采用路径</h2><ul><li><a href="https://arxiv.org/html/2209.02205v1">EV-Tach论文</a>：配准测速思路。</li><li><a href="https://github.com/obss/sahi">SAHI官方代码</a>／<a href="https://arxiv.org/abs/2202.06934">论文</a>：复用项目已有单ROI放大适配；本次未取得可靠收益，不部署大规模全帧切片。</li><li><a href="https://github.com/radimspetlik/effqpdet">effqpdet官方C++实现</a>：检测、椭圆、频率参考，非即插即用的本机权重。</li><li><a href="https://github.com/uzh-rpg/RVT">RVT官方实现</a>：事件时序表示参考，模型及训练任务不同，本次未复制或重训。</li></ul><p>双窗口先20ms再33ms补检新增测量，但额外推理扰动预算调度，FRED7只61/120、95.83px；未采用。首版未知目标强制同步使清晰2s仅1/60发布有效；已增加冷启动和近期几何保护，所有失败目录保留。原已发布ZIP、模型和受保护项目没有改动，没有新依赖安装。</p>
    <h2>下一步</h2><p>优先处理“当前定位更好却未稳定改善实际输出＋0.4s”的速度／加速度观测与输出年龄问题；保留固定检测节奏。FRED7仍有约180px源误差P95、FRED9共同墙钟查询仍退化，不能仅凭框变准宣布完成。自采侧视和清晰段细节见<a href="final_summary.json">汇总数据</a>。数据已用于开发、上游训练可能覆盖FRED，不是盲测或正式前沿排名。</p></html>'''
    (out/'index.html').write_text(text,'utf-8')
    print(json.dumps({k:dict(matched=v['same_old_source_times']) for k,v in summary.items() if k in configs},indent=2))


if __name__=='__main__':main(sys.argv[1])
